# SPDX-License-Identifier: Apache-2.0
"""Durable state for the gateway: API keys, usage, job ownership.

Why this exists: the first deployment kept all three in process memory, which
is a billing bug, not an inelegance. A restart mid-generation orphaned the
job -> owner mapping (the customer paid for a job they could no longer fetch)
and reset every quota counter to zero.

Backends: any SQLAlchemy URL. Railway injects DATABASE_URL for its Postgres
addon; tests and local runs default to SQLite. Keys are stored as SHA-256
HASHES -- a database dump must not be a credential dump, and auth needs only
an equality check.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
from typing import Any

import sqlalchemy as sa

_metadata = sa.MetaData()

api_keys = sa.Table(
    "api_keys", _metadata,
    sa.Column("key_hash", sa.String(64), primary_key=True),
    sa.Column("label", sa.String(200), nullable=False),
    sa.Column("daily_limit", sa.Integer, nullable=True),   # NULL -> gateway default
    sa.Column("revoked_at", sa.DateTime, nullable=True),
    sa.Column("created_at", sa.DateTime, nullable=False),
)

usage = sa.Table(
    "usage", _metadata,
    sa.Column("key_hash", sa.String(64), primary_key=True),
    sa.Column("day", sa.String(10), primary_key=True),      # YYYY-MM-DD
    sa.Column("count", sa.Integer, nullable=False, server_default="0"),
)

#: One row per submitted job. generate_s is filled in when a completion is
#: observed -- it is the number per-second billing would be computed from.
jobs = sa.Table(
    "jobs", _metadata,
    sa.Column("id", sa.String(80), primary_key=True),
    sa.Column("key_hash", sa.String(64), nullable=False, index=True),
    sa.Column("resolution", sa.String(20), nullable=False),
    sa.Column("duration_s", sa.Integer, nullable=False),
    sa.Column("seed", sa.Integer, nullable=True),
    sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
    sa.Column("generate_s", sa.Float, nullable=True),
    #: Submit to first observed completion. This is what a caller actually
    #: waits -- generate_s omits dispatch, model load, decode and transfer,
    #: which is why estimates built on it under-promised badly.
    sa.Column("wall_s", sa.Float, nullable=True),
    sa.Column("created_at", sa.DateTime, nullable=False),
)

#: One row per adventure scene. Unlike customer clips (delivery-buffered and
#: deleted), these ARE the product: generated once, kept forever, shared by
#: every player. The mp4 bytes live in the row -- 14 scenes at a few MB each
#: is nothing to Postgres, and it inherits Railway's durability with no new
#: storage service.
adventure_scenes = sa.Table(
    "adventure_scenes", _metadata,
    sa.Column("id", sa.String(40), primary_key=True),
    sa.Column("story", sa.String(40), nullable=False, index=True),
    sa.Column("position", sa.Integer, nullable=False),   # encounter order
    sa.Column("depth", sa.Integer, nullable=False),
    sa.Column("title", sa.String(200), nullable=False),
    sa.Column("prompt", sa.Text, nullable=False),
    sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
    sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
    #: FL2V continuity: this scene's clip starts from parent's last frame,
    #: so a child is only renderable once its parent is done.
    sa.Column("parent_id", sa.String(40), nullable=True),
    sa.Column("job_id", sa.String(80), nullable=True),
    sa.Column("seed", sa.Integer, nullable=True),
    sa.Column("generate_s", sa.Float, nullable=True),
    sa.Column("error", sa.String(500), nullable=True),
    sa.Column("video", sa.LargeBinary, nullable=True),
    sa.Column("updated_at", sa.DateTime, nullable=False),
)

#: Waitlist signups from the adventure pages. Email is the primary key, so a
#: repeat signup is a quiet no-op instead of a duplicate row, and the table
#: cannot grow past one row per address.
waitlist = sa.Table(
    "waitlist", _metadata,
    sa.Column("email", sa.String(254), primary_key=True),
    sa.Column("source", sa.String(40), nullable=False),
    sa.Column("created_at", sa.DateTime, nullable=False),
)

_engine: sa.Engine | None = None


def hash_key(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        # Local/dev fallback. On Railway, attach the Postgres addon so this is
        # never used there -- container filesystems are ephemeral.
        return "sqlite:////tmp/gateway.db"
    # Railway/Heroku hand out postgres://; SQLAlchemy 2 requires postgresql://
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def engine() -> sa.Engine:
    global _engine
    if _engine is None:
        _engine = sa.create_engine(_url(), pool_pre_ping=True, future=True)
        _metadata.create_all(_engine)
        # create_all never alters an existing table; add columns introduced
        # after first deploy by hand and ignore "already exists".
        for ddl in ("ALTER TABLE jobs ADD COLUMN wall_s FLOAT",
                    "ALTER TABLE adventure_scenes ADD COLUMN parent_id VARCHAR(40)"):
            try:
                with _engine.begin() as cx:
                    cx.execute(sa.text(ddl))
            except sa.exc.DBAPIError:
                pass
    return _engine


def reset_for_tests() -> None:
    """Drop the cached engine so a test can re-point DATABASE_URL."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------

def seed_keys(mapping: dict[str, str]) -> None:
    """Upsert env-provided keys (GATEWAY_KEYS) so the old config path keeps
    working. Seeding never un-revokes: a key revoked in the DB stays revoked
    even if it is still sitting in the env var."""
    if not mapping:
        return
    with engine().begin() as cx:
        for token, label in mapping.items():
            kh = hash_key(token)
            row = cx.execute(sa.select(api_keys.c.key_hash)
                             .where(api_keys.c.key_hash == kh)).first()
            if row is None:
                cx.execute(api_keys.insert().values(
                    key_hash=kh, label=label, created_at=_now()))


def ensure_key(identity: str, label: str,
               daily_limit: int | None = None) -> dict[str, Any] | None:
    """Auto-provision a non-secret identity ("clerk:user_abc", "ip:1.2.3.4")
    as a key row on first sight, so quota and job ownership work identically
    for session users, visitors, and sk_ API keys. Returns {key_hash,
    daily_limit}, or None when the row is revoked -- revoking the row is how
    an account (or address) gets banned. An existing row's daily_limit wins
    over the argument, so a hand-edited override in the DB sticks."""
    kh = hash_key(identity)
    with engine().connect() as cx:
        row = cx.execute(sa.select(api_keys.c.revoked_at, api_keys.c.daily_limit)
                         .where(api_keys.c.key_hash == kh)).first()
    if row is not None:
        if row[0] is not None:
            return None
        return {"key_hash": kh, "daily_limit": row[1]}
    try:
        with engine().begin() as cx:
            cx.execute(api_keys.insert().values(
                key_hash=kh, label=label, daily_limit=daily_limit,
                created_at=_now()))
    except sa.exc.IntegrityError:
        pass                    # two first requests raced; the row exists now
    return {"key_hash": kh, "daily_limit": daily_limit}


def lookup_key(token: str) -> dict[str, Any] | None:
    """The key's row, or None when unknown or revoked."""
    kh = hash_key(token)
    with engine().connect() as cx:
        row = cx.execute(sa.select(api_keys).where(
            api_keys.c.key_hash == kh,
            api_keys.c.revoked_at.is_(None))).mappings().first()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------

def try_consume_quota(key_hash: str, day: str, limit: int) -> bool:
    """Atomically count one generation against the day, False if over limit."""
    with engine().begin() as cx:
        query = sa.select(usage.c.count).where(
            usage.c.key_hash == key_hash, usage.c.day == day)
        if cx.dialect.name == "postgresql":
            # Row lock so two concurrent submits cannot both read count == limit-1
            # and both pass. SQLite serialises writers on its own.
            query = query.with_for_update()
        row = cx.execute(query).first()
        current = row[0] if row else 0
        if current >= limit:
            return False
        if row is None:
            cx.execute(usage.insert().values(key_hash=key_hash, day=day, count=1))
        else:
            cx.execute(usage.update().where(
                usage.c.key_hash == key_hash, usage.c.day == day)
                .values(count=usage.c.count + 1))
    return True


def usage_today(key_hash: str, day: str) -> int:
    with engine().connect() as cx:
        row = cx.execute(sa.select(usage.c.count).where(
            usage.c.key_hash == key_hash, usage.c.day == day)).first()
    return row[0] if row else 0


def refund_quota(key_hash: str, day: str) -> None:
    """A failed submit must not bill the quota."""
    with engine().begin() as cx:
        cx.execute(usage.update().where(
            usage.c.key_hash == key_hash, usage.c.day == day,
            usage.c.count > 0).values(count=usage.c.count - 1))


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

def record_job(job_id: str, key_hash: str, *, resolution: str, duration_s: int,
               seed: int | None) -> None:
    """Idempotent for the same owner: a retried submit that lands on the same
    job id must not 500 after the quota was already consumed. A duplicate id
    owned by a DIFFERENT key is an anomaly worth failing loudly on."""
    with engine().begin() as cx:
        existing = cx.execute(sa.select(jobs.c.key_hash)
                              .where(jobs.c.id == job_id)).first()
        if existing is not None:
            if existing[0] != key_hash:
                raise RuntimeError(f"job id collision across keys: {job_id}")
            return
        cx.execute(jobs.insert().values(
            id=job_id, key_hash=key_hash, resolution=resolution,
            duration_s=duration_s, seed=seed, created_at=_now()))


# ---------------------------------------------------------------------------
# adventure scenes
# ---------------------------------------------------------------------------

def adventure_seed(story: str, scenes: list[dict[str, Any]]) -> None:
    """Insert missing scene rows; existing rows (and their videos) are never
    touched, so re-deploys are free and finished work is never re-done.

    One deliberate exception: a row whose parent_id is NULL while the
    definition has one predates the continuity migration -- its clip was
    rendered without the parent's last frame. Backfill the parent and requeue
    it so the whole tree is continuous. Runs exactly once per row."""
    with engine().begin() as cx:
        have = {r[0]: r[1] for r in cx.execute(
            sa.select(adventure_scenes.c.id, adventure_scenes.c.parent_id)
            .where(adventure_scenes.c.story == story)).all()}
        for scene in scenes:
            if scene["id"] not in have:
                cx.execute(adventure_scenes.insert().values(
                    id=scene["id"], story=story, position=scene["position"],
                    depth=scene["depth"], title=scene["title"],
                    prompt=scene["prompt"], parent_id=scene.get("parent_id"),
                    updated_at=_now()))
            elif have[scene["id"]] is None and scene.get("parent_id"):
                cx.execute(adventure_scenes.update()
                           .where(adventure_scenes.c.id == scene["id"])
                           .values(parent_id=scene["parent_id"],
                                   status="queued", attempts=0,
                                   updated_at=_now()))


def adventure_requeue_stale(story: str, older_than_s: int = 900) -> None:
    """A 'rendering' row whose app died mid-job would block forever; put it
    back in the queue after a generous timeout."""
    cutoff = _now() - _dt.timedelta(seconds=older_than_s)
    with engine().begin() as cx:
        cx.execute(adventure_scenes.update().where(
            adventure_scenes.c.story == story,
            adventure_scenes.c.status == "rendering",
            adventure_scenes.c.updated_at < cutoff,
        ).values(status="queued", updated_at=_now()))


def adventure_next(story: str, max_attempts: int = 3) -> dict[str, Any] | None:
    """The next scene to render: strictly by encounter order. Failed scenes
    come back around until max_attempts so one flake cannot hole the story,
    then stay failed for a human to look at."""
    with engine().connect() as cx:
        row = cx.execute(
            sa.select(adventure_scenes.c.id, adventure_scenes.c.prompt,
                      adventure_scenes.c.attempts)
            .where(adventure_scenes.c.story == story,
                   adventure_scenes.c.status.in_(("queued", "failed")),
                   adventure_scenes.c.attempts < max_attempts)
            .order_by(adventure_scenes.c.position).limit(1)).first()
    return {"id": row[0], "prompt": row[1], "attempts": row[2]} if row else None


def adventure_claim(story: str, max_attempts: int = 3) -> dict[str, Any] | None:
    """Atomically take the next scene in encounter order and flip it to
    'rendering'. The claim is a compare-and-swap on (status, attempts), so
    parallel render lanes never double-claim on ANY engine -- a plain
    select-then-update raced under SQLite's deferred transactions and let
    two lanes bump the same scene twice. A child is only claimable once its
    parent's clip exists (its start frame comes from it) or the parent is
    terminally failed (render without continuity rather than never). The
    attempt is spent at claim time -- a lane that dies still burned its try."""
    parent = adventure_scenes.alias("parent")
    parent_done = sa.or_(
        adventure_scenes.c.parent_id.is_(None),
        sa.exists(sa.select(parent.c.id).where(
            parent.c.id == adventure_scenes.c.parent_id,
            sa.or_(parent.c.status == "ready",
                   sa.and_(parent.c.status == "failed",
                           parent.c.attempts >= max_attempts)))))
    for _ in range(10):                    # races are rare; retries settle them
        with engine().connect() as cx:
            row = cx.execute(
                sa.select(adventure_scenes.c.id, adventure_scenes.c.prompt,
                          adventure_scenes.c.attempts)
                .where(adventure_scenes.c.story == story,
                       adventure_scenes.c.status.in_(("queued", "failed")),
                       adventure_scenes.c.attempts < max_attempts,
                       parent_done)
                .order_by(adventure_scenes.c.position).limit(1)).first()
        if row is None:
            return None
        with engine().begin() as cx:
            won = cx.execute(adventure_scenes.update().where(
                adventure_scenes.c.id == row[0],
                adventure_scenes.c.status.in_(("queued", "failed")),
                adventure_scenes.c.attempts == row[2],
            ).values(status="rendering", attempts=row[2] + 1,
                     updated_at=_now())).rowcount
        if won:
            return {"id": row[0], "prompt": row[1]}
    return None


def adventure_any_rendering(story: str) -> bool:
    with engine().connect() as cx:
        row = cx.execute(sa.select(adventure_scenes.c.id).where(
            adventure_scenes.c.story == story,
            adventure_scenes.c.status == "rendering").limit(1)).first()
    return row is not None


def adventure_mark(scene_id: str, status: str, *, job_id: str | None = None,
                   seed: int | None = None, generate_s: float | None = None,
                   error: str | None = None, video: bytes | None = None,
                   bump_attempts: bool = False) -> None:
    values: dict[str, Any] = {"status": status, "updated_at": _now()}
    if job_id is not None:
        values["job_id"] = job_id
    if seed is not None:
        values["seed"] = seed
    if generate_s is not None:
        values["generate_s"] = generate_s
    if error is not None:
        values["error"] = error[:500]
    if video is not None:
        values["video"] = video
    if bump_attempts:
        values["attempts"] = adventure_scenes.c.attempts + 1
    with engine().begin() as cx:
        cx.execute(adventure_scenes.update()
                   .where(adventure_scenes.c.id == scene_id).values(**values))


def adventure_status(story: str) -> dict[str, dict[str, Any]]:
    """Per-scene status WITHOUT the video bytes -- the page polls this."""
    with engine().connect() as cx:
        rows = cx.execute(
            sa.select(adventure_scenes.c.id, adventure_scenes.c.status,
                      adventure_scenes.c.seed, adventure_scenes.c.attempts)
            .where(adventure_scenes.c.story == story)).all()
    return {r[0]: {"status": r[1], "seed": r[2], "attempts": r[3]} for r in rows}


def adventure_video(scene_id: str) -> bytes | None:
    with engine().connect() as cx:
        row = cx.execute(sa.select(adventure_scenes.c.video).where(
            adventure_scenes.c.id == scene_id,
            adventure_scenes.c.status == "ready")).first()
    return row[0] if row else None


def recent_wall_times(limit: int = 200) -> dict[tuple[str, int], list[float]]:
    """The last `limit` completions' wall-clock seconds (submit to observed
    completion), grouped by (resolution, duration_s). The estimate the page
    advertises is computed from these rather than hand-measured constants,
    so it tracks what a caller actually waits as the fleet changes."""
    with engine().connect() as cx:
        rows = cx.execute(
            sa.select(jobs.c.resolution, jobs.c.duration_s, jobs.c.wall_s)
            .where(jobs.c.wall_s.is_not(None))
            .order_by(jobs.c.created_at.desc()).limit(limit)).all()
    out: dict[tuple[str, int], list[float]] = {}
    for res, dur, wall in rows:
        out.setdefault((res, dur), []).append(wall)
    return out


def job_owned_by(job_id: str, key_hash: str) -> bool:
    with engine().connect() as cx:
        row = cx.execute(sa.select(jobs.c.id).where(
            jobs.c.id == job_id, jobs.c.key_hash == key_hash)).first()
    return row is not None


def waitlist_add(email: str, source: str) -> bool:
    """Record one signup. True if the address is new, False if it was already
    on the list -- both are success to the caller."""
    try:
        with engine().begin() as cx:
            cx.execute(waitlist.insert().values(
                email=email, source=source[:40], created_at=_now()))
        return True
    except sa.exc.IntegrityError:
        return False


def waitlist_count() -> int:
    with engine().connect() as cx:
        return int(cx.execute(sa.select(sa.func.count()).select_from(waitlist)).scalar_one())


def mark_job(job_id: str, status: str, generate_s: float | None = None) -> None:
    values: dict[str, Any] = {"status": status}
    if generate_s is not None:
        values["generate_s"] = generate_s
    with engine().begin() as cx:
        if status == "completed":
            # First observation of the completion stamps the wall time; the
            # status endpoint re-marks on every poll and must not creep it up.
            row = cx.execute(sa.select(jobs.c.created_at, jobs.c.wall_s)
                             .where(jobs.c.id == job_id)).first()
            if row is not None and row[1] is None and row[0] is not None:
                values["wall_s"] = max(0.0, (_now() - row[0]).total_seconds())
        cx.execute(jobs.update().where(jobs.c.id == job_id).values(**values))
