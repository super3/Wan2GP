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


def job_owned_by(job_id: str, key_hash: str) -> bool:
    with engine().connect() as cx:
        row = cx.execute(sa.select(jobs.c.id).where(
            jobs.c.id == job_id, jobs.c.key_hash == key_hash)).first()
    return row is not None


def mark_job(job_id: str, status: str, generate_s: float | None = None) -> None:
    values: dict[str, Any] = {"status": status}
    if generate_s is not None:
        values["generate_s"] = generate_s
    with engine().begin() as cx:
        cx.execute(jobs.update().where(jobs.c.id == job_id).values(**values))
