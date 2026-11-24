#!/usr/bin/env python3
"""Publish to Endless TV from a GPU box (Python 3.8+, standard library only), so the box can be thrown away after.

The server and its upload token come from the environment, never the command line:
    export ENDLESS_TV_URL=https://<your app>.up.railway.app
    export ENDLESS_TV_TOKEN=<the server's INGEST_TOKEN>

    publish.py channel ID [--number N] [--name NAME] [--genre G] [--blurb TEXT] [--schedule FILE.json] [--logo FILE] [--poster FILE]
    publish.py clip ID FILE.mp4 [--title T] [--prompt TEXT | --prompt-file F] [--seed N] [--sequence N] [--run RUN] [--generate-s S]
    publish.py session ID DIR      the clips of an H3 Live session (clip_*.mp4 in session.json's order), and its numbers as a run
    publish.py run FILE.json [--id RUN]
    publish.py seed FILE.json [--media DIR]   channels, logos, posters and clips described in a seed file (seed/channels.json)
    publish.py delete-clip ID CLIP | delete-channel ID | list
"""
import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

TIMEOUT = 600


class PublishError(Exception):
    pass


def server():
    url = os.environ.get("ENDLESS_TV_URL", "").rstrip("/")
    if not url:
        raise PublishError("set ENDLESS_TV_URL to the server, e.g. https://endless-tv.up.railway.app")
    return url


def request(method, path, body=None, content_type=None, headers=None, auth=True):
    """The server's JSON answer (None for 204)."""
    all_headers = dict(headers or {})
    if auth:
        token = os.environ.get("ENDLESS_TV_TOKEN", "")
        if not token:
            raise PublishError("set ENDLESS_TV_TOKEN to the server's INGEST_TOKEN")
        all_headers["Authorization"] = f"Bearer {token}"
    if content_type:
        all_headers["Content-Type"] = content_type
    req = urllib.request.Request(server() + path, data=body, method=method, headers=all_headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            raw = response.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:300]
        raise PublishError(f"{method} {path}: HTTP {error.code} {detail}") from None
    except urllib.error.URLError as error:
        raise PublishError(f"{method} {path}: {error.reason}") from None


def send_json(method, path, payload):
    return request(method, path, json.dumps(payload).encode(), "application/json")


def send_file(method, path, file, headers=None):
    with open(file, "rb") as handle:
        data = handle.read()
    kind = mimetypes.guess_type(file)[0] or "application/octet-stream"
    return request(method, path, data, kind, headers)


def channel_path(channel_id, *rest):
    return "/api/channels/" + "/".join(urllib.parse.quote(part, safe="") for part in (channel_id, *rest))


def put_channel(channel_id, fields=None, logo=None, poster=None):
    fields = {key: value for key, value in (fields or {}).items() if value is not None}
    result = send_json("PUT", channel_path(channel_id), fields)
    for kind, file in (("logo", logo), ("poster", poster)):
        if file:
            result = send_file("PUT", channel_path(channel_id, kind), file)
    return result["channel"]


def put_clip(channel_id, file, title=None, prompt=None, seed=None, sequence=None, run=None, generate_s=None):
    query = {key: value for key, value in (("title", title), ("seed", seed), ("sequence", sequence), ("run", run), ("generate_s", generate_s)) if value is not None}
    headers = {"X-Clip-Prompt": urllib.parse.quote(prompt[:6000])} if prompt else None
    path = channel_path(channel_id, "clips") + ("?" + urllib.parse.urlencode(query) if query else "")
    return send_file("POST", path, file, headers)["clip"]


def put_run(payload, run_id=None):
    if run_id:
        payload = {**payload, "id": run_id}
    return send_json("POST", "/api/runs", payload)["run"]


def publish_session(channel_id, folder):
    """Every clip of an H3 Live session folder, in order, then its summary as a run."""
    summary_path = os.path.join(folder, "session.json")
    with open(summary_path) as handle:
        summary = json.load(handle)
    run_id = "s-" + os.path.basename(os.path.normpath(folder)).lower().replace("_", "-")[:38]
    run = put_run({"kind": "h3-live-session", "channel": channel_id, **{k: v for k, v in summary.items() if k != "records"}}, run_id)
    sent = []
    for record in summary.get("records", []):
        if "error" in record or not record.get("mp4"):
            continue
        file = os.path.join(folder, record["mp4"])
        if not os.path.exists(file):  # a rolling window deleted it
            continue
        sent.append(put_clip(channel_id, file, title=f"Clip {record['clip'] + 1}", prompt=record.get("prompt"), seed=record.get("seed"),
                             run=run["id"], generate_s=record.get("generate_s")))
    return run, sent


def publish_seed(seed_file, media_dir):
    with open(seed_file) as handle:
        seed = json.load(handle)
    media = media_dir or os.path.dirname(os.path.abspath(seed_file))
    found = lambda name: name and os.path.exists(os.path.join(media, name)) and os.path.join(media, name)  # noqa: E731
    for channel in seed["channels"]:
        fields = {key: channel.get(key) for key in ("number", "name", "genre", "blurb", "schedule")}
        missing = [name for name in [channel.get("logo"), channel.get("poster"), *channel.get("clips", [])] if name and not found(name)]
        if missing:
            print(f"  {channel['id']}: not found in {media}: {', '.join(missing)}", file=sys.stderr)
        put_channel(channel["id"], fields, found(channel.get("logo")), found(channel.get("poster")))
        clips = [put_clip(channel["id"], found(name), title=channel.get("name")) for name in channel.get("clips", []) if found(name)]
        print(f"  {channel['id']}: channel {channel.get('number')}, {len(clips)} clip(s)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Publish to Endless TV (server and token from ENDLESS_TV_URL and ENDLESS_TV_TOKEN).")
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("channel")
    c.add_argument("id")
    c.add_argument("--number", type=int)
    c.add_argument("--name")
    c.add_argument("--genre")
    c.add_argument("--blurb")
    c.add_argument("--schedule", help="a JSON file: a list of {title, episode, blurb, badges, minutes, isNew}")
    c.add_argument("--logo")
    c.add_argument("--poster")
    k = commands.add_parser("clip")
    k.add_argument("id")
    k.add_argument("file")
    k.add_argument("--title")
    k.add_argument("--prompt")
    k.add_argument("--prompt-file")
    k.add_argument("--seed", type=int)
    k.add_argument("--sequence", type=float)
    k.add_argument("--run")
    k.add_argument("--generate-s", type=float)
    s = commands.add_parser("session")
    s.add_argument("id")
    s.add_argument("dir")
    r = commands.add_parser("run")
    r.add_argument("file")
    r.add_argument("--id")
    e = commands.add_parser("seed")
    e.add_argument("file")
    e.add_argument("--media", help="folder with the files the seed names (default: the seed file's folder)")
    d = commands.add_parser("delete-clip")
    d.add_argument("id")
    d.add_argument("clip")
    x = commands.add_parser("delete-channel")
    x.add_argument("id")
    commands.add_parser("list")
    args = parser.parse_args(argv)

    try:
        if args.command == "channel":
            schedule = None
            if args.schedule:
                with open(args.schedule) as handle:
                    schedule = json.load(handle)
            channel = put_channel(args.id, {"number": args.number, "name": args.name, "genre": args.genre, "blurb": args.blurb, "schedule": schedule},
                                  args.logo, args.poster)
            print(f"channel {channel['id']}: {channel['name']} ({len(channel['clips'])} clips)")
        elif args.command == "clip":
            prompt = args.prompt
            if args.prompt_file:
                with open(args.prompt_file) as handle:
                    prompt = handle.read()
            clip = put_clip(args.id, args.file, args.title, prompt, args.seed, args.sequence, args.run, args.generate_s)
            print(f"clip {clip['id']}: {clip['duration']:.2f} s at {clip['url']}")
        elif args.command == "session":
            run, sent = publish_session(args.id, args.dir)
            print(f"run {run['id']}: {len(sent)} clip(s) on {args.id}")
        elif args.command == "run":
            with open(args.file) as handle:
                run = put_run(json.load(handle), args.id)
            print(f"run {run['id']}")
        elif args.command == "seed":
            publish_seed(args.file, args.media)
        elif args.command == "delete-clip":
            request("DELETE", channel_path(args.id, "clips", args.clip))
            print(f"deleted clip {args.clip}")
        elif args.command == "delete-channel":
            request("DELETE", channel_path(args.id))
            print(f"deleted channel {args.id}")
        else:
            for channel in request("GET", "/api/channels", auth=False)["channels"]:
                print(f"{channel['number']:>5}  {channel['id']:<24} {len(channel['clips']):>4} clips  {channel['name']}")
    except (PublishError, OSError, ValueError, KeyError) as error:
        print(f"publish.py: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
