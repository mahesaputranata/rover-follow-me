#!/usr/bin/env python3
"""
fetch_and_summarize.py - PC side: ask the Pi for a run, download it, summarize it.

Usage:
    python fetch_and_summarize.py [--host raspberrypi.local] [--run last] [--out C:\\Users\\you\\runs] [--list] [--no-open]

Talks to rover_server.py over the same websocket Buddy uses (ws://<host>:8765), asks for
{"action": "get_run", "run_id": "last"}, downloads the .tgz over HTTP (port in the reply),
extracts it into --out and runs summarize_run.py on it.

Needs: pip install websockets rosbags<0.10 numpy matplotlib
"""
import argparse
import asyncio
import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_run as S   # noqa: E402


async def ask(host, payload, port=8765):
    async with websockets.connect(f"ws://{host}:{port}", max_size=None) as ws:
        await ws.send(json.dumps(payload))
        # the server may push other messages (telemetry acks); wait for the one that answers us
        for _ in range(20):
            reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            if payload["action"] == "list_runs" and "runs" in reply:
                return reply
            if payload["action"] == "get_run" and ("url_path" in reply or "error" in reply):
                return reply
        raise RuntimeError("no matching reply from the server")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="raspberrypi.local")
    ap.add_argument("--run", default="last")
    ap.add_argument("--out", default=str(Path.home() / "runs"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    if args.list:
        reply = asyncio.run(ask(args.host, {"action": "list_runs", "limit": 20}))
        for r in reply["runs"]:
            print(f"{r['run_id']:<55} {r.get('mode','?'):<10} {r.get('duration_s','?')} s  {r['size_mb']} MB"
                  + ("  [recording]" if r.get("recording") else ""))
        return

    reply = asyncio.run(ask(args.host, {"action": "get_run", "run_id": args.run}))
    if "error" in reply:
        sys.exit(f"Pi says: {reply['error']}")
    url = f"http://{args.host}:{reply['http_port']}{reply['url_path']}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tgz = out / Path(reply["url_path"]).name
    print(f"downloading {url} ({reply['size_mb']} MB)")
    urllib.request.urlretrieve(url, tgz)
    with tarfile.open(tgz) as tar:
        tar.extractall(out)
    run_dir = out / reply["run_id"]

    data, duration = S.load_run(run_dir)
    summary, zone = S.analyze(data, duration)
    summary["run"] = run_dir.name
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    S.print_report(summary)
    S.make_plots(data, zone, run_dir / "plots.png", run_dir.name)
    print(f"\nrun folder: {run_dir}")
    if not args.no_open and hasattr(os, "startfile"):
        os.startfile(str(run_dir / "plots.png"))
        os.startfile(str(run_dir / "summary.json"))


if __name__ == "__main__":
    main()
