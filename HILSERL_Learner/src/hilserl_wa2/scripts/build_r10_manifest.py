#!/usr/bin/env python3
"""Build/compare R10 manifests without importing ROS."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path


def _bootstrap(repo: Path) -> None:
    for path in (
        repo / "src",
        repo / "src" / "hil-serl-main" / "examples",
        repo / "src" / "hil-serl-main" / "serl_launcher",
    ):
        sys.path.insert(0, str(path))


def build(args) -> dict:
    repo = Path(args.repo).resolve()
    _bootstrap(repo)
    from hilserl_wa2.experiments.env_factory import build_space_signature
    from hilserl_wa2.experiments.r10_protocol import (
        PROTOCOL_VERSION,
        TRANSITION_SCHEMA_VERSION,
        network_config_hash,
        sha256_file,
        source_tree_manifest,
    )
    from hilserl_wa2.experiments.task_config import load_task

    task = load_task(args.task)
    rows, source_hash = source_tree_manifest(repo)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "transition_schema_version": TRANSITION_SCHEMA_VERSION,
        "role": args.role,
        "task_id": task.task_id,
        "exp_name": task.exp_name,
        "config_bundle_hash": task.config_bundle_hash(),
        "network_config_hash": network_config_hash(args.network_config),
        "space_hash": build_space_signature(task, args.role)["space_hash"],
        "params_tree_signature": args.params_tree_signature,
        "agentlace_version": importlib.metadata.version("agentlace"),
        "agentlace_wheel_sha256": sha256_file(args.agentlace_wheel),
        "source_tree_sha256": source_hash,
        "source_files": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def compare(left: Path, right: Path) -> int:
    a = json.loads(left.read_text(encoding="utf-8"))
    b = json.loads(right.read_text(encoding="utf-8"))
    ignored = {"role", "source_files"}
    keys = sorted((set(a) | set(b)) - ignored)
    mismatch = {k: [a.get(k), b.get(k)] for k in keys if a.get(k) != b.get(k)}
    if mismatch:
        print(json.dumps(mismatch, indent=2, sort_keys=True))
        print("R10_MANIFEST_MATCH: FAIL")
        return 1
    print("R10_MANIFEST_MATCH: PASS")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo")
    parser.add_argument("--task", default="bottle_pick")
    parser.add_argument("--network-config")
    parser.add_argument("--agentlace-wheel")
    parser.add_argument("--params-tree-signature", default="")
    parser.add_argument("--role", choices=("actor", "learner"))
    parser.add_argument("--output")
    parser.add_argument("--compare", nargs=2, metavar=("LEFT", "RIGHT"))
    args = parser.parse_args()
    if args.compare:
        raise SystemExit(compare(Path(args.compare[0]), Path(args.compare[1])))
    required = (args.repo, args.network_config, args.agentlace_wheel, args.role, args.output)
    if not all(required):
        parser.error("build mode requires --repo --network-config --agentlace-wheel --role --output")
    manifest = build(args)
    print(json.dumps({k: v for k, v in manifest.items() if k != "source_files"}, indent=2, sort_keys=True))
    print("R10_MANIFEST_BUILD: PASS")


if __name__ == "__main__":
    main()

