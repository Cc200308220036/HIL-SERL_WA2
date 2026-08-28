#!/usr/bin/env python3
"""Build/compare R13 manifests. Shared Actor/Learner script, no ROS."""

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
    from hilserl_wa2.experiments.r10_protocol import network_config_hash, sha256_file, source_tree_manifest
    from hilserl_wa2.experiments.r13_protocol import (
        ACTION_DIM,
        END_EPISODE,
        PROTOCOL_VERSION,
        TRANSITION_SCHEMA_VERSION,
    )
    from hilserl_wa2.experiments.task_config import load_task
    from hilserl_wa2.envs.contracts import WA2EnvContract

    task = load_task(args.task)
    contract = WA2EnvContract.from_yaml(task.contract_path)
    rows, source_hash = source_tree_manifest(repo)
    demo = Path(args.demo_pkl).expanduser().resolve()
    wheel = Path(args.agentlace_wheel).expanduser().resolve() if args.agentlace_wheel else (
        repo / "artifacts" / "wheels" / "agentlace-0.1.3-py3-none-any.whl"
    )
    if not wheel.is_file():
        raise SystemExit(f"R13_MANIFEST: FAIL — missing agentlace wheel {wheel}")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "transition_schema_version": TRANSITION_SCHEMA_VERSION,
        "role": args.role,
        "task_id": task.task_id,
        "exp_name": task.exp_name,
        "config_bundle_hash": task.config_bundle_hash(),
        "network_config_hash": network_config_hash(args.network_config),
        "space_hash": build_space_signature(task, args.role, grasp_action=True)["space_hash"],
        "params_tree_signature": args.params_tree_signature,
        "agentlace_version": importlib.metadata.version("agentlace"),
        "agentlace_wheel_sha256": sha256_file(wheel),
        "source_tree_sha256": source_hash,
        "demo_pkl_sha256": sha256_file(demo),
        "action_dim": ACTION_DIM,
        "end_episode": bool(END_EPISODE),
        "action_scale": float(args.action_scale),
        "policy_hz": float(contract.policy_hz),
        "servo_hz": float(contract.control_hz),
        "servo_ticks_per_action": int(contract.servo_ticks_per_action),
        "discount": float(task.discount),
        "classifier_consecutive_n": int(task.classifier_consecutive_n),
        "source_files": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def compare(left: Path, right: Path) -> int:
    a = json.loads(left.read_text(encoding="utf-8"))
    b = json.loads(right.read_text(encoding="utf-8"))
    ignored = {"role", "source_files", "params_tree_signature"}
    keys = sorted((set(a) | set(b)) - ignored)
    mismatch = {k: [a.get(k), b.get(k)] for k in keys if a.get(k) != b.get(k)}
    if mismatch:
        print(json.dumps(mismatch, indent=2, sort_keys=True))
        print("R13_MANIFEST_MATCH: FAIL")
        return 1
    print("R13_MANIFEST_MATCH: PASS")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo")
    parser.add_argument("--task", default="bottle_pick")
    parser.add_argument("--network-config")
    parser.add_argument("--agentlace-wheel", default="")
    parser.add_argument("--demo-pkl", dest="demo_pkl")
    parser.add_argument("--params-tree-signature", default="")
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--role", choices=("actor", "learner"))
    parser.add_argument("--output")
    parser.add_argument("--compare", nargs=2, metavar=("LEFT", "RIGHT"))
    args = parser.parse_args()
    if args.compare:
        raise SystemExit(compare(Path(args.compare[0]), Path(args.compare[1])))
    required = (args.repo, args.network_config, args.demo_pkl, args.role, args.output)
    if not all(required):
        parser.error("build mode requires --repo --network-config --demo-pkl --role --output")
    manifest = build(args)
    print(json.dumps({k: v for k, v in manifest.items() if k != "source_files"}, indent=2, sort_keys=True))
    print("R13_MANIFEST_BUILD: PASS")


if __name__ == "__main__":
    main()
