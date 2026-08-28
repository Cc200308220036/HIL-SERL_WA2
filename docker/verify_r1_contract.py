#!/usr/bin/env python3
"""R1 Gate verifier: contract YAML + evidence files (no robot motion)."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

def _repo_root() -> Path:
    here = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
    for cand in (here.parent.parent if here.name.endswith(".py") else here, Path("/home/naviai/hilserl_orin"), Path.cwd()):
        if (cand / "configs/experiments/wa2_env_contract.yaml").is_file():
            return cand
    return Path("/home/naviai/hilserl_orin")


ROOT = _repo_root()
YAML_PATH = ROOT / "configs/experiments/wa2_env_contract.yaml"
DOC_PATH = ROOT / "docs/WA2Env接口契约.md"
SAMPLES = ROOT / "调试日志/阶段验收日志/r1_samples"


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def ok(msg: str) -> None:
    print(f"PASS: {msg}")


def count_msgs(path: Path) -> int:
    text = path.read_text(errors="ignore")
    return text.count("\n---")


def main() -> None:
    if yaml is None:
        fail("PyYAML not installed")

    if not YAML_PATH.is_file():
        fail(f"missing {YAML_PATH}")
    if not DOC_PATH.is_file():
        fail(f"missing {DOC_PATH}")

    data = yaml.safe_load(YAML_PATH.read_text())
    doc = DOC_PATH.read_text(encoding="utf-8")

    # --- core freezes ---
    if data.get("arm") != "left":
        fail("arm must be left")
    action = data["action"]
    if action.get("dim") != 6:
        fail("action.dim must be 6")
    if action.get("position_frame") != "base":
        fail("position_frame must be base")
    if action.get("rotation_frame") != "tool":
        fail("rotation_frame must be tool")
    for key in ("TBD", "TBD_H", "TBD_W", "TBD_tool_or_base"):
        dumped = yaml.safe_dump(data)
        if key in dumped:
            fail(f"YAML still contains {key}")
    ok("action/arm frames frozen (6D left, base+tool)")

    obs = data["observation"]
    for key, shape in {
        "tcp_pose": [7],
        "tcp_vel": [6],
        "joint_pos": [8],
        "hand_joints": [6],
    }.items():
        if obs["state"][key]["shape"] != shape:
            fail(f"state.{key}.shape expected {shape}")
    ok("state shapes match contract")

    head = obs["images"]["head"]
    wrist = obs["images"]["wrist"]
    if head["topic"] != "/zj_humanoid/sensor/realsense_head/color/image_raw":
        fail("head topic mismatch")
    if head.get("enabled") is not True:
        fail("head must be enabled")
    if head["shape"] != [128, 128, 3] or head["raw_shape"] != [720, 1280, 3]:
        fail("head shapes mismatch")
    if wrist["topic"] != "/zj_humanoid/sensor/left_wrist/image_raw":
        fail("wrist topic mismatch")
    # R1: wrist reserved; R6+: wrist.enabled=true with live topic
    if wrist.get("enabled") not in (True, False):
        fail("wrist.enabled must be bool")
    if wrist.get("missing_policy") != "zero_image":
        fail("wrist missing_policy must be zero_image")
    ok(f"camera keys: head enabled; wrist.enabled={wrist.get('enabled')}")

    if data["control"].get("sole_action_publisher") != "WA2Env":
        fail("sole_action_publisher must be WA2Env")
    reset_strategy = data["reset"].get("strategy")
    if reset_strategy not in ("manual_pose_tolerance", "scene_yaml_movej"):
        fail(f"reset strategy unexpected: {reset_strategy}")
    if data["episode"].get("max_steps") != 400:
        fail("max_steps must be 400")
    if data["fake_env"].get("touches_ros") is not False:
        fail("fake_env must not touch ROS")
    ok("reset/control/fake_env frozen")

    # doc must not be skeleton
    for bad in ("从盘点与评审填入", "从 r1_samples 填入"):
        if bad in doc:
            fail(f"contract doc still has placeholder: {bad}")
    if ("0.1.0" not in doc and "0.1.1" not in doc) or (
        "manual_pose_tolerance" not in doc and "人工" not in doc
    ):
        fail("contract doc incomplete")
    ok("human-readable contract filled")

    # evidence
    required_samples = [
        "tcp_pose_left_arm.txt",
        "joint_states.txt",
        "uplimb_state.txt",
        "hand_joint_states.txt",
        "tcp_speed_dual_arm.txt",
    ]
    for name in required_samples:
        path = SAMPLES / name
        if not path.is_file():
            fail(f"missing sample {path}")
        n = count_msgs(path)
        if n < 10:
            fail(f"{name} has only {n} messages (<10)")
    ok("state/hand/tcp_speed samples >=10 each")

    head_meta = SAMPLES / "head" / "head_image_meta.txt"
    head_png = SAMPLES / "head" / "head_sample.png"
    if not head_meta.is_file():
        fail("missing head image meta")
    if not head_png.is_file():
        fail(f"missing {head_png}")
    ok("head meta + sample png present")

    wrist_status = SAMPLES / "wrist" / "STATUS.md"
    wrist_info = SAMPLES / "wrist" / "info_left_wrist.txt"
    if not wrist_status.is_file():
        fail("missing wrist STATUS.md")
    if not wrist_info.is_file():
        fail("missing wrist info_left_wrist.txt")
    text = wrist_info.read_text(errors="ignore")
    if "None" not in text:
        fail("wrist info should show no publishers")
    ok("wrist documented as not publishing")

    hz_summary = SAMPLES / "hz_summary.md"
    if not hz_summary.is_file():
        fail("missing hz_summary.md")
    ok("hz_summary.md present")

    # space constructibility smoke (no gymnasium required)
    action_dim = action["dim"]
    img_shape = tuple(head["shape"])
    print(
        "SPACE_SUMMARY:",
        f"action=({action_dim},)",
        f"tcp_pose=(7,)",
        f"images.head={img_shape}",
        f"images.wrist={tuple(wrist['shape'])}",
    )
    print("R1 GATE: PASS")


if __name__ == "__main__":
    main()
