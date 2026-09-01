#!/bin/bash
set -euo pipefail

SOURCE_REPO=/home/naviai/ros_docker_test/catkin_ws/src/hil-serl-main
SOURCE_WHEEL=/home/naviai/ros_docker_test/catkin_ws/artifacts/wheels/agentlace-0.1.3-py3-none-any.whl

TARGET_ROOT=/home/naviai/hilserl_orin
TARGET_REPO=${TARGET_ROOT}/catkin_ws/src/hil-serl-main
TARGET_WHEEL_DIR=${TARGET_ROOT}/artifacts/wheels
TARGET_WHEEL=${TARGET_WHEEL_DIR}/agentlace-0.1.3-py3-none-any.whl

if [[ ! -f "${SOURCE_WHEEL}" ]]; then
    echo "Agentlace wheel not found: ${SOURCE_WHEEL}" >&2
    exit 1
fi

mkdir -p "${TARGET_ROOT}/catkin_ws/src" "${TARGET_WHEEL_DIR}"

# 只读取 ros_docker_test；所有新文件均写入 hilserl_orin。
if [[ -d "${TARGET_REPO}" ]]; then
    echo "Keeping existing independent source: ${TARGET_REPO}"
else
    if [[ ! -d "${SOURCE_REPO}" ]]; then
        echo "HIL-SERL source not found: ${SOURCE_REPO}" >&2
        exit 1
    fi
    cp -a "${SOURCE_REPO}" "${TARGET_REPO}"
    echo "Prepared independent HIL-SERL source: ${TARGET_REPO}"
fi

if [[ -f "${TARGET_WHEEL}" ]]; then
    echo "Keeping existing Agentlace wheel: ${TARGET_WHEEL}"
else
    install -m 0644 "${SOURCE_WHEEL}" "${TARGET_WHEEL}"
    echo "Prepared Agentlace wheel: ${TARGET_WHEEL}"
fi

(
    cd "${TARGET_WHEEL_DIR}"
    sha256sum "$(basename "${TARGET_WHEEL}")" > SHA256SUMS
)

echo "Original /home/naviai/ros_docker_test was not modified."
