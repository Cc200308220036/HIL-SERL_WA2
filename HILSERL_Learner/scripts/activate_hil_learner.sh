#!/usr/bin/env bash
# Source this file: source /home/cyw/orin_hilserl/HILSERL_Learner/scripts/activate_hil_learner.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "ERROR: source this script instead of executing it" >&2
  exit 2
fi

source /home/cyw/anaconda3/etc/profile.d/conda.sh
conda activate hil-learner

_HILSERL_ACTIVATE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export HILSERL_LEARNER_ROOT="$(cd -- "${_HILSERL_ACTIVATE_DIR}/.." && pwd -P)"
unset _HILSERL_ACTIVATE_DIR

if [[ "${HILSERL_LEARNER_ROOT}" != "/home/cyw/orin_hilserl/HILSERL_Learner" ]]; then
  echo "ERROR: unexpected Learner root: ${HILSERL_LEARNER_ROOT}" >&2
  return 2
fi
export PYTHONNOUSERSITE=1
export PYTHONPATH="${HILSERL_LEARNER_ROOT}/src:${HILSERL_LEARNER_ROOT}/src/hil-serl-main/examples:${HILSERL_LEARNER_ROOT}/src/hil-serl-main/serl_launcher"
unset JAX_PLATFORMS

export CUDA_ROOT=/usr/local/cuda
export XLA_FLAGS=--xla_gpu_autotune_level=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

# Prevent the login shell's ROS2 overlays from entering the Learner Python process.
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION

cd "$HILSERL_LEARNER_ROOT"
echo "HIL-SERL Learner activated: env=${CONDA_DEFAULT_ENV} root=${HILSERL_LEARNER_ROOT}"
