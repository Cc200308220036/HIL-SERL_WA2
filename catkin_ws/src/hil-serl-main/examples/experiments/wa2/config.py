"""Generic WA2 TrainConfig. Tasks come from YAML; do not copy this file per task."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

_CATKIN_SRC = Path(__file__).resolve().parents[4]
if str(_CATKIN_SRC) not in sys.path:
    sys.path.insert(0, str(_CATKIN_SRC))

from experiments.config import DefaultTrainingConfig
from hilserl_wa2.experiments.env_factory import make_wa2_environment
from hilserl_wa2.experiments.task_config import (
    discover_task_ids,
    exp_name_for_task,
    load_task,
)


class WA2TrainConfig(DefaultTrainingConfig):
    """One class for all WA2 tasks. Bind a task_id at construction."""

    def __init__(self, task_id: str):
        self.task = load_task(task_id)
        self.task_id = self.task.task_id
        self.image_keys = list(self.task.image_keys)
        self.proprio_keys = list(self.task.proprio_keys)
        self.classifier_keys = (
            list(self.task.classifier_keys)
            if self.task.classifier_keys is not None
            else None
        )
        self.agent = self.task.agent
        self.setup_mode = self.task.setup_mode
        self.encoder_type = self.task.encoder_type
        self.discount = self.task.discount
        self.batch_size = self.task.batch_size
        self.max_steps = self.task.max_steps
        self.replay_buffer_capacity = self.task.replay_buffer_capacity
        self.random_steps = self.task.random_steps
        self.training_starts = self.task.training_starts
        self.steps_per_update = self.task.steps_per_update
        self.buffer_period = self.task.buffer_period
        self.checkpoint_period = self.task.checkpoint_period

    def get_environment(
        self,
        fake_env: bool = False,
        save_video: bool = False,
        classifier: bool = False,
    ):
        return make_wa2_environment(
            self.task,
            fake_env=fake_env,
            save_video=save_video,
            classifier=classifier,
        )

    def process_demos(self, demo):
        """R8 stub: demos are consumed as already-wrapped transitions in R11."""

        return demo


def _bound_ctor(task_id: str):
    def ctor() -> WA2TrainConfig:
        return WA2TrainConfig(task_id=task_id)

    ctor.__name__ = f"WA2TrainConfig_{task_id}"
    ctor.__qualname__ = ctor.__name__
    return ctor


def discover_wa2_mapping() -> Dict[str, Any]:
    """Scan configs/tasks/*.yaml → {wa2_<task_id>: ctor}."""

    mapping: Dict[str, Any] = {}
    for task_id in discover_task_ids():
        mapping[exp_name_for_task(task_id)] = _bound_ctor(task_id)
    if "wa2" in mapping:
        raise RuntimeError("bare exp_name 'wa2' must not be registered")
    return mapping
