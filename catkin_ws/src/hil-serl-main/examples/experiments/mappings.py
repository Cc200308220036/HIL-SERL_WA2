try:
    from experiments.ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
    from experiments.usb_pickup_insertion.config import TrainConfig as USBPickupInsertionTrainConfig
    from experiments.object_handover.config import TrainConfig as ObjectHandoverTrainConfig
    from experiments.egg_flip.config import TrainConfig as EggFlipTrainConfig

    _FRANKA_MAPPING = {
        "ram_insertion": RAMInsertionTrainConfig,
        "usb_pickup_insertion": USBPickupInsertionTrainConfig,
        "object_handover": ObjectHandoverTrainConfig,
        "egg_flip": EggFlipTrainConfig,
    }
except ImportError:
    # WA2 Orin deployment does not install franka_env; keep Franka keys optional.
    _FRANKA_MAPPING = {}

from experiments.wa2.config import discover_wa2_mapping

CONFIG_MAPPING = {
    **_FRANKA_MAPPING,
    **discover_wa2_mapping(),
}
