def get_config(exp_name: str):
    """Lazy load experiment configs to avoid importing hardware dependencies."""
    if exp_name == "ram_insertion":
        from experiments.ram_insertion.config import TrainConfig
        return TrainConfig
    elif exp_name == "usb_pickup_insertion":
        from experiments.usb_pickup_insertion.config import TrainConfig
        return TrainConfig
    elif exp_name == "object_handover":
        from experiments.object_handover.config import TrainConfig
        return TrainConfig
    elif exp_name == "egg_flip":
        from experiments.egg_flip.config import TrainConfig
        return TrainConfig
    elif exp_name == "cube_stacking_gym":
        from experiments.cube_stacking_gym.config import TrainConfig
        return TrainConfig
    elif exp_name == "cube_stacking_gym_rlpd":
        from experiments.cube_stacking_gym.config_rlpd import TrainConfig
        return TrainConfig
    else:
        raise ValueError(f"Unknown experiment name: {exp_name}")

# Legacy mapping for backward compatibility
CONFIG_MAPPING = {
    "ram_insertion": lambda: get_config("ram_insertion"),
    "usb_pickup_insertion": lambda: get_config("usb_pickup_insertion"),
    "object_handover": lambda: get_config("object_handover"),
    "egg_flip": lambda: get_config("egg_flip"),
    "cube_stacking_gym": lambda: get_config("cube_stacking_gym"),
    "cube_stacking_gym_rlpd": lambda: get_config("cube_stacking_gym_rlpd"),
}