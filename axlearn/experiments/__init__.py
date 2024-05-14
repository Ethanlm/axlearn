# Copyright © 2023 Apple Inc.

"""AXLearn experiments."""
from importlib import import_module
from typing import Dict

from absl import logging

from axlearn.common.config import similar_names
from axlearn.experiments.trainer_config_utils import TrainerConfigFn


def _load_trainer_configs(
    config_module: str, *, optional: bool = False
) -> Dict[str, TrainerConfigFn]:
    try:
        module = import_module(config_module)
        logging.info(f"Ethan Debug found module: {module}")
        return module.named_trainer_configs()
    except (ImportError, AttributeError):
        if not optional:
            raise
        logging.warning(
            "Missing dependencies for %s but it's marked optional -- skipping.", config_module
        )
    return {}


def get_named_trainer_config(config_name: str, *, config_module: str) -> TrainerConfigFn:
    """Looks up TrainerConfigFn by config name.

    Args:
        config_name: Candidate config name.
        config_module: Config module name.

    Returns:
        A TrainerConfigFn corresponding to the config name.

    Raises:
        KeyError: Error containing the message to show to the user.
    """
    config_map = _load_trainer_configs(config_module)
    logging.info(f"ethan debug config_map: {config_map}")
    if callable(config_map):
        logging.info(f"ethan debug callable config_map: {config_name}")
        return config_map(config_name)

    try:
        res = config_map[config_name]
        logging.info(f"ethan debug not callable config_map: {res.__class__}")
        logging.info(f"ethan debug not callable config_map: {res}")
        return config_map[config_name]
    except KeyError as e:
        similar = similar_names(config_name, set(config_map.keys()))
        if similar:
            message = f"Unrecognized config {config_name}; did you mean [{', '.join(similar)}]"
        else:
            message = (
                f"Unrecognized config {config_name} under {config_module}; "
                f"Please make sure that the following conditions are met:\n"
                f"    1. {config_module} can be imported; "
                f"    2. {config_module} defines `named_trainer_configs()`; "
                f"    3. `named_trainer_configs()` returns a dict with '{config_name}' as a key."
            )
        raise KeyError(message) from e
