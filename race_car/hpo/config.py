"""Utilities for applying flat, dotted-key configuration overrides.

The training scripts build a dataclass configuration (for example
``SacZopTrainerConfig``).  These helpers let the HPO driver change individual
fields without duplicating the whole ``create_cfg`` function.  Overrides use
dotted paths::

    gamma: 0.995
    lr_q: 1.0e-4
    critic_mlp.hidden_dims: [128, 128, 128]
    log.tensorboard_logger: false

The same YAML file can be passed to ``run_sac_zop.py --config_overrides`` to
re-run a configuration found by the HPO study.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def to_plain(value: Any) -> Any:
    """Convert tuples and other simple containers to YAML-friendly Python types."""
    if isinstance(value, Mapping):
        return {str(key): to_plain(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        # numpy scalars (and similar) expose .item()
        return value.item()
    return value


def load_overrides(path: str | Path) -> dict[str, Any]:
    """Load a flat YAML file containing dotted-key configuration overrides."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a mapping in '{path}', got {type(data).__name__}."
        )
    return {str(key): value for key, value in data.items()}


def _resolve_parent(cfg: Any, dotted_key: str) -> tuple[Any, str]:
    parts = dotted_key.split(".")
    current = cfg
    for part in parts[:-1]:
        if not hasattr(current, part):
            raise AttributeError(
                f"Configuration has no attribute '{part}' while resolving "
                f"'{dotted_key}'."
            )
        current = getattr(current, part)
    return current, parts[-1]


def apply_overrides(cfg: Any, overrides: Mapping[str, Any]) -> None:
    """Apply dotted-key overrides to a dataclass configuration in place.

    Args:
        cfg: The configuration object, e.g. ``SacZopTrainerConfig``.
        overrides: Mapping from dotted attribute paths to new values.

    Raises:
        AttributeError: If an override path does not exist in the configuration.
        TypeError: If a ``hidden_dims`` override is not a sequence.
    """
    for dotted_key, value in overrides.items():
        parent, leaf = _resolve_parent(cfg, dotted_key)
        if not hasattr(parent, leaf):
            raise AttributeError(
                f"Configuration has no attribute '{leaf}' while applying "
                f"'{dotted_key}'."
            )

        # The MLP configs expect a sequence of ints (or None).  YAML lists are
        # fine at runtime, but tuples match the dataclass defaults more closely.
        if leaf == "hidden_dims" and value is not None:
            if not isinstance(value, (list, tuple)):
                raise TypeError(
                    f"'{dotted_key}' must be a list/tuple of ints or null, got "
                    f"{type(value).__name__}."
                )
            value = tuple(int(dim) for dim in value)

        setattr(parent, leaf, value)


def dump_overrides(overrides: Mapping[str, Any], path: str | Path) -> None:
    """Write dotted-key overrides to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            to_plain(dict(overrides)),
            file,
            sort_keys=True,
            default_flow_style=False,
        )


if __name__ == "__main__":
    from dataclasses import dataclass, field

    @dataclass
    class Nested:
        hidden_dims: tuple[int, ...] | None = (64, 64)
        activation: str = "relu"

    @dataclass
    class Demo:
        gamma: float = 0.99
        nested: Nested = field(default_factory=Nested)

    demo = Demo()
    apply_overrides(
        demo,
        {
            "gamma": 0.999,
            "nested.hidden_dims": [128, 128],
            "nested.activation": "tanh",
        },
    )
    assert demo.gamma == 0.999
    assert demo.nested.hidden_dims == (128, 128)
    assert demo.nested.activation == "tanh"

    try:
        apply_overrides(demo, {"does.not.exist": 1})
    except AttributeError:
        pass
    else:
        raise AssertionError("Expected AttributeError for unknown override path")

    print("config override self-test passed")
