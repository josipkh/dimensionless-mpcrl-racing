"""Minimal dotted-key configuration helpers used by the HPO driver.

A YAML file maps attribute paths to values, e.g.::

    tau: 0.005
    lr_q: 0.0003
    log.verbose: false

``apply_overrides`` sets those attributes on a dataclass configuration such as
``SacZopTrainerConfig``.  The same file can be passed to
``run_sac_zop.py --config_overrides`` to re-run a tuned configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_overrides(path: str | Path) -> dict[str, Any]:
    """Load a flat YAML mapping of dotted attribute paths to values."""
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in '{path}', got {type(data).__name__}.")
    return {str(key): value for key, value in data.items()}


def apply_overrides(cfg: Any, overrides: Mapping[str, Any]) -> None:
    """Apply dotted-key overrides to a dataclass configuration in place."""
    for dotted_key, value in overrides.items():
        *parents, leaf = dotted_key.split(".")
        target = cfg
        for part in parents:
            target = getattr(target, part)
        if not hasattr(target, leaf):
            raise AttributeError(f"Unknown configuration key '{dotted_key}'.")
        if leaf == "hidden_dims" and value is not None:
            value = tuple(int(dim) for dim in value)
        setattr(target, leaf, value)


def dump_overrides(overrides: Mapping[str, Any], path: str | Path) -> None:
    """Write dotted-key overrides to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plain = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in overrides.items()
    }
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(plain, file, sort_keys=True)


if __name__ == "__main__":
    from dataclasses import dataclass, field

    @dataclass
    class Inner:
        hidden_dims: tuple[int, ...] | None = (64, 64)

    @dataclass
    class Config:
        tau: float = 0.005
        inner: Inner = field(default_factory=Inner)

    cfg = Config()
    apply_overrides(cfg, {"tau": 0.01, "inner.hidden_dims": [128, 128]})
    assert cfg.tau == 0.01
    assert cfg.inner.hidden_dims == (128, 128)

    try:
        apply_overrides(cfg, {"does.not.exist": 1})
    except AttributeError:
        pass
    else:
        raise AssertionError("Expected AttributeError for an unknown key")

    print("override self-test passed")
