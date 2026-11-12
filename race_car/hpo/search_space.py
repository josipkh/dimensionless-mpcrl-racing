"""Search-space definitions and Optuna sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

SPACE_DIR = Path(__file__).resolve().parent / "search_spaces"

_SUPPORTED_KINDS = {"uniform", "loguniform", "int", "categorical"}


@dataclass(frozen=True)
class ParamSpec:
    """One hyperparameter in a search space.

    Attributes:
        name: Name reported to Optuna, e.g. ``"lr"``.
        kind: One of ``uniform``, ``loguniform``, ``int``, ``categorical``.
        targets: Trainer-config keys (dotted paths) that receive the sampled
            value.  Defaults to ``(name,)``.
        low: Lower bound for numeric distributions.
        high: Upper bound for numeric distributions.
        log: Whether an integer distribution is sampled logarithmically.
        step: Step size for integer distributions.
        choices: Primitive labels for categorical distributions.
        values: Optional actual values corresponding to ``choices``.  This lets
            a categorical dimension represent non-primitive values (e.g. tuples
            of hidden-layer widths) while Optuna only stores primitive labels.
    """

    name: str
    kind: str
    targets: tuple[str, ...]
    low: float | None = None
    high: float | None = None
    log: bool = False
    step: int | None = None
    choices: tuple[Any, ...] = ()
    values: tuple[Any, ...] = ()

    def value_for(self, choice: Any) -> Any:
        """Map a categorical label to the value written to the config."""
        if not self.values:
            return choice
        for label, value in zip(self.choices, self.values):
            if label == choice:
                return value
        raise KeyError(
            f"Unknown categorical label {choice!r} for parameter '{self.name}'."
        )

    @classmethod
    def from_dict(cls, name: str, spec: dict[str, Any]) -> "ParamSpec":
        kind = str(spec.get("type", "")).lower()
        if kind not in _SUPPORTED_KINDS:
            raise ValueError(
                f"Unsupported distribution type '{kind}' for '{name}'. "
                f"Choose one of {sorted(_SUPPORTED_KINDS)}."
            )

        targets = tuple(str(target) for target in spec.get("targets", (name,)))
        if not targets:
            raise ValueError(f"'{name}' must have at least one target.")

        common = {
            "name": name,
            "kind": kind,
            "targets": targets,
            "low": spec.get("low"),
            "high": spec.get("high"),
            "log": bool(spec.get("log", False)),
            "step": spec.get("step"),
            "choices": (),
            "values": (),
        }

        if kind in {"uniform", "loguniform", "int"}:
            if common["low"] is None or common["high"] is None:
                raise ValueError(
                    f"'{name}' ({kind}) requires both 'low' and 'high'."
                )
            if common["low"] >= common["high"]:
                raise ValueError(
                    f"'{name}' has low >= high: {common['low']} >= {common['high']}."
                )
        else:
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(
                    f"'{name}' (categorical) requires a non-empty 'choices' list."
                )
            labels = tuple(_make_primitive(choice, name) for choice in choices)
            if len(set(labels)) != len(labels):
                raise ValueError(f"'{name}' has duplicate categorical choices.")

            values = spec.get("values")
            if values is None:
                values_tuple = ()
            else:
                if not isinstance(values, dict):
                    raise ValueError(f"'{name}' 'values' must be a mapping.")
                missing = [label for label in labels if label not in values]
                if missing:
                    raise ValueError(
                        f"'{name}' 'values' is missing mappings for {missing}."
                    )
                values_tuple = tuple(values[label] for label in labels)

            common["choices"] = labels
            common["values"] = values_tuple

        return cls(**common)

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serialisable representation."""
        data: dict[str, Any] = {"type": self.kind, "targets": list(self.targets)}
        if self.kind == "categorical":
            data["choices"] = _to_plain(list(self.choices))
            if self.values:
                data["values"] = {
                    str(label): _to_plain(value)
                    for label, value in zip(self.choices, self.values)
                }
        else:
            data["low"] = self.low
            data["high"] = self.high
            if self.kind == "int":
                if self.log:
                    data["log"] = True
                if self.step is not None:
                    data["step"] = self.step
        return data


def _make_primitive(value: Any, parameter: str) -> Any:
    """Validate that a categorical choice can be stored by Optuna."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(
        f"Categorical choices for '{parameter}' must be primitive labels "
        "(None, bool, int, float, str). Use the optional 'values' mapping to "
        f"map labels to complex values; got {value!r}."
    )


def _to_plain(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain(val) for key, val in value.items()}
    return value


def resolve_space_path(space: str | Path) -> Path:
    """Resolve a preset name (e.g. ``full``) or a YAML path."""
    space = Path(space)
    if space.suffix in {".yaml", ".yml"}:
        if not space.is_file():
            raise FileNotFoundError(f"Search-space file not found: {space}")
        return space

    candidate = SPACE_DIR / f"{space}.yaml"
    if candidate.is_file():
        return candidate

    available = ", ".join(sorted(path.stem for path in SPACE_DIR.glob("*.yaml")))
    raise FileNotFoundError(
        f"Unknown search space '{space}'. Available presets: {available}. "
        "You can also pass a path to a YAML file."
    )


def load_space(space: str | Path) -> tuple[list[ParamSpec], Path]:
    """Load a search space from a preset name or YAML file.

    Returns:
        The parsed parameter specs and the resolved file path.
    """
    path = resolve_space_path(space)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict) or not data:
        raise ValueError(f"Search space '{path}' must be a non-empty mapping.")

    specs = [ParamSpec.from_dict(str(name), spec) for name, spec in data.items()]
    _validate_unique_targets(specs)
    return specs, path


def _validate_unique_targets(specs: Iterable[ParamSpec]) -> None:
    seen: dict[str, str] = {}
    for spec in specs:
        for target in spec.targets:
            if target in seen:
                raise ValueError(
                    f"Config key '{target}' is written by both '{seen[target]}' "
                    f"and '{spec.name}'."
                )
            seen[target] = spec.name


def dump_space(specs: Iterable[ParamSpec], path: str | Path) -> None:
    """Write a resolved search space to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {spec.name: spec.to_dict() for spec in specs}
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, default_flow_style=False)


def sample_params(trial: Any, specs: Iterable[ParamSpec]) -> dict[str, Any]:
    """Sample a configuration and expand it to dotted trainer-config keys."""
    sampled: dict[str, Any] = {}
    for spec in specs:
        if spec.kind == "categorical":
            label = trial.suggest_categorical(spec.name, list(spec.choices))
            value = spec.value_for(label)
        elif spec.kind == "uniform":
            value = trial.suggest_float(spec.name, spec.low, spec.high)
        elif spec.kind == "loguniform":
            value = trial.suggest_float(spec.name, spec.low, spec.high, log=True)
        elif spec.kind == "int":
            value = trial.suggest_int(
                spec.name,
                int(spec.low),
                int(spec.high),
                step=int(spec.step or 1),
                log=spec.log,
            )
        else:  # pragma: no cover - guarded by ParamSpec.from_dict
            raise ValueError(f"Unsupported distribution kind: {spec.kind}")

        for target in spec.targets:
            sampled[target] = _to_plain(value)
    return sampled


def expand_params(specs: Iterable[ParamSpec], params: dict[str, Any]) -> dict[str, Any]:
    """Expand Optuna's ``trial.params`` (keyed by parameter name) to config keys."""
    expanded: dict[str, Any] = {}
    for spec in specs:
        if spec.name not in params:
            continue
        value = params[spec.name]
        if spec.kind == "categorical":
            value = spec.value_for(value)
        for target in spec.targets:
            expanded[target] = _to_plain(value)
    return expanded


if __name__ == "__main__":
    specs, path = load_space("small")
    assert [spec.name for spec in specs] == ["lr", "gamma", "tau", "batch_size"]
    assert specs[0].targets == ("lr_q", "lr_pi", "lr_alpha")
    assert path.is_file()

    specs, _ = load_space("full")
    assert any(spec.name == "critic_mlp.hidden_dims" for spec in specs)

    import optuna

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    params = sample_params(trial, specs)
    assert "lr_q" in params and "lr_pi" in params and "lr_alpha" in params
    assert "critic_mlp.hidden_dims" in params
    expanded = expand_params(specs, trial.params)
    assert expanded["lr_q"] == expanded["lr_pi"] == expanded["lr_alpha"]
    print("search-space self-test passed")
