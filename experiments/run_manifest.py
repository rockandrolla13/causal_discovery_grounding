"""JSON conversion and YAML run-manifest writing for experiment outputs."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def json_default(value: Any) -> Any:
    """
    Convert numpy and pandas scalar values into JSON-compatible values.

    Args:
        value: Object passed to ``json.dump`` or ``json.dumps``.

    Returns:
        JSON-compatible representation.
    """
    if isinstance(value, tuple):
        return [json_default(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _manifest_safe_value(value: Any) -> Any:
    """
    Convert nested experiment values into simple YAML-serializable values.

    Args:
        value: Arbitrary experiment metadata or configuration value.

    Returns:
        Value composed only of mappings, lists, strings, numbers, booleans, and
        nulls.
    """

    if is_dataclass(value) and not isinstance(value, type):
        return _manifest_safe_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _manifest_safe_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_manifest_safe_value(item) for item in value]
    if isinstance(value, list):
        return [_manifest_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return json_default(value)
    return json_default(value)


def _yaml_scalar(value: Any) -> str:
    """
    Render one scalar value in conservative YAML syntax.

    Args:
        value: Scalar metadata value.

    Returns:
        YAML scalar text.
    """

    value = _manifest_safe_value(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return json.dumps(str(value))
        return json.dumps(value)
    return json.dumps(str(value))


def _append_yaml_value(lines: list[str], value: Any, indent: int) -> None:
    """
    Append a nested value to a YAML document under an existing key.

    Args:
        lines: Mutable output lines.
        value: Nested metadata value.
        indent: Number of spaces to prepend.
    """

    prefix = " " * indent
    value = _manifest_safe_value(value)
    if isinstance(value, Mapping):
        if not value:
            lines.append(f"{prefix}{{}}")
            return
        for key, item in value.items():
            key_text = str(key)
            if isinstance(item, str) and "\n" in item:
                lines.append(f"{prefix}{key_text}: |")
                for line in item.splitlines():
                    lines.append(f"{prefix}  {line}")
            elif isinstance(item, Mapping):
                lines.append(f"{prefix}{key_text}:")
                _append_yaml_value(lines, item, indent + 2)
            elif isinstance(item, list):
                if item:
                    lines.append(f"{prefix}{key_text}:")
                    _append_yaml_value(lines, item, indent + 2)
                else:
                    lines.append(f"{prefix}{key_text}: []")
            else:
                lines.append(f"{prefix}{key_text}: {_yaml_scalar(item)}")
        return
    if isinstance(value, list):
        if not value:
            lines.append(f"{prefix}[]")
            return
        for item in value:
            if isinstance(item, (Mapping, list)):
                lines.append(f"{prefix}-")
                _append_yaml_value(lines, item, indent + 2)
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return
    lines.append(f"{prefix}{_yaml_scalar(value)}")


def _resolve_manifest_path(manifest_path: str | Path | None) -> Path | None:
    """
    Resolve a user-supplied cluster manifest path against common repo roots.

    Args:
        manifest_path: Optional absolute or relative manifest path.

    Returns:
        Existing resolved path when found, otherwise the best-effort path.
    """

    if manifest_path is None or str(manifest_path) == "":
        return None
    path = Path(manifest_path).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        Path(__file__).resolve().parents[1] / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def write_run_manifest(
    output_dir: str | Path,
    *,
    experiment_name: str,
    script_path: str,
    config: Mapping[str, Any] | Any,
    argv: Sequence[str] | None = None,
    cluster_manifest_path: str | Path | None = None,
    filename: str = "run_manifest.yml",
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """
    Materialize a run-level YAML manifest beside experiment outputs.

    Args:
        output_dir: Directory where the manifest should be written.
        experiment_name: Stable experiment identifier.
        script_path: Python entrypoint path used by the job.
        config: Resolved experiment configuration.
        argv: Command-line arguments received by the entrypoint.
        cluster_manifest_path: Optional Kubernetes YAML template to embed.
        filename: Output manifest filename.
        extra: Optional experiment-specific metadata.

    Returns:
        Path to the written manifest.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    source_path = _resolve_manifest_path(cluster_manifest_path)
    source_info: dict[str, Any] = {
        "requested_path": str(cluster_manifest_path)
        if cluster_manifest_path is not None
        else None,
        "resolved_path": str(source_path) if source_path is not None else None,
        "exists": bool(source_path is not None and source_path.exists()),
    }
    if source_path is not None and source_path.exists():
        source_info["contents"] = source_path.read_text(encoding="utf-8")

    payload: dict[str, Any] = {
        "experiment": experiment_name,
        "script": script_path,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": {
            "argv": list(argv or ()),
        },
        "config": _manifest_safe_value(config),
        "cluster_manifest": source_info,
    }
    if extra:
        payload["extra"] = _manifest_safe_value(extra)

    lines: list[str] = []
    _append_yaml_value(lines, payload, 0)
    manifest_output_path = output_path / filename
    manifest_output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_output_path

