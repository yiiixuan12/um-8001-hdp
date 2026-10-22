"""Portable project paths; saved scientific values are never recalculated here."""
import copy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "outputs/single-gpu-study"
STATE_ROOT = Path(os.environ.get("HDP_STATE_ROOT", ROOT / "local/state")).expanduser().resolve()
UPSTREAM_ROOT = Path(os.environ.get("HDP_UPSTREAM_ROOT", ROOT / "work/hdp")).expanduser().resolve()
LEGACY_ROOTS = (
    "/mnt/c/Users/User/Documents/Codex/2026-10-20/https-github-com-dyson-ai-hdp",
    "C:/Users/User/Documents/Codex/2026-10-20/https-github-com-dyson-ai-hdp",
    "/mnt/d/Coding/HDP-Motivation", "D:/Coding/HDP-Motivation",
)
STATE_PREFIX = "/opt/hdp/experiments/nbp-pilot"


def resolve_path(value):
    value = str(value).replace("\\", "/")
    for legacy in LEGACY_ROOTS:
        if value == legacy or value.startswith(legacy + "/"):
            value = value[len(legacy):].lstrip("/")
            break
    if value == STATE_PREFIX or value.startswith(STATE_PREFIX + "/"):
        return STATE_ROOT / value[len(STATE_PREFIX):].lstrip("/")
    if value == "local/state" or value.startswith("local/state/"):
        return STATE_ROOT / value[len("local/state"):].lstrip("/")
    value = value.replace("/correction-control-dense-plans/", "/grasp-source-plans/")
    value = value.replace("/open_microwave-stage-substitution-val-artifacts/", "/grasp-source-predictions/")
    if value.startswith(("outputs/", "work/", "reports/", "local/", "scripts/")):
        return ROOT / value
    return Path(value)


def _map(value, transform):
    if isinstance(value, dict):
        return {k: _map(v, transform) for k, v in value.items()}
    if isinstance(value, list):
        return [_map(v, transform) for v in value]
    return transform(value) if isinstance(value, str) else value


def portable_record(value):
    def portable(text):
        normalized = text.replace("\\", "/")
        for prefix, replacement in ((ROOT.as_posix(), ""), (STATE_ROOT.as_posix(), "local/state"),
                                    (STATE_PREFIX, "local/state"), *[(r, "") for r in LEGACY_ROOTS]):
            if normalized == prefix or normalized.startswith(prefix + "/"):
                tail = normalized[len(prefix):].lstrip("/")
                normalized = f"{replacement}/{tail}".strip("/")
                break
        normalized = normalized.replace("/correction-control-dense-plans/", "/grasp-source-plans/")
        normalized = normalized.replace("/open_microwave-stage-substitution-val-artifacts/", "/grasp-source-predictions/")
        return normalized if normalized != text.replace("\\", "/") else text
    return _map(value, portable)


def load_json(path):
    raw = json.loads(resolve_path(path).read_text(encoding="utf-8-sig"))
    def expanded(text):
        normalized = text.replace("\\", "/")
        if normalized.startswith(("outputs/", "work/", "reports/", "local/", "scripts/", STATE_PREFIX, *LEGACY_ROOTS)):
            return str(resolve_path(text))
        return text
    return _map(raw, expanded)


def same_experiment_config(first, second):
    """Only machine locations may differ; reject all scientific config changes."""
    a, b = copy.deepcopy(first), copy.deepcopy(second)
    for item in (a, b):
        item.pop("hdp_root", None)
        item.pop("state_root", None)
    return a == b
