from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_string(
    value: str,
    workspace: Path,
) -> str:
    value = value.replace(
        "${WORKSPACE}",
        str(workspace),
    )

    def replace(
        match: re.Match[str],
    ) -> str:
        key = match.group(1)

        if key == "WORKSPACE":
            return str(workspace)

        return os.environ.get(
            key,
            match.group(0),
        )

    return _ENV_PATTERN.sub(
        replace,
        value,
    )


def _expand(
    value: Any,
    workspace: Path,
) -> Any:
    if isinstance(value, str):
        return _expand_string(
            value,
            workspace,
        )

    if isinstance(value, list):
        return [
            _expand(item, workspace)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            key: _expand(item, workspace)
            for key, item in value.items()
        }

    return value


def load_server_config(
    config_path: str | Path,
    workspace: str | Path,
) -> dict[str, dict[str, Any]]:
    """
    Load servers.json and expand
    ${WORKSPACE} and environment variables.
    """

    config_path = Path(config_path)

    workspace = Path(
        workspace
    ).resolve()

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw = json.load(handle)

    servers = raw.get(
        "mcpServers",
        raw,
    )

    if not isinstance(
        servers,
        dict,
    ):
        raise ValueError(
            "servers.json must contain "
            "an 'mcpServers' object."
        )

    expanded = _expand(
        servers,
        workspace,
    )

    unresolved = []

    for name, cfg in expanded.items():
        serialized = json.dumps(cfg)

        unresolved.extend(
            f"{name}: {token}"
            for token in _ENV_PATTERN.findall(
                serialized
            )
            if token != "WORKSPACE"
        )

    if unresolved:
        raise RuntimeError(
            "Missing environment variable(s): "
            + ", ".join(
                sorted(set(unresolved))
            )
        )

    return expanded