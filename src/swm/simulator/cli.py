from __future__ import annotations

from collections.abc import Collection


def parse_cli(
    argv: list[str] | None,
    positional_count: int,
    value_options: Collection[str] = (),
    flag_options: Collection[str] = (),
) -> tuple[list[str], dict[str, str], set[str]]:
    argv = [] if argv is None else argv
    positionals: list[str] = []
    values: dict[str, str] = {}
    flags: set[str] = set()
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            positionals.append(token)
        else:
            name, separator, inline = token[2:].partition("=")
            if name in flag_options and not separator:
                flags.add(name)
            elif name in value_options:
                if not separator:
                    index += 1
                    if index == len(argv):
                        raise SystemExit(f"--{name} requires a value")
                    inline = argv[index]
                values[name] = inline
            else:
                raise SystemExit(f"unknown option: {token}")
        index += 1
    if len(positionals) != positional_count:
        raise SystemExit(f"expected {positional_count} positional arguments")
    return positionals, values, flags


def build_vlm_advisor(options: dict[str, str], flags: set[str]):
    if "vlm-mapping" not in flags:
        return None
    from pathlib import Path

    from .alignment import VLMAlignmentAdvisor

    return VLMAlignmentAdvisor(
        model=options.get("vlm-model", "qwen3.7-plus"),
        api_key_env=options.get("vlm-api-key-env", "BOYUE_API_KEY"),
        base_url=options.get("vlm-base-url", "https://apicz.boyuerichdata.com/v1"),
        cache_dir=Path(
            options.get(
                "vlm-cache", ".cache/domain_logical_simulator/vlm_mapping_v5"
            )
        ),
        env_file=Path(options.get("env-file", ".env")),
        allow_network="vlm-cache-only" not in flags,
        require_complete=True,
        reasoning_effort=options.get("vlm-reasoning-effort"),
        json_mode="vlm-json-mode" in flags,
    )
