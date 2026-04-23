from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from mcvae import generate, train, train_diffusion
except ModuleNotFoundError as exc:
    if exc.name != "mcvae":
        raise
    # Allow running as `python cli.py` from inside `src/mcvae`.
    package_parent = Path(__file__).resolve().parent.parent
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))
    from mcvae import generate, train, train_diffusion


@dataclass(frozen=True)
class Operation:
    """Interactive CLI operation exposed in the main menu."""

    name: str
    description: str
    parser_builder: Callable[[], argparse.ArgumentParser]
    runner: Callable[[list[str]], None]
    subcommand: str | None = None


OPERATIONS: tuple[Operation, ...] = (
    Operation(
        name="Train VAE",
        description="Train a variational autoencoder model.",
        parser_builder=train.build_parser,
        runner=train.main,
    ),
    Operation(
        name="Train Latent Diffusion",
        description="Train diffusion in VAE latent space.",
        parser_builder=train_diffusion.build_parser,
        runner=train_diffusion.main,
    ),
    Operation(
        name="Sample Builds (VAE)",
        description="Generate new builds from the VAE prior.",
        parser_builder=generate.build_parser,
        runner=generate.main,
        subcommand="sample",
    ),
    Operation(
        name="Sample Builds (Diffusion)",
        description="Generate new builds via diffusion + VAE decoder.",
        parser_builder=generate.build_parser,
        runner=generate.main,
        subcommand="sample-diffusion",
    ),
    Operation(
        name="Reconstruct Builds",
        description="Reconstruct one build or a directory with VAE.",
        parser_builder=generate.build_parser,
        runner=generate.main,
        subcommand="reconstruct",
    ),
)


def _find_subparser(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    """Find a named subparser from an argparse parser."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparser = action.choices.get(name)
            if subparser is not None:
                return subparser
    raise ValueError(f"Subcommand parser not found: {name}")


def _is_bool_action(action: argparse.Action) -> bool:
    return isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))


def _display_action_name(action: argparse.Action) -> str:
    if action.option_strings:
        return action.option_strings[0]
    return action.dest


def _normalize_actions(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    actions: list[argparse.Action] = []
    for action in parser._actions:
        if action.dest in {"help", "command"}:
            continue
        if isinstance(action, argparse._SubParsersAction):
            continue
        if action.dest == argparse.SUPPRESS:
            continue
        actions.append(action)
    return actions


def _format_value(action: argparse.Action, value: Any) -> str:
    if value is None and action.required:
        return "<required>"
    return repr(value)


def _parse_yes_no(text: str) -> bool:
    normalized: str = text.strip().lower()
    if normalized in {"y", "yes", "true", "1"}:
        return True
    if normalized in {"n", "no", "false", "0"}:
        return False
    raise ValueError("Expected yes/no.")


def _parse_value(action: argparse.Action, raw: str) -> Any:
    if not raw.strip():
        raise ValueError("Value cannot be blank.")

    if not action.required and raw.strip().lower() == "none":
        return None

    if _is_bool_action(action):
        return _parse_yes_no(raw)

    converter = getattr(action, "type", None)
    value: Any = raw if converter is None else converter(raw)

    choices = getattr(action, "choices", None)
    if choices is not None and value not in choices:
        valid: str = ", ".join(str(choice) for choice in choices)
        raise ValueError(f"Invalid value. Choose one of: {valid}")
    return value


def _input_required(action: argparse.Action) -> Any:
    while True:
        label: str = _display_action_name(action)
        raw: str = input(f"Enter {label}: ").strip()
        if not raw:
            print("A value is required.")
            continue
        try:
            return _parse_value(action, raw)
        except Exception as exc:  # pragma: no cover - interactive error path
            print(f"Invalid value: {exc}")


def _edit_parameter(action: argparse.Action, current_value: Any) -> Any:
    label: str = _display_action_name(action)
    while True:
        if _is_bool_action(action):
            raw: str = input(
                f"Set {label} (y/n, blank keeps {current_value!r}): "
            ).strip()
            if not raw:
                return current_value
        else:
            raw = input(
                f"Set {label} (blank keeps {current_value!r}; 'none' clears optional): "
            ).strip()
            if not raw:
                return current_value
        try:
            return _parse_value(action, raw)
        except Exception as exc:  # pragma: no cover - interactive error path
            print(f"Invalid value: {exc}")


def _build_argv(
    *,
    subcommand: str | None,
    actions: list[argparse.Action],
    values: dict[str, Any],
) -> list[str]:
    argv: list[str] = []
    if subcommand is not None:
        argv.append(subcommand)

    for action in actions:
        value: Any = values[action.dest]

        if isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(action.option_strings[0])
            continue

        if isinstance(action, argparse._StoreFalseAction):
            if not value:
                argv.append(action.option_strings[0])
            continue

        if value is None and not action.required:
            continue

        if action.option_strings:
            argv.append(action.option_strings[0])
            argv.append(str(value))
        else:
            argv.append(str(value))

    return argv


def _prompt_operation() -> Operation | None:
    print("\n=== Minecraft Build CLI ===")
    for idx, operation in enumerate(OPERATIONS, start=1):
        print(f"{idx}. {operation.name} - {operation.description}")
    print("0. Exit")

    while True:
        raw: str = input("Choose an option number: ").strip()
        if raw == "0":
            return None
        if raw.isdigit():
            index: int = int(raw) - 1
            if 0 <= index < len(OPERATIONS):
                return OPERATIONS[index]
        print("Invalid choice. Enter one of the listed numbers.")


def _run_operation(operation: Operation) -> None:
    parser: argparse.ArgumentParser = operation.parser_builder()
    prompt_parser: argparse.ArgumentParser = (
        _find_subparser(parser, operation.subcommand) if operation.subcommand else parser
    )
    actions: list[argparse.Action] = _normalize_actions(prompt_parser)

    values: dict[str, Any] = {}
    for action in actions:
        default = None if action.required else action.default
        values[action.dest] = default

    print(f"\nSelected: {operation.name}")
    for action in actions:
        if action.required:
            values[action.dest] = _input_required(action)

    while True:
        print("\nParameters:")
        for idx, action in enumerate(actions, start=1):
            label: str = _display_action_name(action)
            current: str = _format_value(action, values[action.dest])
            print(f"{idx}. {label} = {current}")

        edit_raw: str = input(
            "Enter parameter number to edit (blank to continue): "
        ).strip()
        if not edit_raw:
            break
        if not edit_raw.isdigit():
            print("Enter a valid parameter number.")
            continue
        edit_idx: int = int(edit_raw) - 1
        if not (0 <= edit_idx < len(actions)):
            print("Enter a valid parameter number.")
            continue
        action = actions[edit_idx]
        values[action.dest] = _edit_parameter(action, values[action.dest])

    argv: list[str] = _build_argv(
        subcommand=operation.subcommand,
        actions=actions,
        values=values,
    )
    print("\nCommand preview:")
    print("  " + " ".join(["python -m", operation.runner.__module__, *argv]))

    while True:
        confirm_raw: str = input("Run this command? [Y/n]: ").strip()
        if not confirm_raw:
            break
        try:
            if _parse_yes_no(confirm_raw):
                break
            print("Cancelled.")
            return
        except ValueError:
            print("Please enter y or n.")

    operation.runner(argv)


def main() -> None:
    """Run the interactive project CLI."""
    while True:
        operation: Operation | None = _prompt_operation()
        if operation is None:
            print("Exiting.")
            return
        try:
            _run_operation(operation)
        except Exception as exc:  # pragma: no cover - interactive error path
            print(f"Command failed: {exc}")


if __name__ == "__main__":
    main()
