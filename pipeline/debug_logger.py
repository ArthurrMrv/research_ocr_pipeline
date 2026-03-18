from __future__ import annotations

import json

from rich.console import Console

_enabled: bool = False
_console = Console()


def enable() -> None:
    global _enabled
    _enabled = True


def is_enabled() -> bool:
    return _enabled


def print_step_start(step_name: str) -> None:
    if not _enabled:
        return
    _console.rule(f"[bold yellow]DEBUG · step: {step_name}")


def print_llm_response(provider_label: str, raw: str, parsed: dict | None = None) -> None:
    if not _enabled:
        return
    _console.print(f"[dim]── {provider_label} raw response ──[/dim]")
    _console.print(raw)
    if parsed is not None:
        _console.print("[dim]── parsed (validated) ──[/dim]")
        _console.print(json.dumps(parsed, indent=2, ensure_ascii=False))


def print_ocr_page(page_num: int, empty: bool) -> None:
    if not _enabled:
        return
    status = "[red]EMPTY[/red]" if empty else "[green]ok[/green]"
    _console.print(f"  page {page_num:>4}: {status}")
