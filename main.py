from __future__ import annotations

import os
import time
from glob import glob

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from pipeline.formatting import run_formatting
from pipeline.ingest import ingest, make_doc_id
from pipeline.ocr import run_ocr
from pipeline.scout import run_scout
from pipeline.tracker import get_all_bronze_rows, get_all_doc_ids, get_supabase_client

ALL_STEPS = ("ingest", "ocr", "scout", "formatting")

console = Console()


def _build_doc_labels(client) -> dict[str, str]:
    """Build a {doc_id: label} map from all bronze rows in a single query."""
    rows = get_all_bronze_rows(client)
    labels: dict[str, str] = {}
    for row in rows:
        doc_id = row["doc_id"]
        labels[doc_id] = row.get("doc_name") or doc_id[:12]
    return labels


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def _run_step(
    step_name: str,
    all_ids: list[str],
    doc_labels: dict[str, str],
    run_fn,
) -> None:
    """Run a generic pipeline step with rich progress display."""
    console.rule(f"[cyan bold]{step_name.upper()}")

    errors: list[tuple[str, str]] = []
    processed = skipped = 0
    step_start = time.monotonic()

    with _make_progress() as progress:
        task_id: TaskID = progress.add_task(
            f"[cyan]{step_name}[/cyan]  ...",
            total=len(all_ids),
        )

        for doc_id in all_ids:
            label = doc_labels.get(doc_id, doc_id[:12])
            progress.update(task_id, description=f"[cyan]{step_name}[/cyan]  {label}")
            try:
                status = run_fn(doc_id)
            except Exception as exc:
                status = "error"
                errors.append((label, str(exc)))
            progress.advance(task_id)
            if status == "done":
                processed += 1
            elif status == "skipped":
                skipped += 1

    elapsed = time.monotonic() - step_start
    error_count = len(errors)

    if error_count:
        console.print(
            f"[red]✗[/red] {step_name.upper()} — "
            f"[green]{processed} processed[/green], "
            f"[yellow]{skipped} skipped[/yellow], "
            f"[red]{error_count} errors[/red] "
            f"in {_format_elapsed(elapsed)}"
        )
        error_lines = "\n".join(f"  [bold]{lbl}[/bold]: {msg}" for lbl, msg in errors)
        console.print(Panel(error_lines, title="[red]Errors", border_style="red"))
    else:
        console.print(
            f"[green]✓[/green] {step_name.upper()} — "
            f"[green]{processed} processed[/green], "
            f"[yellow]{skipped} skipped[/yellow] "
            f"in {_format_elapsed(elapsed)}"
        )


def _run_ocr_step(
    all_ids: list[str],
    doc_labels: dict[str, str],
    supa,
    force: bool,
    parse_date: str | None,
) -> None:
    """OCR step with batch sub-progress."""
    console.rule("[cyan bold]OCR")

    errors: list[tuple[str, str]] = []
    warnings: list[tuple[str, int]] = []
    processed = skipped = 0
    step_start = time.monotonic()

    with _make_progress() as progress:
        main_task: TaskID = progress.add_task(
            "[cyan]OCR[/cyan]  ...",
            total=len(all_ids),
        )
        batch_task: TaskID = progress.add_task(
            "[dim]batch ...[/dim]",
            total=1,
            visible=False,
        )

        for doc_id in all_ids:
            label = doc_labels.get(doc_id, doc_id[:12])
            progress.update(main_task, description=f"[cyan]OCR[/cyan]  {label}")

            def _batch_cb(done: int, total: int, _bt=batch_task) -> None:
                progress.update(_bt, completed=done, total=total, visible=True,
                                description=f"[dim]batch {done}/{total}[/dim]")

            try:
                result = run_ocr(
                    doc_id, supa,
                    force=force,
                    since=parse_date,
                    progress_callback=_batch_cb,
                )
                status = result["status"]
                empty_count = result["empty_pages"]
                if empty_count > 0:
                    warnings.append((label, empty_count))
            except Exception as exc:
                status = "error"
                errors.append((label, str(exc)))

            progress.update(batch_task, visible=False)
            progress.advance(main_task)

            if status == "done":
                processed += 1
            elif status == "skipped":
                skipped += 1

    elapsed = time.monotonic() - step_start
    error_count = len(errors)

    if error_count:
        console.print(
            f"[red]✗[/red] OCR — "
            f"[green]{processed} processed[/green], "
            f"[yellow]{skipped} skipped[/yellow], "
            f"[red]{error_count} errors[/red] "
            f"in {_format_elapsed(elapsed)}"
        )
        error_lines = "\n".join(f"  [bold]{lbl}[/bold]: {msg}" for lbl, msg in errors)
        console.print(Panel(error_lines, title="[red]Errors", border_style="red"))
    else:
        console.print(
            f"[green]✓[/green] OCR — "
            f"[green]{processed} processed[/green], "
            f"[yellow]{skipped} skipped[/yellow] "
            f"in {_format_elapsed(elapsed)}"
        )

    if warnings:
        warn_lines = "\n".join(
            f"  [bold]{lbl}[/bold]: {cnt} empty page(s)" for lbl, cnt in warnings
        )
        console.print(Panel(warn_lines, title="[yellow]Empty pages", border_style="yellow"))


def _run_formatting_step(
    all_ids: list[str],
    doc_labels: dict[str, str],
    supa,
    *,
    force: bool = False,
) -> None:
    """Formatting step with warnings for failed steps."""
    console.rule("[cyan bold]FORMATTING")
    from pipeline import debug_logger

    errors: list[tuple[str, str]] = []
    warnings: list[tuple[str, list[dict], list[dict]]] = []
    processed = skipped = 0
    step_start = time.monotonic()

    with _make_progress() as progress:
        task_id: TaskID = progress.add_task(
            "[cyan]formatting[/cyan]  ...",
            total=len(all_ids),
        )

        for doc_id in all_ids:
            label = doc_labels.get(doc_id, doc_id[:12])
            progress.update(task_id, description=f"[cyan]formatting[/cyan]  {label}")
            try:
                result = run_formatting(doc_id, supa, force=force)
                status = result["status"]
                failed_details = result.get("failed_details", [])
                attempt_history = result.get("attempt_history", [])
                if failed_details or attempt_history:
                    warnings.append((label, failed_details, attempt_history))
            except Exception as exc:
                status = "error"
                errors.append((label, str(exc)))

            progress.advance(task_id)
            if status == "done":
                processed += 1
            elif status == "skipped":
                skipped += 1

    elapsed = time.monotonic() - step_start
    error_count = len(errors)

    if error_count:
        console.print(
            f"[red]✗[/red] FORMATTING — "
            f"[green]{processed} processed[/green], "
            f"[yellow]{skipped} skipped[/yellow], "
            f"[red]{error_count} errors[/red] "
            f"in {_format_elapsed(elapsed)}"
        )
        error_lines = "\n".join(f"  [bold]{lbl}[/bold]: {msg}" for lbl, msg in errors)
        console.print(Panel(error_lines, title="[red]Errors", border_style="red"))
    else:
        console.print(
            f"[green]✓[/green] FORMATTING — "
            f"[green]{processed} processed[/green], "
            f"[yellow]{skipped} skipped[/yellow] "
            f"in {_format_elapsed(elapsed)}"
        )

    if warnings:
        warn_parts: list[str] = []
        for lbl, details, attempt_history in warnings:
            warn_parts.append(f"  [bold]{lbl}[/bold]:")
            for d in details:
                _append_formatting_failure_lines(warn_parts, d, indent="    ", debug=debug_logger.is_enabled())
            if debug_logger.is_enabled() and attempt_history:
                for attempt in attempt_history:
                    warn_parts.append(f"    attempt {attempt['attempt']}:")
                    for failure in attempt["failures"]:
                        _append_formatting_failure_lines(warn_parts, failure, indent="      ", debug=True)
        console.print(Panel("\n".join(warn_parts), title="[yellow]Failed formatting steps", border_style="yellow"))


def _append_formatting_failure_lines(
    parts: list[str],
    failure: dict,
    *,
    indent: str,
    debug: bool,
) -> None:
    """Render one formatting failure entry, including full raw output in debug mode."""
    parts.append(f"{indent}{failure['step']}: {failure['reason']}")
    raw_output = failure.get("raw_output")
    if debug and isinstance(raw_output, str):
        parts.append(f"{indent}raw output:")
        for line in raw_output.splitlines() or [""]:
            parts.append(f"{indent}  {line}")


@click.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--parse-all", is_flag=True, help="Force re-process all selected pipeline steps.")
@click.option(
    "--parse-date",
    type=str,
    default=None,
    help="Re-process documents added on or after this ISO date (e.g. 2024-01-01).",
)
@click.option(
    "--step",
    type=click.Choice(["ingest", "ocr", "scout", "formatting"]),
    multiple=True,
    help="Run only specific pipeline step(s). Omit to run all.",
)
@click.option(
    "--force",
    "force_steps",
    type=click.Choice(["ingest", "ocr", "scout", "formatting"]),
    multiple=True,
    help="Force re-run specific pipeline step(s). Repeat as needed.",
)
@click.option("--debug", is_flag=True, help="Print full LLM responses and per-page OCR status.")
def run(
    pdf_path: str,
    parse_all: bool,
    parse_date: str | None,
    step: tuple[str, ...],
    force_steps: tuple[str, ...],
    debug: bool,
) -> None:
    """Run the ingestion pipeline on PDF_PATH (a single PDF file or a directory of PDFs)."""
    load_dotenv()
    pipeline_start = time.monotonic()
    supa = get_supabase_client()

    if debug:
        from pipeline import debug_logger
        debug_logger.enable()

    steps_to_run = set(step) if step else set(ALL_STEPS)
    forced_steps = set(force_steps)
    if parse_all:
        forced_steps = set(ALL_STEPS)

    # Resolve single file vs directory
    if os.path.isfile(pdf_path):
        pdfs = [pdf_path]
        single_file_basename: str | None = os.path.basename(pdf_path)
        single_file_doc_id: str | None = make_doc_id(pdf_path)
    else:
        pdfs = glob(f"{pdf_path}/*.pdf")
        single_file_basename = None
        single_file_doc_id = None

    # Banner
    console.rule("[cyan bold]Ingestion Pipeline")
    console.print(f"  [dim]PDF path     [/dim] : {pdf_path}")
    console.print(f"  [dim]Steps        [/dim] : {', '.join(s for s in ALL_STEPS if s in steps_to_run)}")
    if debug:
        console.print("  [dim]Debug mode  [/dim] : [yellow bold]ON[/yellow bold]")
    if parse_all:
        console.print("  [dim]Mode         [/dim] : [red bold]force re-process ALL[/red bold]")
    elif parse_date:
        console.print(f"  [dim]Mode         [/dim] : re-process since {parse_date}")
    else:
        console.print("  [dim]Mode         [/dim] : incremental (skip already-done)")
    if forced_steps and not parse_all:
        console.print(f"  [dim]Force steps  [/dim] : {', '.join(s for s in ALL_STEPS if s in forced_steps)}")

    # --- INGEST ---
    if "ingest" in steps_to_run:
        console.rule("[cyan bold]INGEST")
        t0 = time.monotonic()
        console.print(f"  Found [bold]{len(pdfs)}[/bold] PDF file(s)")
        new_ids = ingest(pdfs, supa)
        elapsed = time.monotonic() - t0
        console.print(
            f"  [green]✓[/green] [green]{len(new_ids)} new[/green], "
            f"{len(pdfs) - len(new_ids)} already registered "
            f"in {_format_elapsed(elapsed)}"
        )

    all_ids = get_all_doc_ids(supa)

    # Pre-fetch doc labels in a single query
    with console.status("[bold]Loading document index...[/bold]"):
        doc_labels = _build_doc_labels(supa)

    if single_file_doc_id:
        all_ids = [did for did in all_ids if did == single_file_doc_id]
        if not all_ids:
            console.print(f"[red]No document registered for '{single_file_basename}' at '{pdf_path}'[/red]")
            return

    console.print(f"  Total documents: [bold]{len(all_ids)}[/bold]")

    # --- OCR ---
    if "ocr" in steps_to_run:
        _run_ocr_step(all_ids, doc_labels, supa, "ocr" in forced_steps, parse_date)

    # --- SCOUT ---
    if "scout" in steps_to_run:
        _run_step(
            "scout",
            all_ids,
            doc_labels,
            lambda doc_id: run_scout(doc_id, supa, force="scout" in forced_steps),
        )

    # --- FORMATTING ---
    if "formatting" in steps_to_run:
        _run_formatting_step(all_ids, doc_labels, supa, force="formatting" in forced_steps)

    # Final summary
    total_elapsed = time.monotonic() - pipeline_start
    console.rule("[cyan bold]Pipeline complete")
    console.print(f"  [bold]{len(all_ids)}[/bold] docs · {_format_elapsed(total_elapsed)} total")
    console.print()


if __name__ == "__main__":
    run()
