from __future__ import annotations

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
from pipeline.ingest import ingest
from pipeline.ocr import run_ocr
from pipeline.scout import run_scout
from pipeline.tracker import get_all_doc_ids, get_bronze_row, get_supabase_client

ALL_STEPS = ("ingest", "ocr", "scout", "formatting")

console = Console()


def _doc_label(client, doc_id: str) -> str:
    """Return a human-readable label for a doc_id (doc_name or truncated id)."""
    row = get_bronze_row(client, doc_id)
    if row and row.get("doc_name"):
        return row["doc_name"]
    return doc_id[:12]


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
            label = doc_labels[doc_id]
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
    parse_all: bool,
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
            label = doc_labels[doc_id]
            progress.update(main_task, description=f"[cyan]OCR[/cyan]  {label}")

            def _batch_cb(done: int, total: int, _bt=batch_task) -> None:
                progress.update(_bt, completed=done, total=total, visible=True,
                                description=f"[dim]batch {done}/{total}[/dim]")

            try:
                result = run_ocr(
                    doc_id, supa,
                    force=parse_all,
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
) -> None:
    """Formatting step with warnings for failed steps."""
    console.rule("[cyan bold]FORMATTING")

    errors: list[tuple[str, str]] = []
    warnings: list[tuple[str, list[dict]]] = []
    processed = skipped = 0
    step_start = time.monotonic()

    with _make_progress() as progress:
        task_id: TaskID = progress.add_task(
            "[cyan]formatting[/cyan]  ...",
            total=len(all_ids),
        )

        for doc_id in all_ids:
            label = doc_labels[doc_id]
            progress.update(task_id, description=f"[cyan]formatting[/cyan]  {label}")
            try:
                result = run_formatting(doc_id, supa)
                status = result["status"]
                failed_details = result.get("failed_details", [])
                if failed_details:
                    warnings.append((label, failed_details))
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
        for lbl, details in warnings:
            warn_parts.append(f"  [bold]{lbl}[/bold]:")
            for d in details:
                warn_parts.append(f"    {d['step']}: {d['reason']}")
        console.print(Panel("\n".join(warn_parts), title="[yellow]Failed formatting steps", border_style="yellow"))


@click.command()
@click.argument("pdf_dir", type=click.Path(exists=True))
@click.option("--parse-all", is_flag=True, help="Re-process all documents.")
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
def run(
    pdf_dir: str,
    parse_all: bool,
    parse_date: str | None,
    step: tuple[str, ...],
) -> None:
    """Run the ingestion pipeline on all PDFs in PDF_DIR."""
    load_dotenv()
    pipeline_start = time.monotonic()
    supa = get_supabase_client()

    steps_to_run = set(step) if step else set(ALL_STEPS)

    # Banner
    console.rule("[cyan bold]Ingestion Pipeline")
    console.print(f"  [dim]PDF directory[/dim] : {pdf_dir}")
    console.print(f"  [dim]Steps        [/dim] : {', '.join(s for s in ALL_STEPS if s in steps_to_run)}")
    if parse_all:
        console.print("  [dim]Mode         [/dim] : [red bold]force re-process ALL[/red bold]")
    elif parse_date:
        console.print(f"  [dim]Mode         [/dim] : re-process since {parse_date}")
    else:
        console.print("  [dim]Mode         [/dim] : incremental (skip already-done)")

    # --- INGEST ---
    if "ingest" in steps_to_run:
        console.rule("[cyan bold]INGEST")
        t0 = time.monotonic()
        pdfs = glob(f"{pdf_dir}/*.pdf")
        console.print(f"  Found [bold]{len(pdfs)}[/bold] PDF files in directory")
        new_ids = ingest(pdfs, supa)
        elapsed = time.monotonic() - t0
        console.print(
            f"  [green]✓[/green] [green]{len(new_ids)} new[/green], "
            f"{len(pdfs) - len(new_ids)} already registered "
            f"in {_format_elapsed(elapsed)}"
        )

    all_ids = get_all_doc_ids(supa)

    # Pre-fetch doc labels with spinner
    doc_labels: dict[str, str] = {}
    with console.status("[bold]Loading document index...[/bold]"):
        for did in all_ids:
            doc_labels[did] = _doc_label(supa, did)

    console.print(f"  Total documents: [bold]{len(all_ids)}[/bold]")

    # --- OCR ---
    if "ocr" in steps_to_run:
        _run_ocr_step(all_ids, doc_labels, supa, parse_all, parse_date)

    # --- SCOUT ---
    if "scout" in steps_to_run:
        _run_step(
            "scout",
            all_ids,
            doc_labels,
            lambda doc_id: run_scout(doc_id, supa, force=parse_all),
        )

    # --- FORMATTING ---
    if "formatting" in steps_to_run:
        _run_formatting_step(all_ids, doc_labels, supa)

    # Final summary
    total_elapsed = time.monotonic() - pipeline_start
    console.rule("[cyan bold]Pipeline complete")
    console.print(f"  [bold]{len(all_ids)}[/bold] docs · {_format_elapsed(total_elapsed)} total")
    console.print()


if __name__ == "__main__":
    run()
