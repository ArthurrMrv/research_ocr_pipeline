from glob import glob

import click
from dotenv import load_dotenv

from pipeline.formatting import run_formatting
from pipeline.ingest import ingest
from pipeline.ocr import run_ocr
from pipeline.scout import run_scout
from pipeline.tracker import get_all_doc_ids, get_supabase_client

ALL_STEPS = ("ingest", "ocr", "scout", "formatting")


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
    supa = get_supabase_client()

    steps_to_run = set(step) if step else set(ALL_STEPS)

    if "ingest" in steps_to_run:
        pdfs = glob(f"{pdf_dir}/*.pdf")
        new_ids = ingest(pdfs, supa)
        click.echo(f"Ingested {len(new_ids)} new documents.")

    all_ids = get_all_doc_ids(supa)
    click.echo(f"Processing {len(all_ids)} total documents.")

    if "ocr" in steps_to_run:
        for doc_id in all_ids:
            run_ocr(doc_id, supa, force=parse_all, since=parse_date)

    if "scout" in steps_to_run:
        for doc_id in all_ids:
            run_scout(doc_id, supa, force=parse_all)

    if "formatting" in steps_to_run:
        for doc_id in all_ids:
            run_formatting(doc_id, supa)

    click.echo(f"Done. {len(all_ids)} total documents processed.")


if __name__ == "__main__":
    run()
