from glob import glob

import click
from dotenv import load_dotenv

from pipeline.formatting import get_kimi_client, run_formatting
from pipeline.ingest import ingest
from pipeline.ocr import run_ocr
from pipeline.tracker import get_all_doc_ids, get_supabase_client


@click.command()
@click.argument("pdf_dir", type=click.Path(exists=True))
@click.option("--parse-all", is_flag=True, help="Re-process all documents.")
@click.option(
    "--parse-date",
    type=str,
    default=None,
    help="Re-process documents added on or after this ISO date (e.g. 2024-01-01).",
)
def run(pdf_dir: str, parse_all: bool, parse_date: str | None) -> None:
    """Run the ingestion pipeline on all PDFs in PDF_DIR."""
    load_dotenv()
    supa = get_supabase_client()
    kimi = get_kimi_client()

    pdfs = glob(f"{pdf_dir}/*.pdf")
    new_ids = ingest(pdfs, supa)
    click.echo(f"Ingested {len(new_ids)} new documents.")

    all_ids = get_all_doc_ids(supa)
    click.echo(f"Processing {len(all_ids)} total documents.")

    for doc_id in all_ids:
        run_ocr(doc_id, supa, force=parse_all, since=parse_date)

    for doc_id in all_ids:
        run_formatting(doc_id, supa, kimi)

    click.echo(f"Done. {len(new_ids)} new, {len(all_ids)} total.")


if __name__ == "__main__":
    run()
