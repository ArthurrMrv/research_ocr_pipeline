import re


def parse_filename(doc_name: str) -> tuple[str | None, str | None]:
    """
    Parse 'companyName_YYYY-MM-DD.pdf' -> (company_name, date_str).
    Returns (None, None) if the filename does not match the expected pattern.

    The company name is the portion before the first underscore.
    """
    stem = doc_name.removesuffix(".pdf") if doc_name.endswith(".pdf") else doc_name
    parts = stem.split("_", maxsplit=1)
    if len(parts) != 2:
        return None, None

    company_name = parts[0].strip()
    date_str = parts[1].strip()

    if not company_name:
        return None, None
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
        return None, None

    return company_name, date_str
