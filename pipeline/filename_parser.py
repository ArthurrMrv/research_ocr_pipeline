import re


def parse_filename(doc_name: str) -> tuple[str | None, str | None]:
    """
    Parse filename into (institution, date_str).

    Supported formats:
      - 'Institution_YYYYMMDD.pdf'       -> ('institution', 'YYYY-MM-DD')
      - 'Institution_YYYYMMDD_a.pdf'     -> ('institution', 'YYYY-MM-DD')
      - 'Institution_YYYY-MM-DD.pdf'     -> ('institution', 'YYYY-MM-DD')

    Returns (None, None) if the filename does not match.
    """
    stem = doc_name.removesuffix(".pdf") if doc_name.endswith(".pdf") else doc_name
    parts = stem.split("_")
    if len(parts) < 2:
        return None, None

    institution = parts[0].strip()
    if not institution:
        return None, None

    # Search remaining parts for a date token
    for part in parts[1:]:
        # YYYYMMDD (compact)
        if re.fullmatch(r"\d{8}", part):
            date_str = f"{part[:4]}-{part[4:6]}-{part[6:8]}"
            return institution.lower(), date_str
        # YYYY-MM-DD (dashed)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", part):
            return institution.lower(), part

    return None, None
