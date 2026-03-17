import pytest

from pipeline.filename_parser import parse_filename


class TestParseFilename:
    def test_valid_company_date(self):
        company, date = parse_filename("Apple_2024-01-15.pdf")
        assert company == "Apple"
        assert date == "2024-01-15"

    def test_valid_company_date_no_extension(self):
        company, date = parse_filename("Apple_2024-01-15")
        assert company == "Apple"
        assert date == "2024-01-15"

    def test_no_underscore_returns_none(self):
        assert parse_filename("report.pdf") == (None, None)

    def test_invalid_date_format_returns_none(self):
        assert parse_filename("Apple_Jan2024.pdf") == (None, None)

    def test_empty_company_returns_none(self):
        assert parse_filename("_2024-01-01.pdf") == (None, None)

    def test_multiple_underscores_splits_on_first(self):
        # company name is before the first underscore only
        assert parse_filename("Apple_2024_01_01.pdf") == (None, None)

    def test_just_pdf_extension(self):
        assert parse_filename(".pdf") == (None, None)

    def test_empty_string(self):
        assert parse_filename("") == (None, None)

    def test_valid_with_different_dates(self):
        company, date = parse_filename("Microsoft_2023-12-31.pdf")
        assert company == "Microsoft"
        assert date == "2023-12-31"
