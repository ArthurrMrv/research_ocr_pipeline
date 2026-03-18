import pytest

from pipeline.filename_parser import parse_filename


class TestParseFilename:
    def test_compact_date_format(self):
        institution, date = parse_filename("Amundi_20230727.pdf")
        assert institution == "amundi"
        assert date == "2023-07-27"

    def test_compact_date_another(self):
        institution, date = parse_filename("JPMorgan_20110930.pdf")
        assert institution == "jpmorgan"
        assert date == "2011-09-30"

    def test_compact_date_with_suffix(self):
        institution, date = parse_filename("Amundi_20220127_a.pdf")
        assert institution == "amundi"
        assert date == "2022-01-27"

    def test_dashed_date_format(self):
        institution, date = parse_filename("Apple_2024-01-15.pdf")
        assert institution == "apple"
        assert date == "2024-01-15"

    def test_dashed_date_no_extension(self):
        institution, date = parse_filename("Apple_2024-01-15")
        assert institution == "apple"
        assert date == "2024-01-15"

    def test_no_underscore_returns_none(self):
        assert parse_filename("report.pdf") == (None, None)

    def test_invalid_date_format_returns_none(self):
        assert parse_filename("Apple_Jan2024.pdf") == (None, None)

    def test_empty_company_returns_none(self):
        assert parse_filename("_2024-01-01.pdf") == (None, None)

    def test_just_pdf_extension(self):
        assert parse_filename(".pdf") == (None, None)

    def test_empty_string(self):
        assert parse_filename("") == (None, None)

    def test_institution_is_lowercased(self):
        institution, _ = parse_filename("Microsoft_2023-12-31.pdf")
        assert institution == "microsoft"

    def test_multiple_suffixes_after_date(self):
        institution, date = parse_filename("BlackRock_20200315_final_v2.pdf")
        assert institution == "blackrock"
        assert date == "2020-03-15"
