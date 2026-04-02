from dashboard.mermaid_utils import build_mermaid_export_html, sanitize_mermaid_diagram


class TestSanitizeMermaidDiagram:
    def test_normalizes_header_typo_and_inline_body(self):
        raw = "flowCchart TD A[Input] --> B[Output]"

        result = sanitize_mermaid_diagram(raw)

        assert result == "flowchart TD\nA[Input] --> B[Output]"

    def test_strips_mermaid_code_fences(self):
        raw = "```mermaid\nflowchart TD\nA --> B\n```"

        result = sanitize_mermaid_diagram(raw)

        assert result == "flowchart TD\nA --> B"

    def test_preserves_non_flowchart_text_when_unsalvageable(self):
        raw = "A --> B"

        result = sanitize_mermaid_diagram(raw)

        assert result == "A --> B"


class TestBuildMermaidExportHtml:
    def test_embeds_render_status_and_sanitized_diagram_source(self):
        reports = [
            {
                "doc_name": "Doc",
                "institution": "Firm",
                "model_name": "CAPM",
                "steps_summary": "Summary",
                "mermaid_diagram": "flowCchart TD A --> B",
            }
        ]

        html = build_mermaid_export_html(reports)

        assert "window.__mermaidStatus" in html
        assert '"total": 1' in html
        assert "Mermaid render failed for this diagram. Source included instead." in html
        assert "flowchart TD\nA --&gt; B" in html
