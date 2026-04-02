"""Utilities for normalizing Mermaid diagrams and building export HTML."""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone

_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:mermaid)?\s*(.*?)\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)
_FLOW_HEADER_RE = re.compile(
    r"^\s*(?:flow\s*c?chart|graph)\s*([A-Za-z]{2})\b",
    re.IGNORECASE,
)


def _slugify(text: str) -> str:
    """Create an anchor-friendly slug from text."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"[\s_]+", "-", slug)


def _dedupe_anchors(report_list: list[dict]) -> list[tuple[str, str]]:
    """Return (title, unique_anchor) pairs, appending a suffix on duplicates."""
    seen: dict[str, int] = {}
    results: list[tuple[str, str]] = []
    for report in report_list:
        title = f"{report['doc_name']} — {report['institution']}"
        base = _slugify(title)
        count = seen.get(base, 0)
        seen[base] = count + 1
        anchor = base if count == 0 else f"{base}-{count}"
        results.append((title, anchor))
    return results


def sanitize_mermaid_diagram(diagram: object) -> str:
    """Normalize common Mermaid formatting issues without changing semantics."""
    if not isinstance(diagram, str):
        return ""

    text = diagram.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    fence_match = _CODE_FENCE_RE.match(text)
    if fence_match:
        text = fence_match.group(1).strip()

    text = text.lstrip("\ufeff").strip()
    header_match = _FLOW_HEADER_RE.match(text)
    if not header_match:
        return text

    direction = header_match.group(1).upper()
    if direction not in {"TD", "TB", "BT", "RL", "LR"}:
        direction = "TD"

    rest = text[header_match.end():].lstrip(" \t;")
    rest = rest.lstrip("\n").strip()
    if not rest:
        return f"flowchart {direction}"
    return f"flowchart {direction}\n{rest}"


def build_mermaid_export_html(report_list: list[dict]) -> str:
    """Build a self-contained HTML file that renders Mermaid safely."""
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title_anchors = _dedupe_anchors(report_list)

    toc_items = []
    sections = []

    for i, (report, (title, anchor)) in enumerate(zip(report_list, title_anchors), 1):
        escaped_title = html.escape(title)
        toc_items.append(
            f'      <li><a href="#{html.escape(anchor)}">{escaped_title}</a></li>'
        )

        meta_lines = []
        if report["model_name"]:
            meta_lines.append(
                f"<p><strong>Model:</strong> {html.escape(report['model_name'])}</p>"
            )
        if report["steps_summary"]:
            meta_lines.append(
                f"<p><strong>Summary:</strong> {html.escape(report['steps_summary'])}</p>"
            )
        meta_html = "\n        ".join(meta_lines)

        diagram = sanitize_mermaid_diagram(report["mermaid_diagram"])
        diagram_html = html.escape(diagram)

        sections.append(f"""    <section id="{html.escape(anchor)}">
      <h2>{i}. {escaped_title}</h2>
      {meta_html}
      <div class="mermaid-shell">
        <pre class="mermaid-source">{diagram_html}</pre>
      </div>
    </section>""")

    toc_html = "\n".join(toc_items)
    sections_html = "\n\n".join(sections)
    total_diagrams = len(report_list)
    render_state = json.dumps(
        {
            "total": total_diagrams,
            "rendered": 0,
            "failed": 0,
            "done": False,
            "errors": [],
        }
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Mermaid Charts Export</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      max-width: 900px;
      margin: 0 auto;
      padding: 2rem;
      color: #1a1a1a;
      line-height: 1.6;
    }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5rem; }}
    h2 {{ color: #2c3e50; margin-top: 2rem; }}
    .meta {{ color: #555; font-size: 0.9rem; }}
    nav ol {{ columns: 2; column-gap: 2rem; }}
    nav a {{ text-decoration: none; color: #2563eb; }}
    nav a:hover {{ text-decoration: underline; }}
    section {{ margin-bottom: 2rem; }}
    .mermaid-shell {{
      background: #f8f9fa;
      padding: 1rem;
      border-radius: 6px;
      overflow: hidden;
    }}
    .mermaid-shell svg {{
      max-width: 100%;
      height: auto;
    }}
    .mermaid-source,
    .mermaid-fallback {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.9rem;
    }}
    .mermaid-error {{
      margin-top: 0.75rem;
      color: #b45309;
      font-size: 0.9rem;
    }}
    hr {{ border: none; border-top: 1px solid #ddd; margin: 2rem 0; }}

    .print-btn {{
      position: fixed;
      top: 1rem;
      right: 1rem;
      padding: 0.5rem 1.2rem;
      background: #2563eb;
      color: white;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.9rem;
      z-index: 1000;
    }}
    .print-btn:hover {{ background: #1d4ed8; }}

    @media print {{
      body {{ padding: 0; max-width: 100%; }}
      nav {{ page-break-after: always; }}
      section {{ page-break-before: always; }}
      a {{ color: inherit; text-decoration: none; }}
      .print-btn {{ display: none; }}
    }}
  </style>
</head>
<body>
  <button class="print-btn" onclick="window.print()">Print / Save as PDF</button>
  <h1>Mermaid Charts Export</h1>
  <p class="meta">Generated: {now}</p>

  <nav>
    <h2>Table of Contents</h2>
    <ol>
{toc_html}
    </ol>
  </nav>

  <hr>

{sections_html}

  <script>
    window.__mermaidStatus = {render_state};

    function markAllAsFailed(message) {{
      const status = window.__mermaidStatus;
      const nodes = Array.from(document.querySelectorAll(".mermaid-source"));
      for (const node of nodes) {{
        node.className = "mermaid-fallback";
        const msg = document.createElement("p");
        msg.className = "mermaid-error";
        msg.textContent = message;
        node.parentElement.appendChild(msg);
      }}
      status.failed = nodes.length;
      status.errors.push(message);
      status.done = true;
    }}

    async function renderAllMermaid() {{
      const status = window.__mermaidStatus;
      const nodes = Array.from(document.querySelectorAll(".mermaid-source"));
      if (!nodes.length) {{
        status.done = true;
        return;
      }}
      if (!window.mermaid) {{
        markAllAsFailed("Mermaid library failed to load. Diagram source included instead.");
        return;
      }}

      mermaid.initialize({{ startOnLoad: false, theme: "default", securityLevel: "loose" }});

      for (const node of nodes) {{
        const source = node.textContent.trim();
        const wrapper = node.parentElement;
        const container = document.createElement("div");
        container.className = "mermaid";
        container.textContent = source;
        wrapper.replaceChild(container, node);
        try {{
          await mermaid.run({{ nodes: [container] }});
          status.rendered += 1;
        }} catch (err) {{
          const fallback = document.createElement("pre");
          fallback.className = "mermaid-fallback";
          fallback.textContent = source;
          wrapper.replaceChild(fallback, container);
          const msg = document.createElement("p");
          msg.className = "mermaid-error";
          msg.textContent = "Mermaid render failed for this diagram. Source included instead.";
          wrapper.appendChild(msg);
          status.failed += 1;
          status.errors.push(String(err));
        }}
      }}

      status.done = true;
    }}

    window.addEventListener("load", function () {{
      renderAllMermaid().catch(function (err) {{
        markAllAsFailed("Unexpected Mermaid rendering error. Diagram source included instead.");
        window.__mermaidStatus.errors.push(String(err));
      }});
    }});
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
</body>
</html>"""
