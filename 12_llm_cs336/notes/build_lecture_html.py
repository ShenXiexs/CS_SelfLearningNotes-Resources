from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path
from typing import Any


LECTURES = ["lecture_01", "lecture_02", "lecture_06", "lecture_07", "lecture_10", "lecture_12"]


def inline_markdown(text: str) -> str:
    placeholders: list[str] = []

    def stash(value: str) -> str:
        placeholders.append(value)
        return f"\x00{len(placeholders) - 1}\x00"

    text = re.sub(r"`([^`]+)`", lambda m: stash(f"<code>{html.escape(m.group(1))}</code>"), text)
    text = re.sub(r"</?font[^>]*>", lambda m: stash(m.group(0)), text)
    text = html.escape(text, quote=False)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)

    for index, value in enumerate(placeholders):
        text = text.replace(f"\x00{index}\x00", value)
    return text


def markdown_block(markdown: str) -> str:
    lines = markdown.splitlines() or [markdown]
    if len(lines) > 1:
        return "".join(markdown_block(line) for line in lines)
    line = lines[0].strip()
    if not line:
        return ""
    if line.startswith("### "):
        return f"<h3>{inline_markdown(line[4:])}</h3>"
    if line.startswith("## "):
        return f"<h2>{inline_markdown(line[3:])}</h2>"
    if line.startswith("# "):
        return f"<h1>{inline_markdown(line[2:])}</h1>"
    if line.startswith("> "):
        return f"<blockquote>{inline_markdown(line[2:])}</blockquote>"
    match = re.match(r"^(\d+)\.\s+(.*)$", line)
    if match:
        return f"<ol start=\"{match.group(1)}\"><li>{inline_markdown(match.group(2))}</li></ol>"
    if line.startswith("- "):
        return f"<ul><li>{inline_markdown(line[2:])}</li></ul>"
    return f"<p>{inline_markdown(line)}</p>"


def style_attr(style: dict[str, Any]) -> str:
    parts = []
    for key, value in style.items():
        css_key = re.sub(r"([A-Z])", lambda m: "-" + m.group(1).lower(), key)
        css_value = f"{value}px" if isinstance(value, (int, float)) and key in {"width", "height"} else str(value)
        parts.append(f"{css_key}: {css_value}")
    return html.escape("; ".join(parts), quote=True)


def render_item(item: dict[str, Any]) -> str:
    item_type = item.get("type")
    style = style_attr(item.get("style") or {})
    style_fragment = f' style="{style}"' if style else ""

    if item_type == "markdown":
        return markdown_block(str(item.get("data", "")))
    if item_type == "image":
        src = html.escape(str(item.get("data", "")), quote=True)
        return f'<figure><img src="{src}"{style_fragment}></figure>'
    if item_type == "video":
        src = html.escape(str(item.get("data", "")), quote=True)
        return f'<video controls{style_fragment}><source src="{src}"></video>'
    if item_type == "link":
        external = item.get("external_link")
        internal = item.get("internal_link")
        label = html.escape(str(item.get("data") or "link"))
        if external:
            url = html.escape(str(external.get("url") or "#"), quote=True)
            title = html.escape(str(external.get("title") or item.get("data") or url))
            return f'<a class="ref" href="{url}" title="{title}">{label}</a>'
        if internal:
            target = html.escape(f"{internal.get('path')}:{internal.get('line_number')}", quote=True)
            return f'<a class="ref" href="#{target}">{label}</a>'
    return f"<span{style_fragment}>{html.escape(str(item.get('data', '')))}</span>"


def render_lecture(lecture: str) -> str:
    trace_path = Path("var/traces") / f"{lecture}.json"
    trace = json.loads(trace_path.read_text())
    body: list[str] = []
    for step in trace["steps"]:
        for item in step.get("renderings") or []:
            body.append(render_item(item))
    title = lecture.replace("_", " ").title()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f6f7f9; color: #18202a; }}
    main {{ max-width: 920px; margin: 0 auto; padding: 44px 28px 72px; background: white; min-height: 100vh; }}
    h1, h2, h3 {{ line-height: 1.18; margin: 1.45em 0 0.45em; }}
    h1 {{ font-size: 2.1rem; }}
    h2 {{ font-size: 1.55rem; border-top: 1px solid #e6e8ec; padding-top: 1.1em; }}
    h3 {{ font-size: 1.22rem; }}
    p, li, blockquote {{ font-size: 1.02rem; line-height: 1.58; }}
    p {{ margin: 0.55em 0; }}
    ul, ol {{ margin: 0.35em 0 0.35em 1.35em; padding: 0; }}
    blockquote {{ margin: 1em 0; padding: 0.7em 1em; border-left: 4px solid #c8d1dc; background: #f7f9fb; }}
    a {{ color: #2457d6; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .ref {{ margin-left: 0.2em; white-space: nowrap; }}
    figure {{ margin: 1.1em 0; text-align: center; }}
    img, video {{ max-width: 100%; height: auto; border: 0; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #eef1f5; padding: 0.08em 0.28em; border-radius: 4px; }}
    @media print {{
      body {{ background: white; }}
      main {{ padding: 0; max-width: none; }}
      a {{ color: #18202a; }}
      h2 {{ break-after: avoid; }}
      figure {{ break-inside: avoid; }}
    }}
  </style>
</head>
<body>
<main>
{chr(10).join(body)}
</main>
</body>
</html>
"""


def main() -> None:
    lectures = sys.argv[1:] or LECTURES
    for lecture in lectures:
        output_path = Path(f"{lecture}.html")
        output_path.write_text(render_lecture(lecture))
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
