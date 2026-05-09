"""Gera reports/relatorio_tecnico.pdf a partir de reports/relatorio_tecnico.md.

Usa mistune (markdown -> HTML) + Chrome headless (HTML -> PDF) com CSS @page
de margens estreitas (1.2 cm) para o conteudo ocupar toda a largura util da A4.

Substitui o pipeline anterior (extensao VSCode Markdown PDF -> wkhtmltopdf), que
aplicava o CSS do GitHub Markdown com max-width: 980px e centralizava o conteudo.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import mistune

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
MD_PATH = REPORTS_DIR / "relatorio_tecnico.md"
PDF_PATH = REPORTS_DIR / "relatorio_tecnico.pdf"

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]


def find_chrome() -> Path:
    for c in CHROME_CANDIDATES:
        if c.exists():
            return c
    raise SystemExit("Chrome/Edge nao encontrado em locais padrao. Instale ou ajuste CHROME_CANDIDATES.")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Extrai bloco YAML simples (sem aninhamento) entre --- no topo do arquivo."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block = text[3:end].strip()
    rest = text[end + 4 :].lstrip("\n")
    meta: dict[str, str] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, rest


def preprocess(md_text: str) -> str:
    """Transforma sintaxes pandoc-especificas em HTML/marcadores entendidos por mistune."""

    md_text = re.sub(r"^\\newpage\s*$", '<div class="page-break"></div>', md_text, flags=re.MULTILINE)

    def _img_with_attrs(match: re.Match[str]) -> str:
        alt = match.group(1)
        src = match.group(2)
        width_m = re.search(r"width=(\d+)%?", match.group(3) or "")
        width = f"{width_m.group(1)}%" if width_m else "100%"
        abs_src = (REPORTS_DIR / src).resolve().as_uri()
        return (
            f'<figure class="figure">'
            f'<img src="{abs_src}" style="width:{width};max-width:100%;" alt="{alt}">'
            f"<figcaption>{alt}</figcaption></figure>"
        )

    md_text = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)(\{[^}]*\})?",
        _img_with_attrs,
        md_text,
    )
    return md_text


def build_toc(md_text: str) -> str:
    """Sumario simples a partir dos headings de nivel 1 e 2."""
    entries: list[tuple[int, str, str]] = []
    for line in md_text.splitlines():
        m = re.match(r"^(#{1,2})\s+(.+?)\s*$", line)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2)
        slug = re.sub(r"[^\w\- ]", "", title).strip().lower().replace(" ", "-")
        entries.append((level, title, slug))

    items: list[str] = ['<nav class="toc"><h1 class="toc-title">Sumario</h1><ul>']
    last_level = 1
    for level, title, slug in entries:
        if level > last_level:
            items.append("<ul>")
        elif level < last_level:
            items.append("</ul>")
        items.append(f'<li><a href="#{slug}">{title}</a></li>')
        last_level = level
    while last_level > 1:
        items.append("</ul>")
        last_level -= 1
    items.append("</ul></nav>")
    return "\n".join(items)


def slugify_headings_in_html(html: str) -> str:
    """Adiciona id="slug" aos h1/h2 para casar com o sumario."""

    def _add_id(match: re.Match[str]) -> str:
        tag = match.group(1)
        inner = match.group(2)
        text = re.sub(r"<[^>]+>", "", inner)
        slug = re.sub(r"[^\w\- ]", "", text).strip().lower().replace(" ", "-")
        return f'<{tag} id="{slug}">{inner}</{tag}>'

    return re.sub(r"<(h[12])>(.*?)</\1>", _add_id, html, flags=re.DOTALL)


CSS = """
@page {
  size: A4;
  margin: 1.2cm 1.2cm 1.4cm 1.2cm;
}
html, body {
  margin: 0;
  padding: 0;
  width: 100%;
}
body {
  font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.45;
  color: #1f2328;
}
.cover {
  text-align: center;
  margin-top: 6cm;
}
.cover h1 {
  font-size: 26pt;
  margin-bottom: 0.4em;
  border-bottom: none;
  padding-bottom: 0;
}
.cover .subtitle { font-size: 14pt; color: #555; margin-bottom: 2em; }
.cover .author { font-size: 11pt; }
.cover .date { font-size: 11pt; color: #555; margin-top: 0.3em; }
h1 {
  font-size: 18pt;
  border-bottom: 2px solid #1f2328;
  padding-bottom: 0.2em;
  margin-top: 1em;
}
h2 {
  font-size: 13.5pt;
  margin-top: 1.2em;
  color: #0a3d62;
}
h3 { font-size: 11.5pt; margin-top: 1em; }
p { margin: 0.4em 0 0.6em 0; text-align: justify; }
ul, ol { margin: 0.3em 0 0.6em 1.6em; padding: 0; }
li { margin: 0.15em 0; }
strong { color: #111; }
code {
  background: #f3f4f6;
  border-radius: 3px;
  padding: 1px 4px;
  font-family: "Cascadia Mono", Consolas, "Courier New", monospace;
  font-size: 0.92em;
}
pre code {
  display: block;
  padding: 8px 12px;
  overflow-x: auto;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.6em 0;
  font-size: 9.5pt;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid #d0d7de;
  padding: 4px 8px;
  text-align: left;
  vertical-align: top;
}
th { background: #f3f4f6; }
tr:nth-child(even) td { background: #fafbfc; }
figure.figure {
  margin: 0.6em 0 0.8em 0;
  text-align: center;
  page-break-inside: avoid;
}
figure.figure img { display: inline-block; }
figcaption {
  font-size: 9pt;
  color: #57606a;
  margin-top: 0.3em;
  font-style: italic;
}
.page-break {
  break-before: page;
  page-break-before: always;
  height: 0;
}
hr {
  border: 0;
  border-top: 1px solid #d0d7de;
  margin: 1em 0;
}
a { color: #0969da; text-decoration: none; }
.toc { font-size: 10.5pt; }
.toc .toc-title {
  font-size: 18pt;
  border-bottom: 2px solid #1f2328;
  padding-bottom: 0.2em;
}
.toc ul {
  list-style: none;
  margin: 0.3em 0 0.3em 0.8em;
  padding: 0;
}
.toc > ul { margin-left: 0; }
.toc li { margin: 0.18em 0; }
.toc a { color: #1f2328; }
"""


def main() -> None:
    text = MD_PATH.read_text(encoding="utf-8")
    meta, body_md = parse_frontmatter(text)

    body_md = preprocess(body_md)
    toc_html = build_toc(body_md)

    md = mistune.create_markdown(
        plugins=["table", "strikethrough", "footnotes"],
        escape=False,
    )
    body_html = md(body_md)
    body_html = slugify_headings_in_html(body_html)

    title = meta.get("title", "Relatorio")
    subtitle = meta.get("subtitle", "")
    author = meta.get("author", "")
    date = meta.get("date", "")

    cover = f"""
    <section class="cover">
      <h1>{title}</h1>
      <div class="subtitle">{subtitle}</div>
      <div class="author">{author}</div>
      <div class="date">{date}</div>
    </section>
    <div class="page-break"></div>
    {toc_html}
    <div class="page-break"></div>
    """

    html_doc = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
{cover}
{body_html}
</body>
</html>
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8", dir=REPORTS_DIR
    ) as tmp:
        tmp.write(html_doc)
        html_path = Path(tmp.name)

    chrome = find_chrome()
    try:
        cmd = [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--no-margins",
            f"--print-to-pdf={PDF_PATH}",
            html_path.as_uri(),
        ]
        print("Rodando:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            print("STDOUT:", result.stdout, file=sys.stderr)
            print("STDERR:", result.stderr, file=sys.stderr)
            raise SystemExit(f"Chrome retornou codigo {result.returncode}")
    finally:
        html_path.unlink(missing_ok=True)

    size_kb = PDF_PATH.stat().st_size / 1024
    print(f"OK: {PDF_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
