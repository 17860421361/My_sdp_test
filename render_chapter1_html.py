from pathlib import Path
import base64
import html

from docx import Document


DOCX = Path(r"D:\组会\2026\8.15\Chapter_1_English_Polished.docx")
HTML = Path(r"E:\py_code\bupt_hjk\sdp_test\My_sdp_test\docx_qa_final\chapter1_qa.html")


def esc(value):
    return html.escape(value, quote=True)


def main():
    doc = Document(DOCX)
    blocks = []
    for paragraph in doc.paragraphs:
        text = esc(paragraph.text)
        style = paragraph.style.name
        if not text:
            continue
        if style == "Title":
            blocks.append(f'<h1 class="title">{text}</h1>')
        elif style == "Subtitle":
            blocks.append(f'<p class="subtitle">{text}</p>')
        elif style == "Heading 1":
            blocks.append(f'<h1 class="h1">{text}</h1>')
        elif style == "Heading 2":
            blocks.append(f'<h2>{text}</h2>')
        elif style == "Contribution":
            blocks.append(f'<p class="contribution">{text}</p>')
        elif style == "Caption":
            blocks.append(f'<p class="caption">{text}</p>')
        else:
            blocks.append(f'<p>{text}</p>')

    if doc.inline_shapes:
        rel = None
        for r in doc.part.rels.values():
            if "image" in r.reltype:
                rel = r
                break
        if rel is not None:
            encoded = base64.b64encode(rel.target_part.blob).decode("ascii")
            mime = rel.target_part.content_type
            cap_index = next(i for i, x in enumerate(blocks) if 'class="caption"' in x)
            blocks.insert(cap_index, f'<figure><img src="data:{mime};base64,{encoded}" /></figure>')

    css = """
    @page { size: A4; margin: 25.4mm 28mm; }
    * { box-sizing: border-box; }
    body { font-family: 'Times New Roman', serif; font-size: 11pt; line-height: 1.15; color: #000; margin: 0; }
    h1.title { font-size: 16pt; text-align: center; line-height: 1.05; margin: 0 0 12pt; }
    p.subtitle { font-size: 10.5pt; color: #505050; font-style: italic; text-align: center; margin: 0 0 16pt; }
    h1.h1 { font-size: 14pt; line-height: 1.05; margin: 6pt 0 10pt; break-after: avoid; }
    h2 { font-size: 12pt; line-height: 1.05; margin: 10pt 0 5pt; break-after: avoid; }
    p { text-align: justify; margin: 0 0 6pt; orphans: 2; widows: 2; }
    p.contribution { font-size: 10.5pt; line-height: 1.1; padding-left: 0.75cm; text-indent: -0.75cm; margin-bottom: 5pt; }
    figure { margin: 6pt 0 2pt; text-align: center; break-inside: avoid; }
    figure img { max-width: 100%; height: auto; }
    p.caption { font-size: 9pt; line-height: 1; margin: 0 0 8pt; break-before: avoid; }
    """
    content = "<!doctype html><html><head><meta charset='utf-8'><style>" + css + "</style></head><body>" + "\n".join(blocks) + "</body></html>"
    HTML.parent.mkdir(parents=True, exist_ok=True)
    HTML.write_text(content, encoding="utf-8")
    print(HTML)


if __name__ == "__main__":
    main()
