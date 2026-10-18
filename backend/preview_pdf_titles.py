"""Preview text-only PDF title decisions without changing the document database."""
import argparse
import asyncio
import json
from pathlib import Path

from app.config import get_settings
from app.document.mineru_parser import load_mineru_pdf
from app.document.splitter import ChineseTextSplitter
from app.document.title_resolver import resolve_titles, title_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--pages", required=True, help="1-based pages, e.g. 43-45 or 25,27")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = set()
    for part in args.pages.split(","):
        bounds = [int(value) for value in part.strip().split("-")]
        if len(bounds) not in (1, 2) or min(bounds) < 1 or bounds[-1] < bounds[0]:
            parser.error("Invalid page range")
        selected.update(range(bounds[0], bounds[-1] + 1))
    settings = get_settings()
    raw = load_mineru_pdf(str(args.pdf), settings, resolve_headings=False)
    raw = [p for p in raw if p.page in selected]
    if {p.page for p in raw} != selected:
        parser.error("Page range exceeds the PDF")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "llm-input.json").write_text(json.dumps(title_inputs(raw), ensure_ascii=False, indent=2), encoding="utf-8")
    pages = asyncio.run(resolve_titles(raw, settings, cache_dir=args.output / "title-cache"))
    chunks = ChineseTextSplitter(settings.chunk_size, settings.chunk_overlap).split_pages(pages, "preview", args.pdf.name)
    result = {"model": settings.deepseek_model, "images_sent": False,
              "pages": [p.model_dump() for p in pages], "chunks": [c.model_dump() for c in chunks]}
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{"page": p.page, "heading": p.metadata.get("heading"),
                       "section": p.metadata.get("section_heading"),
                       "decision": p.metadata.get("title_resolution")} for p in pages], ensure_ascii=False))
    if any(p.metadata.get("title_resolution", {}).get("status") == "failed" for p in pages):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
