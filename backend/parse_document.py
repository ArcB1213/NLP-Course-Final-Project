"""Preview PDF extraction without changing the document database or indexes."""

import argparse
import json
from pathlib import Path

from app.document.loaders import PDFLoader
from app.document.processing_report import build_processing_report
from app.document.splitter import ChineseTextSplitter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pages = PDFLoader().load(str(args.pdf))
    report = build_processing_report(pages)
    chunks = ChineseTextSplitter().split_pages(pages, "preview", args.pdf.name)
    report["chunk_count"] = len(chunks)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    text = "\n\n".join(f"## 第 {page.page} 页\n\n{page.text}" for page in pages)
    (args.output / "document.md").write_text(text, encoding="utf-8")
    print(json.dumps({"pages": len(pages), "chunks": len(chunks), "warnings": len(report["warnings"]),
                      "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
