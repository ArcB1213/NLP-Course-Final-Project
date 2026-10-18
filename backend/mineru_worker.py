"""Called only by the backend through the isolated MinerU Python interpreter."""
import json
from pathlib import Path
import sys

from mineru import MinerUParser, ParseResult
from mineru.parser.writer import FileBasedDataWriter
from mineru.render.markdown import render_markdown
from mineru.render.contracts import RenderMode


def main():
    source, output = map(Path, sys.argv[1:3])
    if "--render-only" not in sys.argv[3:]:
        with MinerUParser(tier="standard", parse_mode="auto", image_analysis=True) as parser:
            result = parser.parse(source)
            result.save(FileBasedDataWriter(str(output)))
    # save() externalizes assets on a copy. Render that persisted representation,
    # and exclude image payloads/URLs from text consumed by retrieval and LLMs.
    result = ParseResult.from_json((output / "middle_json.json").read_text(encoding="utf-8"))
    pages = []
    for page in result.pages:
        candidates = []
        for block in page.blocks:
            if block.type not in {"header", "paragraph_title", "text"}:
                continue
            if block.type == "text" and (not block.bbox or block.bbox[1] > 0.35):
                continue
            candidate_page = page.model_copy(update={"blocks": [block]})
            candidate_doc = result.middle_json.model_copy(update={"pages": [candidate_page], "is_full_document": False})
            candidate_text = render_markdown(candidate_doc, mode=RenderMode.FULL, image_renderer=lambda b: "").strip().lstrip("# ")
            if candidate_text and len(candidate_text) <= 600:
                candidates.append({"block_id": block.index, "type": str(block.type), "bbox": block.bbox, "text": candidate_text})
        blocks = [block for block in page.blocks if block.type not in {"header", "footer", "page_number"}]
        body = page.model_copy(update={"blocks": blocks})
        one = result.middle_json.model_copy(update={"pages": [body], "is_full_document": False})
        text = render_markdown(one, image_renderer=lambda block: "").strip()
        kinds = {block.type for block in blocks}
        warnings = []
        if "table" in kinds or "equation" in kinds:
            warnings.append("recognized_math_table_requires_review")
        if "image" in kinds:
            warnings.append("image_semantics_not_extracted")
        if not text:
            warnings.append("no_native_text")
        title = next((block for block in blocks if block.type == "paragraph_title"), None)
        heading = None
        if title is not None:
            title_page = page.model_copy(update={"blocks": [title]})
            heading = render_markdown(one.model_copy(update={"pages": [title_page]}), image_renderer=lambda block: "").strip().lstrip("# ")
        pages.append({"text": text, "page": page.page_idx + 1, "metadata": {
            "heading": heading, "extraction_method": "mineru-standard-4.0.12-llama-cpp",
            "warnings": warnings, "artifact_dir": str(output.resolve()),
            "title_candidates": candidates,
            "blocks": [{"type": block.type, "bbox": block.bbox} for block in blocks],
        }})
    target = output / "pages-text-v3.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


if __name__ == "__main__":
    main()
