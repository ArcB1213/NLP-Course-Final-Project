import re

from app.core.schemas import ChunkCreate, RawPage


SENTENCE_PATTERN = re.compile(r"([^。？！；\n]+[。？！；\n]?)")


class ChineseTextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, max(chunk_size - 1, 0))

    def split_pages(self, pages: list[RawPage], document_id: str, source_file: str) -> list[ChunkCreate]:
        chunks: list[ChunkCreate] = []
        for page in pages:
            pieces = self._split_text(page.text)
            heading = page.metadata.get("heading")
            for piece in pieces:
                chunks.append(
                    ChunkCreate(
                        document_id=document_id,
                        chunk_index=len(chunks),
                        text=piece,
                        source_file=source_file,
                        page=page.page,
                        heading=heading,
                    )
                )
        return chunks

    def _split_text(self, text: str) -> list[str]:
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            units = self._split_paragraph(paragraph)
            for unit in units:
                if not current:
                    current = unit
                elif len(current) + len(unit) <= self.chunk_size:
                    current = f"{current}\n{unit}"
                else:
                    chunks.append(current.strip())
                    current = self._with_overlap(current, unit)

        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _split_paragraph(self, paragraph: str) -> list[str]:
        if len(paragraph) <= self.chunk_size:
            return [paragraph]
        sentences = [match.group(1).strip() for match in SENTENCE_PATTERN.finditer(paragraph)]
        pieces: list[str] = []
        current = ""
        for sentence in sentences or [paragraph]:
            if not current:
                current = sentence
            elif len(current) + len(sentence) <= self.chunk_size:
                current += sentence
            else:
                pieces.append(current)
                current = self._with_overlap(current, sentence)
        if current:
            pieces.append(current)
        return pieces

    def _with_overlap(self, previous: str, next_text: str) -> str:
        if self.chunk_overlap <= 0:
            return next_text
        overlap = previous[-self.chunk_overlap :]
        return f"{overlap}{next_text}"

