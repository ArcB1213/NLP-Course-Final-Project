from app.core.schemas import RetrievedChunk


QA_SYSTEM_PROMPT = """
你是一个严谨的中文课程学习助手。
你必须优先根据给定的课程资料回答问题。
如果资料中没有充分依据，请明确说明“课程资料中未找到充分依据”，不要编造。
回答要清晰、简洁，适合学生复习使用。
""".strip()


def format_chunks(chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        location = chunk.source_file
        if chunk.page is not None:
            location += f"，第 {chunk.page} 页"
        if chunk.heading:
            location += f"，标题：{chunk.heading}"
        lines.append(f"[{index}] 来源：{location}\n{chunk.text}")
    return "\n\n".join(lines)


def build_qa_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = format_chunks(chunks)
    return f"""
【课程资料片段】
{context}

【用户问题】
{query}

【回答要求】
1. 先直接回答问题。
2. 再给出必要解释。
3. 如果资料依据不足，请说明依据不足。
4. 不要编造课程资料中没有的信息。
""".strip()

