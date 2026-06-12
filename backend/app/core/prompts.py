from app.core.schemas import RetrievedChunk


QA_SYSTEM_PROMPT = """
你是一个严谨的中文课程学习助手。
你必须优先根据给定的课程资料回答问题。
如果资料中没有充分依据，请明确说明“课程资料中未找到充分依据”，不要编造。
回答要清晰、简洁，适合学生复习使用。
""".strip()


SUMMARY_SYSTEM_PROMPT = """
你是一个严谨的中文课程学习助手。
你必须根据给定课程资料做知识点总结。
如果资料覆盖不足，请在总结开头明确说明“课程资料依据不足，以下仅基于已检索到的片段整理”。
不要编造课程资料中没有的信息。
""".strip()


QUIZ_SYSTEM_PROMPT = """
你是一个严谨的中文课程助教。
你必须根据给定课程资料生成练习题。
题目、答案和考察点都必须有资料依据，不要编造资料中没有的信息。
""".strip()


GRADING_SYSTEM_PROMPT = """
你是一个严谨的中文课程助教。
你必须根据给定课程资料批改学生答案并给出学习反馈。
如果资料依据不足，请明确说明依据不足，不要编造参考答案。
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


def build_summary_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = format_chunks(chunks)
    return f"""
【课程资料片段】
{context}

【用户要求】
{query}

【总结要求】
1. 只基于课程资料片段总结。
2. 如果资料依据不足，请先说明依据不足。
3. 输出结构化提纲，适合学生复习。
4. 不要编造课程资料中没有的信息。

【输出格式】
一、核心概念
二、主要方法或知识点
三、方法之间的关系
四、易混淆点
五、复习建议
""".strip()


def build_quiz_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = format_chunks(chunks)
    return f"""
【课程资料片段】
{context}

【出题要求】
{query}

【要求】
1. 题目必须基于课程资料片段。
2. 每道题给出参考答案。
3. 每道题标注考察知识点。
4. 如果用户没有指定题型，默认生成判断题、选择题、简答题各一道。
5. 如果资料依据不足，请先说明依据不足。
""".strip()


def build_grading_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = format_chunks(chunks)
    return f"""
【课程资料片段】
{context}

【学生提交内容】
{query}

【批改要求】
1. 只依据课程资料片段评价答案。
2. 如果资料依据不足，请明确说明。
3. 不要编造课程资料中没有的信息。

【输出格式】
总体评价：基本正确 / 部分正确 / 存在明显问题
答对的点：
需要补充或修改的点：
建议修改后的参考答案：
相关资料依据：
""".strip()
