import json
from enum import Enum

import httpx

from app.core.llm_client import DeepSeekClient
from app.core.schemas import AgentDecision


class TaskType(str, Enum):
    QA = "qa"
    SUMMARY = "summary"
    QUIZ = "quiz"
    GRADE = "grade"


class RouterAgent:
    def __init__(self, llm: DeepSeekClient | None = None):
        self.llm = llm

    async def decide(self, query: str, user_task_type: str | None = None) -> AgentDecision:
        if user_task_type and user_task_type != "auto":
            task_type = self.route(query, user_task_type)
            return AgentDecision(
                task_type=task_type.value,
                retrieval_profile=_default_profile(task_type),
                rewritten_query=query,
                needs_pro_model=task_type == TaskType.GRADE,
                confidence="high",
                reason="user_explicit_task_type",
            )

        if self.llm is not None:
            try:
                return await self._decide_with_llm(query)
            except (RuntimeError, httpx.HTTPError, ValueError, json.JSONDecodeError):
                pass

        task_type = self.route(query, user_task_type)
        return AgentDecision(
            task_type=task_type.value,
            retrieval_profile=_default_profile(task_type),
            rewritten_query=query,
            needs_pro_model=task_type == TaskType.GRADE,
            confidence="medium",
            reason="rule_fallback",
        )

    def route(self, query: str, user_task_type: str | None = None) -> TaskType:
        if user_task_type and user_task_type != "auto":
            return TaskType(user_task_type)

        text = query.strip()
        if any(word in text for word in ("总结", "梳理", "归纳", "提纲", "复习")):
            return TaskType.SUMMARY
        if any(word in text for word in ("出题", "出三道题", "道题", "题目", "练习题", "选择题", "判断题", "简答题", "考考我")):
            return TaskType.QUIZ
        if any(word in text for word in ("我的答案", "批改", "对不对", "帮我看看", "评分", "评价一下")):
            return TaskType.GRADE
        return TaskType.QA

    async def _decide_with_llm(self, query: str) -> AgentDecision:
        assert self.llm is not None
        content = await self.llm.chat(
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        payload = _parse_json_object(content)
        decision = AgentDecision.model_validate(payload)
        if not decision.rewritten_query:
            decision.rewritten_query = query
        return decision


ROUTER_SYSTEM_PROMPT = """
你是中文课程学习助手的任务路由 Agent。
你只负责判断任务类型和检索策略，不回答用户问题。
必须只输出一个 JSON 对象，不要输出 Markdown，不要解释。

可选 task_type：
- qa：回答具体问题
- summary：总结、梳理、归纳知识点
- quiz：生成练习题、选择题、判断题、简答题
- grade：批改或评价学生答案

可选 retrieval_profile：
- qa：精确问答检索
- summary：广覆盖总结检索

输出 JSON 字段：
{
  "task_type": "qa|summary|quiz|grade",
  "retrieval_profile": "qa|summary",
  "rewritten_query": "用于检索的改写查询，保留核心术语",
  "needs_pro_model": false,
  "confidence": "high|medium|low",
  "reason": "简短原因"
}

规则：
- 总结、梳理、归纳、复习类任务用 summary。
- 出题、练习题、考考我类任务用 quiz。
- 批改、评分、我的答案、对不对类任务用 grade，并通常 needs_pro_model=true。
- 具体概念解释和事实问答用 qa。
- quiz 和 grade 通常使用 summary retrieval_profile，以覆盖更多资料。
""".strip()


def _parse_json_object(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Router output does not contain a JSON object")
    return json.loads(text[start : end + 1])


def _default_profile(task_type: TaskType) -> str:
    if task_type == TaskType.QA:
        return "qa"
    return "summary"
