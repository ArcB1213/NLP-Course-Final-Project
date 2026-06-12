from enum import Enum


class TaskType(str, Enum):
    QA = "qa"
    SUMMARY = "summary"
    QUIZ = "quiz"
    GRADE = "grade"


class RouterAgent:
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
