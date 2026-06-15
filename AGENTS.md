# Repository Guidelines

## Project Structure & Module Organization

This repository is a Course RAG Agent with a FastAPI backend and static frontend.

- `backend/app/api/` defines HTTP routes for health, documents, search, and chat.
- `backend/app/core/` contains agent orchestration, prompts, schemas, planning, and LLM integration.
- `backend/app/retrieval/`, `document/`, and `tools/` handle retrieval, ingestion, and QA/summary/quiz/grading tools.
- `backend/data/` stores local uploads, indexes, and SQLite data; commit only `.gitkeep` placeholders.
- `frontend/` contains the static HTML, CSS, and JavaScript served by FastAPI.
- `docs/` and root Markdown files hold design and API notes.

## Build, Test, and Development Commands

```powershell
conda create -n course-rag-agent python=3.10 -y
conda activate course-rag-agent
cd backend
pip install -r requirements.txt
copy .env.example .env
python run.py
```

- `python run.py` starts FastAPI with reload on `http://localhost:8000/`.
- `Invoke-RestMethod http://localhost:8000/api/health` checks the backend.
- `pytest` runs tests once tests are added.

## Coding Style & Naming Conventions

Use Python 3.10, 4-space indentation, type hints where they clarify interfaces, and Pydantic models for structured request/response data. Match the existing style: small service functions, explicit imports, and descriptive names such as `AgentService`, `HybridRetriever`, and `SummaryTool`. Use snake_case for modules, functions, variables, and environment keys; use PascalCase for classes.

Frontend code is plain JavaScript, HTML, and CSS. Keep it dependency-light and follow existing CSS variable naming.

## Testing Guidelines

`pytest` is listed as a backend dependency. Add tests under `backend/tests/` using names like `test_retriever.py` and `test_chat_api.py`. Prefer focused tests for retrieval behavior, document ingestion, API schemas, and agent fallback logic. For API tests, mock the LLM client layer.

## Commit & Pull Request Guidelines

Recent commits use Conventional Commit-style prefixes, for example `feat: improve summary coverage retrieval` and `feat(frontend): add markdown latex and agent trace`. Continue using `feat:`, `fix:`, `docs:`, `test:`, or scoped forms like `feat(prompts):`.

When the user asks for a commit, assess whether the staged or intended changes materially affect setup, public behavior, APIs, user-facing workflows, architecture, or project status. Update `README.md` before committing when that documentation would otherwise become stale. If the user gives an explicit instruction about README updates, follow that instruction.

Pull requests should include a concise summary, commands run, configuration changes, and screenshots or clips for frontend changes. Link related issues when available.

## Security & Configuration Tips

Do not commit `backend/.env`, uploaded files, indexes, or SQLite databases. Keep secrets in `.env`, starting from `backend/.env.example`. Call out changes to model names, upload limits, retrieval weights, or CORS origins.

## Agent-Specific Instructions

Make surgical changes only. State assumptions when behavior is ambiguous, prefer the simplest working implementation, and verify with a command or targeted test before handing off.
