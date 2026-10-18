"""Run MinerU in its own environment; keep the web backend dependency-light."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from threading import Lock

from app.config import Settings
from app.core.schemas import RawPage
from app.document.title_resolver import resolve_titles_sync


_parse_lock = Lock()


def load_mineru_pdf(file_path: str, settings: Settings, *, resolve_headings: bool = True) -> list[RawPage]:
    source = Path(file_path).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = settings.mineru_cache_dir.resolve() / f"{digest}-standard-v1"
    with _parse_lock:
        result = output / "pages-text-v3.json"
        if not result.exists():
            python = settings.mineru_python.resolve()
            if not python.is_file():
                raise ValueError("MinerU 环境不存在，请配置 MINERU_PYTHON 并安装标准版模型。")
            home = settings.mineru_home.resolve()
            env = os.environ.copy()
            env.update(MINERU_HOME=str(home / "state"), MINERU_CONFIG=str(home / "state/config.yaml"),
                       MINERU_MODEL_BASE_DIR=str(home / "models"), MINERU_MODEL_SOURCE="local",
                       MINERU_MODEL_SMALL_BACKEND="onnx", MINERU_MODEL_VLM_ENGINE="llama-cpp",
                       MINERU_MODEL_VLM_SERVER_URL="", PYTHONIOENCODING="utf-8")
            output.mkdir(parents=True, exist_ok=True)
            worker = Path(__file__).resolve().parents[2] / "mineru_worker.py"
            command = [str(python), str(worker), str(source), str(output)]
            if (output / "middle_json.json").exists():
                command.append("--render-only")
            try:
                with (output / "run.log").open("wb") as log:
                    run = subprocess.run(command,
                                         env=env, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=settings.mineru_timeout)
            except subprocess.TimeoutExpired as exc:
                result.unlink(missing_ok=True)
                raise ValueError(f"MinerU 解析超时，日志：{output / 'run.log'}") from exc
            if run.returncode or not result.exists():
                result.unlink(missing_ok=True)
                raise ValueError(f"MinerU 解析失败，日志：{output / 'run.log'}")
        pages = [RawPage.model_validate(page) for page in json.loads(result.read_text(encoding="utf-8"))]
        if resolve_headings:
            pages = resolve_titles_sync(pages, settings, output / "titles")
    return pages
