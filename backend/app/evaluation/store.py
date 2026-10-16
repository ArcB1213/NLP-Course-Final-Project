import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


class BusyError(ValueError):
    pass


class EvaluationStore:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "evaluations.sqlite"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                    lease TEXT, busy_until REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS samples (
                    id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL, payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS samples_dataset ON samples(dataset_id, ordinal);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, dataset: dict, samples: list[dict]) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO datasets(id, payload) VALUES (?, ?)",
                       (dataset["id"], json.dumps(dataset, ensure_ascii=False)))
            db.executemany("INSERT INTO samples VALUES (?, ?, ?, ?)", [
                (s["id"], dataset["id"], i, json.dumps(s, ensure_ascii=False))
                for i, s in enumerate(samples)
            ])

    def append(self, dataset: dict, samples: list[dict]) -> None:
        # The caller holds the dataset lease; metadata and rows commit together.
        with self.connect() as db:
            db.execute("UPDATE datasets SET payload=? WHERE id=?",
                       (json.dumps(dataset, ensure_ascii=False), dataset["id"]))
            offset = db.execute("SELECT COALESCE(MAX(ordinal), -1) + 1 FROM samples WHERE dataset_id=?",
                                (dataset["id"],)).fetchone()[0]
            db.executemany("INSERT INTO samples VALUES (?, ?, ?, ?)", [
                (s["id"], dataset["id"], offset + i, json.dumps(s, ensure_ascii=False))
                for i, s in enumerate(samples)
            ])

    def get(self, dataset_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT payload, busy_until FROM datasets WHERE id=?", (dataset_id,)).fetchone()
            if row is None:
                raise KeyError("评测集不存在")
            data = json.loads(row[0])
            data["busy"] = row[1] > time.time()
            return data

    def list_datasets(self) -> list[dict]:
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT payload FROM datasets ORDER BY rowid DESC")]

    def samples(self, dataset_id: str) -> list[dict]:
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute(
                "SELECT payload FROM samples WHERE dataset_id=? ORDER BY ordinal", (dataset_id,))]

    def save_sample(self, sample: dict) -> None:
        with self.connect() as db:
            db.execute("UPDATE samples SET payload=? WHERE id=? AND dataset_id=?",
                       (json.dumps(sample, ensure_ascii=False), sample["id"], sample["dataset_id"]))

    @contextmanager
    def lease(self, dataset_id: str):
        self.get(dataset_id)
        token = uuid4().hex
        with self.connect() as db:
            changed = db.execute(
                "UPDATE datasets SET lease=?, busy_until=? WHERE id=? AND busy_until<?",
                (token, time.time() + 1800, dataset_id, time.time()),
            ).rowcount
        if not changed:
            raise BusyError("该评测集正在处理，请等待当前步骤完成")
        try:
            yield
        finally:
            with self.connect() as db:
                db.execute("UPDATE datasets SET lease=NULL, busy_until=0 WHERE id=? AND lease=?",
                           (dataset_id, token))
