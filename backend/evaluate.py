"""Offline CLI for the same workflow used by /evaluation.html."""
import argparse
import asyncio
import json
import os
from pathlib import Path


def main() -> int:
    os.chdir(Path(__file__).resolve().parent)
    from app.config import get_settings
    from app.evaluation.schemas import AppendRequest, DatasetRequest
    from app.evaluation.service import EvaluationService

    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Create a dataset, or clone fixed questions")
    create.add_argument("--name", default="课程检索评测")
    create.add_argument("--count", type=int, default=20)
    create.add_argument("--top-k", type=int, default=5)
    create.add_argument("--profile", choices=["qa", "summary", "default"], default="qa")
    create.add_argument("--seed", type=int, default=42)
    create.add_argument("--judge-passes", type=int, choices=[1, 2], default=2)
    create.add_argument("--generation-model", default=settings.eval_generation_model)
    create.add_argument("--judge-model", default=settings.eval_judge_model)
    create.add_argument("--documents", nargs="*", default=[])
    create.add_argument("--source", help="Existing dataset id; reuse its questions")
    append = commands.add_parser("append", help="Append unique-question plans to an existing dataset")
    append.add_argument("dataset_id")
    append.add_argument("--count", type=int, required=True)
    run = commands.add_parser("run", help="Resume pending steps; skip failed samples")
    run.add_argument("dataset_id")
    run.add_argument("--max-steps", type=int, default=1000)
    retry = commands.add_parser("retry", help="Clear failed-step markers, keeping saved results")
    retry.add_argument("dataset_id")
    for name in ("status", "export"):
        sub = commands.add_parser(name)
        sub.add_argument("dataset_id")
    args = parser.parse_args()
    service = EvaluationService(settings)
    if args.command == "create":
        request = DatasetRequest(
            name=args.name, count=args.count, top_k=args.top_k, profile=args.profile,
            seed=args.seed, judge_passes=args.judge_passes, generation_model=args.generation_model,
            judge_model=args.judge_model, document_ids=args.documents)
        result = service.create(request, source_id=args.source)
        print(result["id"])
    elif args.command == "append":
        result = service.append(args.dataset_id, AppendRequest(count=args.count))
        print(json.dumps(result["progress"], ensure_ascii=False))
    elif args.command == "run":
        async def run_steps():
            for _ in range(args.max_steps):
                result = await service.step(args.dataset_id)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                if result["status"] == "idle":
                    return int(any(s["error"] for s in service.store.samples(args.dataset_id)))
            return int(any(s["error"] for s in service.store.samples(args.dataset_id)))
        return asyncio.run(run_steps())
    elif args.command == "retry":
        service.store.get(args.dataset_id)
        for sample in service.store.samples(args.dataset_id):
            if sample["error"]:
                service.retry(args.dataset_id, sample["id"])
        print("已清除失败标记；使用 run 继续")
    else:
        result = service.detail(args.dataset_id)
        samples = result.pop("samples")
        if args.command == "status":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(json.dumps({"record_type": "metadata", **result}, ensure_ascii=False))
            for sample in samples:
                print(json.dumps({"record_type": "sample", **sample}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
