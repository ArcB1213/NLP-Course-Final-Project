"""Evaluate the actual single-turn Agent pipeline on one or more annotated datasets."""
import argparse
import asyncio
import json
import os
from pathlib import Path


def main() -> int:
    caller_directory = Path.cwd()
    os.chdir(Path(__file__).resolve().parent)
    from app.config import get_settings
    from app.evaluation.store import EvaluationStore
    from app.evaluation.system import SystemEvaluator, SystemRunConfig, read_export

    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List existing annotation datasets")
    create = commands.add_parser("create", help="Select datasets and create an isolated system run")
    create.add_argument("--datasets", nargs="+", default=[])
    create.add_argument("--files", nargs="+", default=[])
    create.add_argument("--name", default="系统端到端评测")
    create.add_argument("--task-type", choices=["auto", "qa"], default="auto")
    create.add_argument("--top-k", type=int)
    create.add_argument("--use-pro-model", action="store_true")
    create.add_argument("--judge-model", default=settings.eval_judge_model)
    create.add_argument("--reviewed-only", action="store_true")
    create.add_argument("--split", choices=["all", "calibration", "test"], default="all")
    create.add_argument("--limit", type=int)
    run = commands.add_parser("run", help="Resume answers and judging; continue past failed cases")
    run.add_argument("run_id")
    run.add_argument("--retry-errors", action="store_true")
    run.add_argument("--max-steps", type=int, default=10000)
    report = commands.add_parser("report")
    report.add_argument("run_id")
    export = commands.add_parser("export")
    export.add_argument("run_id")
    export.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "list":
        source = EvaluationStore(settings.eval_dir)
        for d in source.list_datasets():
            rows = source.samples(d["id"])
            print(json.dumps({"id": d["id"], "name": d["config"]["name"], "total": len(rows),
                              "generated": sum(bool(s.get("generated")) for s in rows)},
                             ensure_ascii=False))
        return 0

    evaluator = SystemEvaluator(settings)
    if args.command == "create":
        if not args.datasets and not args.files:
            parser.error("provide --datasets and/or --files")
        source = EvaluationStore(settings.eval_dir)
        bundles = [{"metadata": source.get(key), "samples": source.samples(key)} for key in args.datasets]
        bundles.extend(read_export((caller_directory / name).resolve()) for name in args.files)
        config = SystemRunConfig(
            name=args.name, task_type=args.task_type, top_k=args.top_k,
            use_pro_model=args.use_pro_model, judge_model=args.judge_model,
            reviewed_only=args.reviewed_only, split=args.split, limit=args.limit)
        metadata = evaluator.create(bundles, config)
        print(metadata["id"])
    elif args.command == "run":
        if args.max_steps < 1:
            parser.error("--max-steps must be positive")
        if args.retry_errors:
            evaluator.retry_errors(args.run_id)
        async def process():
            for _ in range(args.max_steps):
                result = await evaluator.step(args.run_id)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                if result["status"] == "idle":
                    break
            report = evaluator.report(args.run_id)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            evaluator.export(args.run_id, settings.eval_dir / "system-runs" / "exports")
            return int(bool(report["overall_unique_cases"]["errors"]))
        return asyncio.run(process())
    elif args.command == "report":
        print(json.dumps(evaluator.report(args.run_id), ensure_ascii=False, indent=2))
    else:
        evaluator.export(args.run_id, (caller_directory / args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
