import asyncio
import json
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from app.config import get_settings
from app.evaluation.schemas import AppendRequest, DatasetRequest, Review
from app.evaluation.service import EvaluationService
from app.evaluation.store import BusyError


router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


@lru_cache
def get_evaluation_service() -> EvaluationService:
    return EvaluationService(get_settings())


def service_dependency():
    try:
        yield get_evaluation_service()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/defaults")
def defaults() -> dict:
    settings = get_settings()
    return {"generation_model": settings.eval_generation_model, "judge_model": settings.eval_judge_model}


@router.get("/datasets")
def list_datasets(service: EvaluationService = Depends(service_dependency)):
    return service.store.list_datasets()


@router.post("/datasets", status_code=201)
def create_dataset(request: DatasetRequest, service: EvaluationService = Depends(service_dependency)):
    return service.create(request)


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, service: EvaluationService = Depends(service_dependency)):
    return service.detail(dataset_id)


@router.post("/datasets/{dataset_id}/append")
def append_dataset(dataset_id: str, request: AppendRequest,
                   service: EvaluationService = Depends(service_dependency)):
    return service.append(dataset_id, request)


@router.post("/datasets/{dataset_id}/step")
def step(dataset_id: str, service: EvaluationService = Depends(service_dependency)):
    # A sync route runs in FastAPI's worker pool, keeping model loading off the UI event loop.
    return asyncio.run(service.step(dataset_id))


@router.post("/datasets/{dataset_id}/clone", status_code=201)
def clone(dataset_id: str, request: DatasetRequest, service: EvaluationService = Depends(service_dependency)):
    return service.create(request, source_id=dataset_id)


@router.post("/datasets/{dataset_id}/samples/{sample_id}/retry")
def retry(dataset_id: str, sample_id: str, service: EvaluationService = Depends(service_dependency)):
    service.retry(dataset_id, sample_id)
    return {"status": "ready"}


@router.put("/datasets/{dataset_id}/samples/{sample_id}/review")
def review(dataset_id: str, sample_id: str, request: Review,
           service: EvaluationService = Depends(service_dependency)):
    return service.review(dataset_id, sample_id, request)


@router.get("/datasets/{dataset_id}/compare/{baseline_id}")
def compare(dataset_id: str, baseline_id: str, service: EvaluationService = Depends(service_dependency)):
    return service.compare(dataset_id, baseline_id)


@router.get("/datasets/{dataset_id}/export")
def export(dataset_id: str, service: EvaluationService = Depends(service_dependency)):
    dataset = service.detail(dataset_id)
    samples = dataset.pop("samples")
    rows = [{"record_type": "metadata", **dataset}]
    rows.extend({"record_type": "sample", **s} for s in samples)
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    return Response(content, media_type="application/x-ndjson",
                    headers={"Content-Disposition": f'attachment; filename="evaluation-{dataset_id}.jsonl"'})
