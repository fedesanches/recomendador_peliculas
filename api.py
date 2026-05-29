import json
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.recommender import MovieRecommender

METRICS_PATH          = Path("data/metrics/metrics_clips.json")
METRICS_COMBINED_PATH = Path("data/metrics/metrics_clips_combined.json")

METRICS_SIGLIP_PATH   = Path("data/metrics/metrics_siglip.json")
METRICS_DINOV2_PATH   = Path("data/metrics/metrics_dinov2.json")

app = FastAPI(title="Recomendador de Películas", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = MovieRecommender()


class Movie(BaseModel):
    title:          str
    overview:       str | None
    genres:         str | None
    score:          float
    tmdb_poster_url: str | None


class RecommendResponse(BaseModel):
    results: List[Movie]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Métricas no disponibles. Ejecutar src/metrics/metric_service.py primero.")
    return json.loads(METRICS_PATH.read_text())


@app.get("/metrics/combined")
def get_metrics_combined():
    if not METRICS_COMBINED_PATH.exists():
        raise HTTPException(status_code=404, detail="Métricas combinadas no disponibles. Ejecutar src/metrics/metric_service.py --combined primero.")
    return json.loads(METRICS_COMBINED_PATH.read_text())


class UrlRequest(BaseModel):
    url: str


class BothResponse(BaseModel):
    image:    List[Movie]
    combined: List[Movie]


def _download_to_tmp(url: str) -> str:
    import requests as _req
    r = _req.get(url, timeout=15, stream=True)
    r.raise_for_status()
    suffix = Path(url.split("?")[0]).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        for chunk in r.iter_content(chunk_size=8192):
            tmp.write(chunk)
        return tmp.name


@app.post("/recommend/url/both", response_model=BothResponse)
def recommend_by_url_both(body: UrlRequest, top_k: int = 5):
    """Encode the image once, then search both indices — avoids double CLIP inference."""
    tmp_path = _download_to_tmp(body.url)
    try:
        vector      = recommender.encoder.encode_image(tmp_path)
        img_df      = recommender.index.search(vector, top_k)
        combined_df = recommender.index_combined.search(vector, top_k)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return BothResponse(
        image=    _to_movie_list(img_df),
        combined= _to_movie_list(combined_df),
    )
@app.get("/metrics/siglip")
def get_metrics_siglip():
    if not METRICS_SIGLIP_PATH.exists():
        raise HTTPException(status_code=404, detail="Métricas SigLIP no disponibles. Ejecutar src/metrics/metric_service.py --model siglip primero.")
    return json.loads(METRICS_SIGLIP_PATH.read_text())


@app.get("/metrics/dinov2")
def get_metrics_dinov2():
    if not METRICS_DINOV2_PATH.exists():
        raise HTTPException(status_code=404, detail="Métricas DINOv2 no disponibles. Ejecutar src/metrics/metric_service.py --model dinov2 primero.")
    return json.loads(METRICS_DINOV2_PATH.read_text())


@app.post("/recommend/image", response_model=RecommendResponse)
async def recommend_by_image(
    file: UploadFile = File(...),
    top_k: int = 5,
    combined: bool = False,
    model: str = "clip",
):
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
    if Path(file.filename or "").suffix.lower() not in valid_extensions:
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        results = recommender.recommend_from_image(tmp_path, top_k=top_k, combined=combined, model=model)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return _to_response(results)


@app.post("/recommend/text", response_model=RecommendResponse)
def recommend_by_text(query: str, top_k: int = 5, combined: bool = False, model: str = "clip"):
    if model == "dinov2":
        raise HTTPException(status_code=400, detail="DINOv2 no soporta búsqueda por texto.")
    try:
        results = recommender.recommend_from_text(query, top_k=top_k, combined=combined, model=model)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return _to_response(results)


def _to_movie_list(df) -> List[Movie]:
    return [
        Movie(
            title=           row.get("title", ""),
            overview=        row.get("overview"),
            genres=          row.get("genres"),
            score=           float(row["score"]),
            tmdb_poster_url= row.get("tmdb_poster_url"),
        )
        for _, row in df.iterrows()
    ]


def _to_response(results) -> RecommendResponse:
    return RecommendResponse(results=_to_movie_list(results))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
