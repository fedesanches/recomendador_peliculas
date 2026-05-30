import json
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.recommender import MovieRecommender
from src.index.faiss_index import MovieIndex

METRICS_PATH          = Path("data/metrics/metrics_clips.json")
METRICS_COMBINED_PATH = Path("data/metrics/metrics_clips_combined.json")

METRICS_SIGLIP_PATH             = Path("data/metrics/metrics_siglip.json")
METRICS_SIGLIP_COMBINED_PATH    = Path("data/metrics/metrics_siglip_combined.json")
METRICS_DINOV2_PATH             = Path("data/metrics/metrics_dinov2.json")
METRICS_NOTEXTIMG_PATH          = Path("data/metrics/metrics_clip_notextimg.json")
METRICS_NOTEXTIMG_COMBINED_PATH = Path("data/metrics/metrics_clip_notextimg_combined.json")

app = FastAPI(title="Recomendador de Películas", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = MovieRecommender()

# Índices opcionales no-textimg (mismo encoder CLIP, índice distinto)
_NOTEXTIMG_INDEX_PATH    = Path("data/processed/faiss_notextimg.index")
_NOTEXTIMG_META_PATH     = Path("data/processed/index_metadata_notextimg.csv")
_NOTEXTIMG_COMB_PATH     = Path("data/processed/faiss_notextimg_combined.index")
_NOTEXTIMG_COMB_META     = Path("data/processed/index_metadata_notextimg_combined.csv")

_notextimg_index = None
if _NOTEXTIMG_INDEX_PATH.exists() and _NOTEXTIMG_META_PATH.exists():
    _notextimg_index = MovieIndex()
    _notextimg_index.load(str(_NOTEXTIMG_INDEX_PATH), str(_NOTEXTIMG_META_PATH))

_notextimg_combined_index = None
if _NOTEXTIMG_COMB_PATH.exists() and _NOTEXTIMG_COMB_META.exists():
    _notextimg_combined_index = MovieIndex()
    _notextimg_combined_index.load(str(_NOTEXTIMG_COMB_PATH), str(_NOTEXTIMG_COMB_META))



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

class ModelResult(BaseModel):
    key:     str
    label:   str
    results: List[Movie]

class AllResponse(BaseModel):
    models: List[ModelResult]


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
    tmp_path = _download_to_tmp(body.url)
    try:
        vector      = recommender.encoder.encode_image(tmp_path)
        img_df      = recommender.index.search(vector, top_k)
        combined_df = recommender.index_combined.search(vector, top_k)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return BothResponse(image=_to_movie_list(img_df), combined=_to_movie_list(combined_df))


@app.post("/recommend/url/all", response_model=AllResponse)
def recommend_by_url_all(body: UrlRequest, top_k: int = 5):
    """Encode once per encoder, search all available indices."""
    tmp_path = _download_to_tmp(body.url)
    models = []
    try:
        # CLIP — un solo encoding para todos los índices CLIP
        clip_vec = recommender.encoder.encode_image(tmp_path)
        models.append(ModelResult(key="clip-image",    label="Solo imagen (CLIP)",
                                  results=_to_movie_list(recommender.index.search(clip_vec, top_k))))
        models.append(ModelResult(key="clip-combined", label="Imagen + Texto (CLIP)",
                                  results=_to_movie_list(recommender.index_combined.search(clip_vec, top_k))))
        if _notextimg_index:
            models.append(ModelResult(key="notextimg", label="CLIP sin textimg",
                                      results=_to_movie_list(_notextimg_index.search(clip_vec, top_k))))
        if _notextimg_combined_index:
            models.append(ModelResult(key="notextimg-combined", label="CLIP sin textimg + Texto",
                                      results=_to_movie_list(_notextimg_combined_index.search(clip_vec, top_k))))
        # SigLIP
        if recommender.siglip_encoder and recommender.siglip_index:
            siglip_vec = recommender.siglip_encoder.encode_image(tmp_path)
            models.append(ModelResult(key="siglip", label="SigLIP",
                                      results=_to_movie_list(recommender.siglip_index.search(siglip_vec, top_k))))
        # DINOv2
        if recommender.dinov2_encoder and recommender.dinov2_index:
            dino_vec = recommender.dinov2_encoder.encode_image(tmp_path)
            models.append(ModelResult(key="dinov2", label="DINOv2",
                                      results=_to_movie_list(recommender.dinov2_index.search(dino_vec, top_k))))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return AllResponse(models=models)
@app.get("/metrics/siglip")
def get_metrics_siglip():
    if not METRICS_SIGLIP_PATH.exists():
        raise HTTPException(status_code=404, detail="Métricas SigLIP no disponibles. Ejecutar src/metrics/metric_service.py --model siglip primero.")
    return json.loads(METRICS_SIGLIP_PATH.read_text())


@app.get("/metrics/siglip/combined")
def get_metrics_siglip_combined():
    if not METRICS_SIGLIP_COMBINED_PATH.exists():
        raise HTTPException(status_code=404, detail="Métricas SigLIP combined no disponibles. Ejecutar src/metrics/metric_service.py --model siglip --combined primero.")
    return json.loads(METRICS_SIGLIP_COMBINED_PATH.read_text())


@app.get("/metrics/dinov2")
def get_metrics_dinov2():
    if not METRICS_DINOV2_PATH.exists():
        raise HTTPException(status_code=404, detail="Métricas DINOv2 no disponibles. Ejecutar src/metrics/metric_service.py --model dinov2 primero.")
    return json.loads(METRICS_DINOV2_PATH.read_text())


@app.get("/metrics/notextimg")
def get_metrics_notextimg():
    if not METRICS_NOTEXTIMG_PATH.exists():
        raise HTTPException(status_code=404, detail="Métricas CLIP no-textimg no disponibles.")
    return json.loads(METRICS_NOTEXTIMG_PATH.read_text())


@app.get("/metrics/notextimg/combined")
def get_metrics_notextimg_combined():
    if not METRICS_NOTEXTIMG_COMBINED_PATH.exists():
        raise HTTPException(status_code=404, detail="Métricas CLIP no-textimg combinado no disponibles.")
    return json.loads(METRICS_NOTEXTIMG_COMBINED_PATH.read_text())


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


def _nan_to_none(val):
    """Convierte NaN de pandas a None (Pydantic no acepta NaN como str|None)."""
    import math
    try:
        return None if math.isnan(float(val)) else val
    except (TypeError, ValueError):
        return val


def _to_movie_list(df) -> List[Movie]:
    return [
        Movie(
            title=           row.get("title", "") or "",
            overview=        _nan_to_none(row.get("overview")),
            genres=          _nan_to_none(row.get("genres")),
            score=           float(row["score"]),
            tmdb_poster_url= _nan_to_none(row.get("tmdb_poster_url")),
        )
        for _, row in df.iterrows()
    ]


def _to_response(results) -> RecommendResponse:
    return RecommendResponse(results=_to_movie_list(results))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
