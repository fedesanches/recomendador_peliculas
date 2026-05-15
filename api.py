import io
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.recommender import MovieRecommender

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


@app.post("/recommend/image", response_model=RecommendResponse)
async def recommend_by_image(file: UploadFile = File(...), top_k: int = 5):
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
    if Path(file.filename or "").suffix.lower() not in valid_extensions:
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    results = recommender.recommend_from_image(tmp_path, top_k=top_k)
    Path(tmp_path).unlink(missing_ok=True)
    return _to_response(results)


@app.post("/recommend/text", response_model=RecommendResponse)
def recommend_by_text(query: str, top_k: int = 5):
    results = recommender.recommend_from_text(query, top_k=top_k)
    return _to_response(results)


def _to_response(results) -> RecommendResponse:
    movies = [
        Movie(
            title=          row.get("title", ""),
            overview=       row.get("overview"),
            genres=         row.get("genres"),
            score=          float(row["score"]),
            tmdb_poster_url= row.get("tmdb_poster_url"),
        )
        for _, row in results.iterrows()
    ]
    return RecommendResponse(results=movies)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
