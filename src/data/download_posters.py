import os
import threading
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"


def fetch_movie_details(tmdb_id: int) -> dict:
    url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return {}


def download_poster(poster_path: str, save_path: Path) -> bool:
    if not poster_path:
        return False
    url = f"{TMDB_IMAGE_BASE_URL}{poster_path}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            save_path.write_bytes(response.content)
            return True
    except requests.RequestException:
        pass
    return False


def build_movie_metadata(
    links_path: str,
    movies_path: str,
    posters_dir: str,
    output_path: str,
    max_workers: int = 20,
) -> pd.DataFrame:
    """Pipeline offline: descarga pósters y metadatos de TMDB en paralelo.

    Args:
        links_path: Ruta a links.csv de MovieLens (movieId → tmdbId).
        movies_path: Ruta a movies.csv de MovieLens.
        posters_dir: Directorio donde guardar los pósters descargados.
        output_path: Ruta del CSV de salida con metadatos procesados.
        max_workers: Hilos concurrentes (TMDB permite ~40 req/s; default 20).
    """
    posters_dir = Path(posters_dir)
    posters_dir.mkdir(parents=True, exist_ok=True)

    links = pd.read_csv(links_path).dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)
    movies = pd.read_csv(movies_path)
    df = movies.merge(links, on="movieId")

    # Semáforo para respetar el rate limit de TMDB (~40 req/s)
    # Con 20 workers y 2 requests por película, estamos en ~40 req/s
    rate_limiter = threading.Semaphore(max_workers)

    def process_movie(row) -> dict | None:
        tmdb_id = int(row["tmdbId"])
        with rate_limiter:
            details = fetch_movie_details(tmdb_id)
        if not details:
            return None

        poster_path = details.get("poster_path", "")
        poster_file = posters_dir / f"{tmdb_id}.jpg"

        if not poster_file.exists():
            with rate_limiter:
                download_poster(poster_path, poster_file)

        genres = ", ".join(g["name"] for g in details.get("genres", []))

        return {
            "movieId": row["movieId"],
            "tmdbId": tmdb_id,
            "title": details.get("title", row["title"]),
            "genres": genres,
            "overview": details.get("overview", ""),
            "poster_path": str(poster_file) if poster_file.exists() else "",
        }

    records = []
    rows = [row for _, row in df.iterrows()]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_movie, row): row for row in rows}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching TMDB data"):
            result = future.result()
            if result:
                records.append(result)

    metadata = pd.DataFrame(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_path, index=False)
    print(f"Guardados metadatos de {len(metadata)} películas en {output_path}")
    return metadata


if __name__ == "__main__":
    build_movie_metadata(
        links_path="data/raw/links.csv",
        movies_path="data/raw/movies.csv",
        posters_dir="data/posters",
        output_path="data/processed/metadata.csv",
    )
