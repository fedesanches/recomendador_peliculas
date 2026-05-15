import os
os.environ["KMP_DUPLICATE_LIB_OK"]    = "TRUE"
os.environ["OMP_NUM_THREADS"]          = "1"
os.environ["TOKENIZERS_PARALLELISM"]   = "false"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from logging import getLogger
load_dotenv()

BUCKET     = "buckets/carbonecar/recomendador-peliculas-index"
INDEX_PATH = "data/processed/faiss.index"
META_PATH  = "data/processed/index_metadata.csv"
COMBINED_INDEX_PATH = "data/processed/faiss_combined.index"
COMBINED_META_PATH  = "data/processed/index_metadata_combined.csv"

logger = getLogger(__name__)

def _download_index():
    Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    if not Path(INDEX_PATH).exists() or not Path(META_PATH).exists() \
        or not Path(COMBINED_INDEX_PATH).exists() or not Path(COMBINED_META_PATH).exists():
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        if not Path(INDEX_PATH).exists():
            fs.get(f"{BUCKET}/faiss.index", INDEX_PATH)
        if not Path(META_PATH).exists():
            fs.get(f"{BUCKET}/index_metadata.csv", META_PATH)

        if not Path(COMBINED_INDEX_PATH).exists():
            fs.get(f"{BUCKET}/faiss_combined.index", COMBINED_INDEX_PATH)
        if not Path(COMBINED_META_PATH).exists():
            fs.get(f"{BUCKET}/index_metadata_combined.csv", COMBINED_META_PATH)


_download_index()

import os
import base64
import requests as http_requests
import pandas as pd
import gradio as gr
from src.metrics.metric_service import MetricResult


RECOMMENDER_MODE    = os.getenv("RECOMMENDER_MODE", "embedded")
RECOMMENDER_API_URL = os.getenv("RECOMMENDER_API_URL", "http://localhost:8000")

if RECOMMENDER_MODE == "embedded":
    from src.recommender import MovieRecommender
    _recommender = MovieRecommender()

def _recommend_embedded(image_path: str, top_k: int, combined: bool) -> pd.DataFrame:
    return _recommender.recommend_from_image(image_path, top_k=top_k, combined=combined)

def _recommend_api(image_path: str, top_k: int, combined: bool) -> pd.DataFrame:
    with open(image_path, "rb") as f:
        response = http_requests.post(
            f"{RECOMMENDER_API_URL}/recommend/image",
            files={"file": f},
            params={"top_k": top_k, "combined": combined},
            timeout=30,
        )
    response.raise_for_status()
    return pd.DataFrame(response.json()["results"])

def _recommend(image_path: str, top_k: int, combined: bool) -> pd.DataFrame:
    if RECOMMENDER_MODE == "api":
        return _recommend_api(image_path, top_k, combined)
    return _recommend_embedded(image_path, top_k, combined)


def _img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

_DCG_IMG = f"data:image/png;base64,{_img_to_base64('DCG.png')}"

import json

_METRICS_PATH          = Path("data/processed/metrics.json")
_METRICS_COMBINED_PATH = Path("data/processed/metrics_combined.json")

def _load_metrics(combined: bool = False) -> MetricResult | None:
    if RECOMMENDER_MODE == "api":
        endpoint = f"{RECOMMENDER_API_URL}/metrics{'/combined' if combined else ''}"
        try:
            response = http_requests.get(endpoint, timeout=10)
            response.raise_for_status()
            return MetricResult(**response.json())
        except Exception as e:
            logger.error(f"Error al obtener métricas desde API: {e}")
            return None
    path = _METRICS_COMBINED_PATH if combined else _METRICS_PATH
    if not path.exists():
        return None
    return MetricResult(**json.loads(path.read_text()))

metrics          = _load_metrics(combined=False)
metrics_combined = _load_metrics(combined=True)


METRIC_DEFINITIONS = {
    "Precision@K":         "De las K películas recomendadas, fracción que supera el umbral de similitud de género (Jaccard).",
    "Recall@K":            "De todas las películas del dataset con similitud de género >= umbral, fracción que aparece en las K recomendaciones.",
    "NDCG@K":              f"Normalized Discounted Cumulative Gain: mide la calidad del ranking. Una recomendación relevante en posición 1 vale más que en posición K.<br><img src='{_DCG_IMG}' style='width:240px;margin-top:8px'>",
    "Coherencia de género": "Jaccard promedio entre los géneros de la película consultada y los de cada recomendación.",
    "Aciertos":            "Total de recomendaciones individuales que superan el umbral Jaccard, sobre el total de recomendaciones (muestras × K).",
}


def format_metrics(m: MetricResult | None) -> str:
    if m is None:
        rows_data = [(k, "—") for k in METRIC_DEFINITIONS]
    else:
        rows_data = [
            ("Precision@K",          f"{m.precision:.4f}"),
            ("Recall@K",             f"{m.recall:.4f}"),
            ("NDCG@K",               f"{m.ndcg:.4f}"),
            ("Coherencia de género", f"{m.gender_coherence:.4f}"),
            ("Aciertos",             f"{m.aciertos} ({m.aciertos_pct:.2%})"),
        ]
    rows_html = ""
    for name, value in rows_data:
        definition = METRIC_DEFINITIONS[name]
        rows_html += f"""
        <tr>
            <td style="padding:6px 12px;display:flex;justify-content:space-between;align-items:center">
                <span>{name}</span>
                <span style="position:relative;display:inline-block;cursor:pointer;margin-left:8px">
                    <span style="color:#888;font-size:13px">ℹ</span>
                    <span style="visibility:hidden;width:280px;background:#333;color:#fff;font-size:12px;
                                 border-radius:6px;padding:6px 10px;position:absolute;z-index:10;
                                 bottom:125%;right:0;
                                 opacity:0;transition:opacity 0.2s"
                          class="tip">{definition}</span>
                </span>
            </td>
            <td style="padding:6px 12px;font-weight:bold">{value}</td>
        </tr>"""

    return f"""
    <style>
        span:hover > .tip {{ visibility:visible !important; opacity:1 !important; }}
    </style>
    <table style="border-collapse:collapse;width:100%">
        <thead>
            <tr style="border-bottom:1px solid #ddd">
                <th style="padding:6px 12px;text-align:left">Métrica</th>
                <th style="padding:6px 12px;text-align:left">Valor</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>"""


def recommend(image_path: str, top_k: int, index_mode: str):
    if image_path is None:
        return [], "Por favor, sube una imagen de portada."

    combined = index_mode == "Imagen + Texto"
    results = _recommend(image_path, top_k=int(top_k), combined=combined)

    images = []
    text = ""
    for _, row in results.iterrows():
        url = row.get("tmdb_poster_url", "")
        if url:
            images.append((url, row["title"]))

        text += f"### {row['title']}  ·  score: {row['score']:.3f}\n"
        overview = row.get("overview", "")
        if overview:
            text += f"{overview}\n"
        text += "\n---\n"

    return images, text


with gr.Blocks(title="Recomendador de Películas por Portada") as demo:
    gr.Markdown("# Recomendador de Películas por Portada")
    gr.Markdown(
        "Sube la portada de una película y encontraremos las más similares usando CLIP."
    )

    with gr.Row():
        image_input = gr.Image(type="filepath", label="Portada de consulta")
        with gr.Column():
            top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Recomendaciones")
            index_radio = gr.Radio(
                choices=["Solo imagen", "Imagen + Texto"],
                value="Solo imagen",
                label="Índice de búsqueda",
            )
            with gr.Accordion("Métricas del modelo", open=False):
                gr.Markdown("Evaluación sobre 500 muestras aleatorias con umbral Jaccard = 0.5")
                with gr.Tabs():
                    with gr.Tab("Solo imagen (CLIP)"):
                        gr.HTML(format_metrics(metrics))
                    with gr.Tab("Imagen + Texto (CLIP combinado)"):
                        gr.HTML(format_metrics(metrics_combined))

    recommend_btn = gr.Button("Buscar similares", variant="primary")
    gallery = gr.Gallery(label="Películas recomendadas", columns=5, height="auto")
    output = gr.Markdown()

    recommend_btn.click(
        fn=recommend,
        inputs=[image_input, top_k_slider, index_radio],
        outputs=[gallery, output],
    )

if __name__ == "__main__":
    import torch
    torch.set_num_threads(1)
    demo.launch(max_threads=1)
