import os
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Gradio usa la API interna — CLIP/FAISS se cargan una sola vez en api.py
os.environ.setdefault("RECOMMENDER_MODE",    "api")
os.environ.setdefault("RECOMMENDER_API_URL", "http://localhost:7860")

import importlib.util
from pathlib import Path

# ── Descarga índices desde HF Hub antes de importar api (que carga MovieRecommender) ──
_BUCKET = "buckets/carbonecar/recomendador-peliculas-index"
_REQUIRED = [
    ("data/processed/faiss.index",                         "faiss.index"),
    ("data/processed/index_metadata.csv",                  "index_metadata.csv"),
    ("data/processed/faiss_combined.index",                "faiss_combined.index"),
    ("data/processed/index_metadata_combined.csv",         "index_metadata_combined.csv"),
    ("data/processed/faiss_notextimg.index",               "faiss_notextimg.index"),
    ("data/processed/index_metadata_notextimg.csv",        "index_metadata_notextimg.csv"),
    ("data/processed/faiss_notextimg_combined.index",      "faiss_notextimg_combined.index"),
    ("data/processed/index_metadata_notextimg_combined.csv","index_metadata_notextimg_combined.csv"),
    ("data/processed/faiss_siglip.index",                  "faiss_siglip.index"),
    ("data/processed/index_metadata_siglip.csv",           "index_metadata_siglip.csv"),
    ("data/processed/faiss_dinov2.index",                  "faiss_dinov2.index"),
    ("data/processed/index_metadata_dinov2.csv",           "index_metadata_dinov2.csv"),
]
Path("data/processed").mkdir(parents=True, exist_ok=True)
if any(not Path(local).exists() for local, _ in _REQUIRED):
    from huggingface_hub import HfFileSystem
    _fs = HfFileSystem()
    for local, remote in _REQUIRED:
        if not Path(local).exists():
            _fs.get(f"{_BUCKET}/{remote}", local)

import gradio as gr
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# Carga el app FastAPI + MovieRecommender (CLIP + FAISS)
from api import app

# Carga el demo Gradio (modo api → no carga encoder propio)
_spec = importlib.util.spec_from_file_location(
    "_gradio_app", Path(__file__).parent / "app" / "app.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Monta Gradio en /gradio (retorna el app posiblemente wrapeado)
app = gr.mount_gradio_app(app, _mod.demo, path="/gradio")

# Redirect /gradio → /gradio/ (Gradio requiere barra final)
# Se agrega DESPUÉS del mount para que opere sobre el app correcto
app.add_api_route("/gradio", lambda: RedirectResponse(url="/gradio/"), methods=["GET", "HEAD"])

# Monta la webpage en / al final (captura todo lo que no matchea rutas de la API)
_webpage = Path(__file__).parent / "webpage"
if _webpage.exists():
    app.mount("/", StaticFiles(directory=str(_webpage), html=True), name="webpage")

if __name__ == "__main__":
    import torch
    import uvicorn
    torch.set_num_threads(1)
    uvicorn.run(app, host="0.0.0.0", port=7860)
