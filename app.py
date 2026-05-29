import os
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Gradio usa la API interna — CLIP/FAISS se cargan una sola vez en api.py
os.environ.setdefault("RECOMMENDER_MODE",    "api")
os.environ.setdefault("RECOMMENDER_API_URL", "http://localhost:7860")

import importlib.util
from pathlib import Path

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
