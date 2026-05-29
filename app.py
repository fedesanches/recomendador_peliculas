import os
os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Gradio usará la API interna; CLIP/FAISS se cargan solo una vez en el API thread
os.environ.setdefault("RECOMMENDER_MODE",    "api")
os.environ.setdefault("RECOMMENDER_API_URL", "http://localhost:8000")

import threading
import time
import importlib.util
from pathlib import Path

import uvicorn


def _start_api():
    config = uvicorn.Config(
        "api:app",
        host="0.0.0.0",
        port=8000,
        log_level="warning",
        reload=False,
    )
    server = uvicorn.Server(config)
    server.run()


api_thread = threading.Thread(target=_start_api, daemon=True)
api_thread.start()

# Esperar a que la API esté lista (carga CLIP + FAISS, puede tardar ~30 s)
import requests as _req
for _ in range(60):
    try:
        _req.get("http://localhost:8000/health", timeout=2)
        break
    except Exception:
        time.sleep(2)

# Cargar el app de Gradio (ya en modo api, no carga encoder propio)
spec = importlib.util.spec_from_file_location(
    "_gradio_app", Path(__file__).parent / "app" / "app.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

if __name__ == "__main__":
    import torch
    torch.set_num_threads(1)
    mod.demo.launch(max_threads=1)
