import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("_app", Path(__file__).parent / "app" / "app.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

if __name__ == "__main__":
    mod.demo.launch()
