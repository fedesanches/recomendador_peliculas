import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from src.recommender import MovieRecommender

recommender = MovieRecommender()


def recommend(image_path: str, top_k: int):
    if image_path is None:
        return [], "Por favor, sube una imagen de portada."

    results = recommender.recommend_from_image(image_path, top_k=int(top_k))

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
        top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Recomendaciones")

    recommend_btn = gr.Button("Buscar similares", variant="primary")
    gallery = gr.Gallery(label="Películas recomendadas", columns=5, height="auto")
    output = gr.Markdown()

    recommend_btn.click(
        fn=recommend,
        inputs=[image_input, top_k_slider],
        outputs=[gallery, output],
    )

if __name__ == "__main__":
    demo.launch()
