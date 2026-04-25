import gradio as gr
from src.recommender import MovieRecommender

recommender = MovieRecommender()


def recommend(image_path: str, top_k: int) -> str:
    if image_path is None:
        return "Por favor, sube una imagen de portada."
    results = recommender.recommend_from_image(image_path, top_k=int(top_k))
    output = ""
    for _, row in results.iterrows():
        output += f"### {row['title']}  ·  score: {row['score']:.3f}\n"
        overview = row.get("overview", "")
        if overview:
            output += f"{overview}\n"
        output += "\n---\n"
    return output


with gr.Blocks(title="Recomendador de Películas por Portada") as demo:
    gr.Markdown("# Recomendador de Películas por Portada")
    gr.Markdown(
        "Sube la portada de una película y encontraremos las más similares usando CLIP."
    )

    with gr.Row():
        image_input = gr.Image(type="filepath", label="Portada de consulta")
        top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Recomendaciones")

    recommend_btn = gr.Button("Buscar similares", variant="primary")
    output = gr.Markdown()

    recommend_btn.click(
        fn=recommend,
        inputs=[image_input, top_k_slider],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch()
