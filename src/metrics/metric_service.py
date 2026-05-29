import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import MultiLabelBinarizer
from dataclasses import dataclass

INDEX_PATH          = "data/processed/faiss.index"
INDEX_META_PATH     = "data/processed/index_metadata.csv"
INDEX_COMBINED_PATH      = "data/processed/faiss_combined.index"
INDEX_COMBINED_META_PATH = "data/processed/index_metadata_combined.csv"
SIGLIP_INDEX_PATH               = "data/processed/faiss_siglip.index"
SIGLIP_INDEX_META_PATH          = "data/processed/index_metadata_siglip.csv"
SIGLIP_COMBINED_INDEX_PATH      = "data/processed/faiss_siglip_combined.index"
SIGLIP_COMBINED_INDEX_META_PATH = "data/processed/index_metadata_siglip_combined.csv"
DINOV2_INDEX_PATH      = "data/processed/faiss_dinov2.index"
DINOV2_INDEX_META_PATH = "data/processed/index_metadata_dinov2.csv"


@dataclass
class MetricResult:
    precision:        float
    recall:           float
    ndcg:             float
    gender_coherence: float
    aciertos:         int
    aciertos_pct:     float


def _jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    intersection = np.logical_and(y_true, y_pred).sum()
    union        = np.logical_or(y_true, y_pred).sum()
    return float(intersection / union) if union > 0 else 0.0


def _jaccard_all(y_true: np.ndarray, generos_binarios: np.ndarray) -> np.ndarray:
    intersection = np.logical_and(y_true, generos_binarios).sum(axis=1)
    union        = np.logical_or(y_true, generos_binarios).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, intersection / union, 0.0)


def _dcg(relevances) -> float:
    return sum(r / np.log2(i + 2) for i, r in enumerate(relevances))


def calculate(
    k:              int   = 5,
    n_samples:      int   = 500,
    umbral_jaccard: float = 0.5,
    combined:       bool  = False,
    model:          str   = "clip",
) -> MetricResult:
    if model == "siglip":
        if combined:
            index_path = SIGLIP_COMBINED_INDEX_PATH
            meta_path  = SIGLIP_COMBINED_INDEX_META_PATH
        else:
            index_path = SIGLIP_INDEX_PATH
            meta_path  = SIGLIP_INDEX_META_PATH
    elif model == "dinov2":
        index_path = DINOV2_INDEX_PATH
        meta_path  = DINOV2_INDEX_META_PATH
    else:
        index_path = INDEX_COMBINED_PATH if combined else INDEX_PATH
        meta_path  = INDEX_COMBINED_META_PATH if combined else INDEX_META_PATH
    index    = faiss.read_index(index_path)
    metadata = pd.read_csv(meta_path)

    metadata["genre_list"] = metadata["genres"].fillna("").apply(
        lambda x: [g.strip().lower() for g in x.split(", ")] if x else []
    )
    mlb              = MultiLabelBinarizer()
    generos_binarios = mlb.fit_transform(metadata["genre_list"])

    sample = metadata.sample(n_samples, random_state=42)

    precision_scores  = []
    recall_scores     = []
    ndcg_scores       = []
    coherence_scores  = []
    aciertos          = 0

    for idx, _ in sample.iterrows():
        vector   = index.reconstruct(idx)
        _, indices = index.search(vector.reshape(1, -1), k + 1)
        indices_recomendados = indices[0][1:]

        y_true = generos_binarios[idx]
        y_preds = generos_binarios[indices_recomendados]
        jaccards = np.array([_jaccard(y_true, yp) for yp in y_preds])

        # Precision@K: fracción de las K recomendaciones que superan el umbral
        relevantes  = int((jaccards >= umbral_jaccard).sum())
        aciertos   += relevantes
        precision_scores.append(relevantes / k)

        # Recall@K: fracción de todas las películas relevantes del dataset que recuperamos
        todos_jaccards   = _jaccard_all(y_true, generos_binarios)
        total_relevantes = max((todos_jaccards >= umbral_jaccard).sum() - 1, 1)  # -1 excluye la misma
        recall_scores.append(min(relevantes / total_relevantes, 1.0))

        # NDCG@K: ranking ponderado por posición usando Jaccard como relevancia
        dcg   = _dcg(jaccards)
        ideal = sorted(todos_jaccards, reverse=True)[1:k + 1]
        idcg  = _dcg(ideal)
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        # Coherencia de género: Jaccard promedio de las K recomendaciones
        coherence_scores.append(float(jaccards.mean()))

    total = n_samples * k
    return MetricResult(
        precision=        round(float(np.mean(precision_scores)),  4),
        recall=           round(float(np.mean(recall_scores)),     4),
        ndcg=             round(float(np.mean(ndcg_scores)),       4),
        gender_coherence= round(float(np.mean(coherence_scores)),  4),
        aciertos=         aciertos,
        aciertos_pct=     round(aciertos / total, 4),
    )


def save(result: MetricResult, path: str = "data/processed/metrics.json") -> None:
    import json
    from dataclasses import asdict
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(result), indent=2))

def calculate_and_save(combined: bool = False, model: str = "clip"):
    result = calculate(combined=combined, model=model)
    if model == "siglip":
        path = "data/metrics/metrics_siglip_combined.json" if combined else "data/metrics/metrics_siglip.json"
    elif model == "dinov2":
        path = "data/metrics/metrics_dinov2.json"
    else:
        path = "data/metrics/metrics_clips_combined.json" if combined else "data/metrics/metrics_clips.json"
    save(result, path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", action="store_true", help="Usar índice CLIP imagen+texto")
    parser.add_argument("--model", default="clip", choices=["clip", "siglip", "dinov2"], help="Encoder a evaluar")
    args = parser.parse_args()
    calculate_and_save(combined=args.combined, model=args.model)
    print(calculate(combined=args.combined, model=args.model))