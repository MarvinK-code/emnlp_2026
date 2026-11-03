import pandas as pd
from sentence_transformers import SentenceTransformer


CSV_PATH = "data/emnlp2026_papers.csv"
OUT_PATH = "data/embeddings.parquet"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    data = pd.read_csv(CSV_PATH)
    data = data.dropna(subset=["abstract"]).reset_index(drop=True)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        data["abstract"].tolist(),
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    similarities = model.similarity(embeddings, embeddings)
    print(similarities.shape)

    out = data[["paper_id", "title"]].copy()
    out["embedding"] = list(embeddings)
    out.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(out)} embeddings to {OUT_PATH}")


if __name__ == "__main__":
    main()
