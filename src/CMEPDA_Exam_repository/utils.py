import requests
from pathlib import Path

from CMEPDA_Exam_repository import (
    CMEPDA_EXAM_REPOSITORY_NN_MODELS,
    CMEPDA_EXAM_REPOSITORY_DATA
)

def download_assets():
    # Mappa: URL del file -> Percorso locale
    base_url = "https://github.com/Andrea-Simonelli13/CMEPDA_Exam_repository/releases/download/v1.0.0"
    assets = {
        f"{base_url}/raw_final_dataset.json": CMEPDA_EXAM_REPOSITORY_DATA / "raw/raw_final_dataset.json",
        f"{base_url}/final_articles_normalized_optimal_clustering.json": CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_articles_normalized_optimal_clustering.json",
        f"{base_url}/final_keyword_map_optimal_clustering.json": CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_keyword_map_optimal_clustering.json",
        f"{base_url}/final_dataset_binary_lables_optimal_clustering.npy": CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_dataset_binary_lables_optimal_clustering.npy",
        f"{base_url}/final_keyword_binary_classes_optimal_clustering.npy": CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_keyword_binary_classes_optimal_clustering.npy",
        f"{base_url}/model_Dense_v32_BC_02_25000.weights.h5": CMEPDA_EXAM_REPOSITORY_NN_MODELS / "model_Dense_v32_BC_02_25000.weights.h5",
        f"{base_url}/vectorizer_Dense_v32_BC_02_25000.pkl": CMEPDA_EXAM_REPOSITORY_NN_MODELS / "vectorizer_Dense_v32_BC_02_25000.pkl",
        f"{base_url}/model_CNN_v44_BC_02.weights.h5": CMEPDA_EXAM_REPOSITORY_NN_MODELS / "model_CNN_v44_BC_02.weights.h5",
        f"{base_url}/vectorizer_CNN_v44_BC_02.pkl": CMEPDA_EXAM_REPOSITORY_NN_MODELS / "vectorizer_CNN_v44_BC_02.pkl",
        f"{base_url}/model_LSTM_v81_BC_02.weights.h5": CMEPDA_EXAM_REPOSITORY_NN_MODELS / "model_LSTM_v81_BC_02.weights.h5",
        f"{base_url}/vectorizer_LSTM_v81_BC_02.pkl": CMEPDA_EXAM_REPOSITORY_NN_MODELS / "vectorizer_LSTM_v81_BC_02.pkl"
    }

    for url, path in assets.items():
        if not path.exists():
            print(f"Scaricamento di {path.name}...")
            path.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(url, allow_redirects=True)
            r.raise_for_status() # Genera un errore se il file non esiste sul server
            with open(path, 'wb') as f:
                f.write(r.content)
    print("Tutti i modelli sono pronti!")

if __name__ == "__main__":
    download_assets()
