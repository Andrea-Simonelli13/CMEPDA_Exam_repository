import json
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

def keywords_binarization():
    '''Funzione che binarizza le keywords associate ad ogni articolo dopo che 
    è stato effettuato il clustering. La funzione salva sia le label che serviranno
    per allenare le reti, sia le classi per poter ricostruire le keywords.
    '''
    with open("data/processed/articles_normalized_optimal_clustering.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    #inputs = [article["text"] for article in data]

    labels_raw = [article["keywords"] for article in data]

    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels_raw).astype(np.int8)
    #print(labels)

    np.save("data/processed/keyword_binary_classes_optimal_clustering.npy", mlb.classes_)
    #with open("data/processed/dataset_binary_lables_bigger.json", "w", encoding="utf-8") as file:
        #json.dump(labels.tolist(), file, indent=2, ensure_ascii=False)
    np.save("data/processed/dataset_binary_lables_optimal_clustering.npy", labels)    

if __name__ == "__main__":
    keywords_binarization()
