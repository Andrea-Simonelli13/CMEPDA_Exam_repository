'''In questo script è contenuta la funzione che binarizza le liste di keywords
degli articoli per poterle utilizzare come labels nell'allenamento dei modelli.
'''

import json
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

from CMEPDA_Exam_repository import CMEPDA_EXAM_REPOSITORY_DATA_NEW

def keywords_binarization():
    '''Funzione che binarizza le keywords associate ad ogni articolo dopo che 
    è stato effettuato il clustering. La funzione salva sia le label che serviranno
    per allenare le reti, sia le classi per poter ricostruire le keywords.
    '''
    with open(CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_article_clustering.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    labels_raw = [article["keywords"] for article in data]

    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels_raw).astype(np.int8)

    np.save(CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_keyword_binary_classes.npy", mlb.classes_)

    np.save(CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_dataset_binary_lables.npy", labels)

if __name__ == "__main__":
    keywords_binarization()
