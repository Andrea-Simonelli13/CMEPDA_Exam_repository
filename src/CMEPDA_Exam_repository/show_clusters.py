'''In questo script è contenuta la funzione che stampa il contenuto
dei clusters ottenuti dal clustering e di quelli selezionati per
l'allenamento.
'''

import json
import numpy as np

from CMEPDA_Exam_repository import CMEPDA_EXAM_REPOSITORY_DATA_NEW

def show_clusters_content(min_cluster=0, max_cluster=50):
    '''Funzione che stampa il contenuto dei clusters e le keywords scelte
    per l'allenamento.
    Args:
         min_cluster (int): numero del primo cluster da stampare
         max_cluster (int): numero dell'ultimo cluster da stampare
    '''
    #apro il file in cui c'è la mappa delle keyword e la estraggo
    with open(CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_keywords_map.json", "r", encoding="utf-8") as f:
        map = json.load(f)
    #apro il file in cui c'è la lista delle keyword selezionate per il training
    classes_array = np.load(CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_keyword_binary_classes.npy", allow_pickle=True)
    classes_list = classes_array.tolist()
    #creo un dizionario in cui le key corrispondono alla keyword rappresentativa del cluster
    # e ad ongi key corrisponde la lista di keywords del cluster
    cluster_dict = {}
    for original_keyword, cluster_keyword in map.items():
        if cluster_keyword not in cluster_dict:
            cluster_dict[cluster_keyword] = []

        cluster_dict[cluster_keyword].append(original_keyword)

    #stampo i cluster da min_cluster a max_cluster con la sua keyword rappresentativa
    print('------------------------------------')
    print("Clusters ottenuti")
    print("keyword rappresentativa : [keywords]")
    print('------------------------------------')
    max_count = 0
    for cluster, keywords in cluster_dict.items():
        if max_cluster> max_count >= min_cluster:
            print(f'cluster numero {max_count + 1}')
            print(f'{cluster} : {keywords}')
        max_count += 1
    print('---------------------------------------------------------')
    print("Keywords selezionate per l'allenamento delle reti neurali")
    print('---------------------------------------------------------')
    print(classes_list)

    print(f'numero di keyword selezionate per il training = {len(classes_list)}')
    print(f'numero di cluster = {len(cluster_dict)}')
    print(f'keywords uniche totali = {len(map)}')

    with open(CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_clusters_content.json", "w", encoding="utf-8") as f:
        json.dump(cluster_dict, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    show_clusters_content()
