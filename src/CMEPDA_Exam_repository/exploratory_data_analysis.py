'''In questo script sono contenute le funzioni necessarie per eseguire il clustering
delle keywords. Molto spesso gli articoli hanno keywords simili, quindi si esegue
il clustering per ridurne il numero.
'''

# importa il modulo json per leggere file JSON
import json
# importa Counter, una classe utile per contare quante volte appare ogni elemento
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
#importa sentence transformer per l'embedding delle keyword
from sentence_transformers import SentenceTransformer
#imposrta l'algoritmo di clustering
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.preprocessing import normalize
from kneed import KneeLocator

from CMEPDA_Exam_repository import CMEPDA_EXAM_REPOSITORY_DATA_NEW
#funzione per normalizzare il testo delle keywords, rende tutto minuscolo
#toglie eventuali trattini e underscore, toglie spazi indesiderati
def normalize_keywords(k):
    '''Funzione per normalizzare il testo delle keywords, rende tutto minuscolo,
    toglie eventuali trattini e underscore, toglie spazi indesiderati.
    
    '''
    k = k.lower()
    k = k.replace("-", " ")
    k = k.replace("_", " ")
    k = k.replace(":", " ")
    k = k.strip()
    return k

def find_optimal_threshold(embeddings, start=0.5, stop=1.5, step=0.05, plot=True):
    """Funzione che trova un valore ottimale di distance_threshold per AgglomerativeClustering.
    
    Args:
        embeddings (np.array): matrice NxD degli embeddings delle keyword.
        start (float): valore iniziale di threshold.
        stop (float): valore finale di threshold.
        step (float): passo per i valori di threshold.
        plot (bool): se True disegna il grafico clusters vs threshold.
        
    Returns:
        float: distance_threshold consigliato.
    """
    thresholds = np.arange(start, stop, step)
    cluster_counts = []

    for t in thresholds:
        print(f"valore distance_thrashold = {t:.2f}")
        #clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=t)
        clustering = Birch(
        threshold=0.5,#0.5,
        branching_factor=50,
        n_clusters=AgglomerativeClustering(n_clusters=None, distance_threshold=t)
        )
        clusters = clustering.fit_predict(embeddings)
        cluster_counts.append(len(set(clusters)))

    if plot:
        plt.plot(thresholds, cluster_counts, marker="o")
        plt.xlabel("distance threshold")
        plt.ylabel("number of clusters")
        plt.title("Cluster count vs threshold")
        plt.grid(True)
        plt.show()

    # Calcolo del punto "ginocchio" come soglia ottimale:
    # differenze consecutive del numero di cluster
    #diffs = np.diff(cluster_counts)
    #diffs = np.abs(np.diff(cluster_counts))
    # cerco il primo punto(indice) dove la riduzione diventa minima (curva più piatta)
    #idx = np.argmax(diffs >= -1)  # -1 significa che la riduzione rallenta molto
    #idx = np.argmin(diffs)
    #optimal_threshold = thresholds[idx]
    # Trova il punto di knee
    knee = KneeLocator(thresholds, cluster_counts, curve='convex', direction='decreasing')
    optimal_threshold = knee.knee
    print(f"Valore consigliato di distance_threshold: {optimal_threshold:.2f}")
    return optimal_threshold

def exploratory_data_analysis():
    '''Funzione che esegue l'Exploratory Data Anlysis delle keywords e ne esegue il clustering
    per ridurne il numero. Il clustering viene eseguito facendo un embedding delle keywords,
    i vettori vengono poi procesati con un algoritmo combinato BIRCH + AgglomerativeClustering.
    La funzione sostituisce in automatico le nuove keywords nella lista degli articoli e salva sia
    la nuova lista che la mappa che permette di ottenere le keywords da quelle nuove.
    '''
    print("inizio funzione EDA")
    json_folder = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "raw/new_raw_dataset.json"
    # apre il file JSON in modalità lettura ("r")
    # encoding="utf-8" serve per leggere correttamente caratteri speciali
    with open(json_folder, "r", encoding="utf-8") as f: #"data/raw/raw_final_dataset.json"

        # json.load legge il file e lo converte in una struttura Python
        # nel tuo caso diventa una lista di dizionari (uno per ogni articolo)
        data = json.load(f)


    # stampa il numero totale di articoli nel dataset
    # len(data) conta quanti elementi ci sono nella lista
    print("Numero articoli:", len(data))


    # crea una lista vuota che conterrà tutte le keyword di tutti gli articoli
    all_keywords = []

    #per ogni articolo prende la lista delle keywords e applica
    # la funzione per normalizzare il testo
    for article in data:
        article["keywords"] = [
            normalize_keywords(k) for k in article["keywords"]
        ]

    # scorre tutti gli articoli del dataset
    # ogni elemento d è un dizionario del tipo:
    # {"text": "...", "keywords": [...]}
    for d in data:

        # prende la lista di keyword dell'articolo
        # e la aggiunge alla lista globale all_keywords
        # extend aggiunge TUTTI gli elementi della lista
        all_keywords.extend(d["keywords"])

    #fa il grafico della frequenza delle keywords
    counter_raw = Counter(all_keywords)
    #freqs = sorted(counter_raw.values(), reverse=True)

    #plt.figure(figsize=(6,4))
    #plt.plot(freqs)

    #plt.xscale("log")
    #plt.yscale("log")

    #plt.xlabel("Keyword rank")
    #plt.ylabel("Frequency")
    #plt.title("Keyword frequency distribution (log-log)")

    #plt.show()

    # set(all_keywords) elimina i duplicati
    # quindi len(set(...)) conta quante keyword diverse esistono
    unique_keywords = list(set(all_keywords))
    print("Keyword uniche:", len(unique_keywords))


    # Counter conta quante volte appare ogni keyword
    # crea un dizionario del tipo:
    # {"dark matter": 1200, "supersymmetry": 850, ...}
    #counter = Counter(all_keywords)


    # most_common(20) restituisce le 20 keyword più frequenti
    # con il numero di occorrenze
    #print(counter.most_common(20))

    #min_freq = 50
    #valid_keywords = {k for k, v in counter.items() if v >= min_freq}

    #embedding
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(unique_keywords, show_progress_bar=True)
    #l'embedding è eseguito guardando alla cosine similarity. keywords semanticamente simili vengono
    #vettorizzate in modo tale che l'angolo tra i vettori sia piccolo.
    #Questi vettori però hanno lunghezze diverse.
    #Il clustering utilizzato dopo utilizza la distanza euclidea.
    # quindi vettori che sarebbero vicini per la cosine similarity,
    #potrebbero risultare distanti usando la distanza euclidea(distanza tra le "punte" dei vettori).
    #Di conseguenza è necessario normalizzare i vettori cosicchè avranno tutti lunghezza 1.
    # Facendo questo vettori con un angolo tra loro compreso piccolo
    #avranno una distanza euclidea piccola e viceversa.
    embeddings_norm = normalize(embeddings)

    optimal_value = find_optimal_threshold(embeddings=embeddings_norm)
    print(f"valore ottimale = {optimal_value:.2f}")

    #clustering
    #clustering = AgglomerativeClustering(
        #n_clusters=None,
        #distance_threshold=1.05#optimal_value#0.9  #0.4
    #)
    clustering = Birch(
        threshold=0.5,#0.5,
        branching_factor=50,
        n_clusters=AgglomerativeClustering(n_clusters=None, distance_threshold=optimal_value)
        )

    clusters = clustering.fit_predict(embeddings_norm)
    print(f"clusters list = {clusters}")
    print(f"Clustering completato. Creati {len(set(clusters))} cluster.")
    #associo ogni keyword al rispettivo cluster
    keyword_to_cluster = dict(zip(unique_keywords, clusters))
    #print(f"keyword to clusters = {keyword_to_cluster}")

    #creo un keyword rappresentativa del cluster prendendo la prima keyword associata al cluster
    #cluster_to_keyword = {}

    #for keyword, cluster in keyword_to_cluster.items():
        #if cluster not in cluster_to_keyword:
            #cluster_to_keyword[cluster] = keyword
    #print(f"cluster to keywords = {cluster_to_keyword}")

    #creo un contatore per i cluster
    cluster_counts = Counter()
    #per tutte le keyword che sono nel dizionario keyword: cluster aumento di 1
    # il conteggio che corrisponde alla key (cluster, keyword)
    for k in all_keywords:
        if k in keyword_to_cluster:
            cluster_counts[(keyword_to_cluster[k], k)] += 1
    #creo un dizionario che associa al numero del cluster la keyword più frequente del cluster
    cluster_to_keyword = {}
    #per ogni key del contatore ordinato in ordine decrescente
    #si guarda se al cluster è già stata assegnata
    #una keyword rappresentativa, se sì è quella più frequente,
    #altrimenti si assegna quella che è stata trovata.
    for (cluster, keyword), count in cluster_counts.most_common():
        if cluster not in cluster_to_keyword:
                cluster_to_keyword[cluster] = keyword


    #creo una mappa che permette di collegare
    #le keyword originali a quelle ottenute dopo il clustering
    keyword_map = {}

    for keyword, cluster in keyword_to_cluster.items():
        keyword_map[keyword] = cluster_to_keyword[cluster]

    #conta la frequeza delle keyword clusterate
    mapped_keywords = [keyword_map.get(k, k) for k in all_keywords]
    counter = Counter(mapped_keywords)

    #fa il grafico della frequenza delle keywords dopo il clustering
    #freqs_clustered = sorted(counter.values(), reverse=True)

    #plt.figure(figsize=(6,4))
    #plt.plot(freqs_clustered)

    #plt.xscale("log")
    #plt.yscale("log")

    #plt.xlabel("Clustered keyword rank")
    #plt.ylabel("Frequency")
    #plt.title("Clustered keyword frequency distribution")

    #plt.show()

    #grafico della distribuzione della frequenza delle keywords prima e dopo il clustering
    max_freq = max(max(counter_raw.values()), max(counter.values()))
    bins = np.arange(-0.5, max_freq + 0.5 ,1)
    plt.hist(counter_raw.values(), bins=bins, alpha=0.8, label="before clustering")
    plt.hist(counter.values(), bins=bins, alpha=0.5, label="after clustering")

    plt.xlim(0, 50)
    #plt.ylim(0, 10)
    plt.xlabel("Keyword frequency")
    plt.ylabel("Number of keywords")
    plt.title("Keyword frequency distribution")

    plt.legend()
    plt.show()

    print("freq media prima:", np.mean(list(counter_raw.values())))
    print("freq media dopo:", np.mean(list(counter.values())))

    #top_keywords = set([k for k, _ in counter.most_common(top_n_keywords)])
    #print(f"Top {top_n_keywords} keyword clusterate selezionate")

    #scelgo la frequenza minima che deve avere una keyword nel dataset
    min_occurency = 200
    filtered_counter = {k: v for k, v in counter.items() if v >= min_occurency}

    #calcolo quante occorrenze totali ci sono nel dataset dopo il clustering
    total_occurences = sum(filtered_counter.values())
    target_coverage = 0.85
    current_sum = 0
    top_keywords = set()

    sorted_filtered = sorted(filtered_counter.items(), key=lambda x: x[1], reverse=True)

    #estraggo le keywords finchè non raggiungo la copertura(target_coverage)
    for k, v in sorted_filtered:
        current_sum += v
        top_keywords.add(k)
        if (current_sum/total_occurences) >= target_coverage:
            break

    print(f"Keyword totali dopo clustering: {len(counter)}")
    print(f"Keyword con almeno {min_occurency} occorrenze: {len(filtered_counter)}")
    print(f"Selezionate automaticamente {len(top_keywords)} keyword per coprire l'{target_coverage*100}% del dataset.")
    #applico il mapping agli articoli, così da sostituire
    # le keyword con quelle ottenute dopo il clustering
    #for article in data:
        #article["keywords"] = [
            #keyword_map.get(k, k) for k in article["keywords"]
        #]
        #article["keywords"] = list(set(article["keywords"]))

    #applico la mappa alle keywords di ogni articolo e tengo solo le keyords più frequenti
    #contando il numero di articoli che rimangono senza keywords
    empty_article = 0
    for article in data:
        mapped = [keyword_map.get(k, k) for k in article["keywords"]]
        article["keywords"] = list({k for k in mapped if k in top_keywords})
        if len(article["keywords"])==0:
            empty_article = empty_article + 1
    print(f"{empty_article} articoli senza keywords")

    #tengo solo gli articoli con le keywords
    data = [a for a in data if len(a["keywords"]) > 0]
    print(f"Numero di articoli rimasti = {len(data)}")
    #salvo la mappa e gli articoli con le nuove keywords
    map_folder = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_keywords_map.json"
    with open(map_folder, "w", encoding="utf-8") as f: #"data/processed/final_keyword_map_optimal_clustering.json"
        json.dump(keyword_map, f, indent=2, ensure_ascii=False)

    article_folder = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_article_clustering.json"
    with open(article_folder, "w", encoding="utf-8") as f: #"data/processed/final_articles_normalized_optimal_clustering.json"
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    exploratory_data_analysis()
