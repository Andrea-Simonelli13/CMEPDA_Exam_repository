=========================
Exploratory Data Analysis
=========================
Dopo aver scaricato il dataset si prosegue con l'Exploratory Data Analysis (EDA) delle keywords. Nel dataset si avranno
molte keywords simili oppure che compaiono una sola volta. Attraverso lo script exploratory_data_analysis.py si riduce il numero di keywords
che la rete dovrà essere in grado di ricostruire dal testo.
Per prima cosa si contano le keywords uniche e si "puliscono" le stringhe delle keywords da elementi come trattini, underscore
e i due punti. Successivamente si esegue un embedding utilizzando un modello pre-allenato di SentenceTransformer ("all-MiniLM-L6-v2").
Questo modello vettorizza le parole in modo tale che parole simili siano rappresentate da vettori
con un'alta cosine similarity, ovvero in modo tale che vettori di parole simili abbiano un piccolo angolo compreso tra essi.
Successivamente è necessario normalizzare i vettori in output dal SentenceTransformer. Si fa questo perchè per ridurre il numero di keywords
si esegue un algoritmo unsupervised di clustering. Per il clustering si utilizzano due algoritmi combinati: Birch e AgglomerativeClustering.
L'algoritmo BIRCH utlizza la distanza euclidea tra due vettori per raggrupparli in cluster e i cluster creati dal BIRCH sono passati all'
AgglomerativeClustering. Si usa anche questo secondo algoritmo perchè non è necessario specificare il numero di cluster che si vuole
ottenere, ma i cluster del BIRCH vengono raggruppati in modo tale da minimizzare la varianza interna al cluster e creare cluster simili.
Più nel dettaglio, inizialmente all'AgglomerativeClustering sono forniti i cluster processati dall'algoritmo BIRCH. L'AgglomerativeClustering
cerca di unire a coppie tutti questi cluster in modo tale da avere l'aumento minore di dispersione. L'algoritmo perferirà unire
due cluster piccoli piuttosto che uno grande e uno piccolo, a meno che quest'ultimi non siano vicinissimi. L'AgglomerativeClustering
prende come parametro distance_threshold che indica il limite massimo di varianza che si è disposti ad osservare. Per scegliere questo parametro
si grafica il numero di cluster ottenuti in funzione del valore di distance_threshold, e tramite il metodo KneeLocator, si trova il valore ottimale da inserire.
Il KneeLocator serve a trovare il punto in cui non si ha più  una diminuzione significativa del numero di cluster.
Una volta che l'algoritmo ha ridotto il numero di clusters, viene creata un mappa (dizionario di python) che associa le vecchie keywords
a quelle nuove. Come nuove keywords sono scelte le più frequenti di ogni cluster.
A questo punto si filtrano le keywords scegliendo solo quelle con una frequenza di almeno 200 e si considerano solo le keywords
più frequenti fino a raggiungere una copertura minima dell'85% del totale delle keywords.
Infine utilizzando la mappa si sostituiscono le keywords vecchie con quelle nuove, scartando gli articoli che rimangono senza keywords.
Alla fine della funzione nella cartella data_new/processed è possibile trovare il dataset con le nuove keywords nel file new_article_clustering.json e la mappa
delle keywords nel file new_keywords_map.json. Dei file di default sono contenuti nella cartella data/processed.

Dopo lo script o la funzione exploratory_data_analysis deve essere sempre utilizzato lo script o la funzione keywords_binarization. Questo script contiene una funzione che binarizza le liste di keywords
di ogni articolo. Per fare questo viene utilizzato un modulo di scikit-learn chiamato MultiLabelBinarizer che prende in ingresso la lista di keywords di ogni
articolo e le binarizza trasformandole in liste di dimensione pari al numero di keywords totali. Queste liste avranno il valore '1' nella posizione corrispondente
alle keywords della lista originale. Le liste così ottenute sono salvate in un file new_dataset_binary_labels.npy, così come le classi del binarizer nel file new_keyword_binary_classes.npy
nella cartella data_new/processed. Dei file di default sono disponibili nella cartella data/processed.
Le classi serviranno a ricostruire le keywords. Le liste binarizzate serviranno come label per l'allenamento dei modelli.
Per osservare le nuove keywords e il contenuto dei cluster è necessario l'utilizzo dello script show_clusters.py che contiene la funzione show_clusters_content(). Questa funzione prende come argomenti
il primo e l'ultimo cluster che si vuole stampare. Di default è impostato l'intervallo [0,50]; solo se si utilizza direttamente la funzione è possibile modificare l'intervallo. Se viene lanciato lo script
da terminale, saranno stampate le prime 50 kyewords rappresentative con la lista di keywords che fanno parte di quel cluster. Inoltre, vengono stampate le keywords selezionate per l'allenamento delle reti neurali.
I clusters e la lista di keywords al loro interno è salvato nel file new_cluster_content.json nella cartella data_new/processed. Un file di default è salvato nella cartella data/processed. 
