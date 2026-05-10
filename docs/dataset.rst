=======
Dataset
=======
Il dataset necessario per questo progetto è composto da articoli scaricati inviando una richiesta HTTP all'API di
|INSPIRE-HEP|. Per fare questo si utilizza lo script raw_final_dataset.py. Questo scrtipt contiene una funzione chiamata 
download_hep_ph_batches(batch_size, max_papers) che prende come argomenti il numero massimo di articoli da scaricare 
e la dimensione del batch di articoli. La funzione invia richieste HTTP all'API di |INSPIRE-HEP| divise in anni di
pubblicazione (1991-2024) degli articoli e filtrando per la categoria |arXiv| hep-ph. Per ogni anno di pubblicazione vengono scaricati
un numero di articoli pari alla dimensione del batch impostata fino ad arrivare al massimo di articoli da scaricare impostato.
Quindi alla fine si avranno max_papers articoli per ogni anno. Per rispettare le direttive dell'API di |INSPIRE-HEP|,
si consiglia di impostare un batch_size inferiore ai 700 e al massimo un max_papers di 9000. Inoltre si consiglia di impostare il
batch_size in modo tale che max_papers sia un multiplo. Infatti se max_papers non è un multiplo di batch_size, il numero
di articoli scaricati per anno sarà il multiplo di batch_size più grande di max_papers.
Per ogni articolo che viene scaricato si acquisisce titolo e abstract di |arXiv|, che verrà utilizzato come testo, e le
keywords di |INSPIRE-HEP|, che verranno usate come labels per i modelli di deep learning. Per ogni articolo viene creata una lista
di python contenente due dizionari, uno contenente il testo e uno la lista di keywords. Una volta eseguito lo script, o la funzione, viene
salvato nella cartella data_new/raw un file new_raw_dataset.json contenente il dataset. Nel repositorio è presente una cartella
data/raw contenente un dataset di default con più di 90000 articoli.
