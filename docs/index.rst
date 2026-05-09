Documentazione del progetto per l'esame
=======================================

Il progetto dell'esame consiste nello sviluppo di un metodo di deep learning per classificare articoli scientifici
usando le keywords associate come labels.
Gli articoli vengono ottenuti dal database di INSPIRE-HEP, filtrando per la categoria arXiv hep-ph. I dati sono elaborati
e utilizzati per il training, la validazione e il test di tre modelli di deep learning. I modelli sono ottimizzati per
ottenere buone prestazioni di ricostruzione delle keywords.
Per poter utilizzare le funzioni e gli script del repositorio è necessario installare il pacchetto con pip su una versione
di python non superiore alla 3.11. Dopo la prima istallazione sarà necessario eseguire lo script utils.py o la funzione download_assets()
in una sessione python, per scaricare i dataset e i modelli di default.
Gli script e le funzioni del repositorio permettono di ottenere un nuovo dataset su cui allenare i modelli e ottenere delle predizioni.
Per maggiori informazioni e esempi di utilizzo consultare la sezione Utilizzo.

.. toctree::
   :maxdepth: 2
   :caption: Indice:

   dataset
   EDA
   models
   utilizzo
   