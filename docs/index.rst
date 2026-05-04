Documentazione del progetto per l'esame
=======================================

Questa documentazione è da aggiornare via via che vengono aggionti src code.
Il progetto dell'esame consiste nello sviluppo di un metodo di deep learning per classificare articoli scientifici
usando le keywords associate come labels.
Gli articoli vengono ottenuti dal database di HEP_INSPIRE, filtrando per la categoria arXiv hep-ph. I dati sono elaborati
e utilizzati per il training, la validazione e il test di tre modelli di deep learning. I modelli sono ottimizzati per
ottenere buone prestazioni di ricostruzione delle keywords.

.. toctree::
   :maxdepth: 2
   :caption: Indice:

   dataset
   EDA
   models
   utilizzo
   