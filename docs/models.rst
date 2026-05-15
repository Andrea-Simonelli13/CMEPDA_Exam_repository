========================
Modelli e Ottimizzazione
========================

Si è scelto di utilizzare tre modelli diversi per classificare gli articoli utilizzando le keywords come labels. I modelli sono stati
ottimizzati e i vari passaggi delle ottimizzazioni sono visibili nel pdf Ottimizzazione_modelli_CMEPDA.pdf presente nel repositorio.
Nella cartella models saranno disponibili i pesi di modelli pre-allenati, dopo aver eseguito lo script utils.py o la funzione download_assets(). Ciascun modello è implementato attraverso la libreria |tensorflow|,
che include |Keras|. Di conseguenza per utilizzare questi modelli sarà necessario avere una versione di |Python| non superiore
alla 3.11.
Il dataset è stato splittato prendendo l'80% come dati di training, il 10% come dati di validazione e il restante 10% come dati di test.
Nella funzione che separa il dataset è anche eseguito il grafico della distribuzione della lunghezza dei testi. Prima di essere processati,
i testi vengono ripuliti dai caratteri di Latex. Inoltre si fa una pulizia anche delle stopwords che possono
rendere più difficile l'apprendimento della rete. 
Prima di passare i testi al modello viene eseguita la vettorizzazione attraverso un layer di tipo TextVectorization. Questo
layer converte le stringhe di testo (titlo + abstract), in vettori contenenti numeri interi che corrispondo alle parole di un vocabolario
costruito con tutti gli articoli utilizzati per il training dalle reti. La lunghezza degli output del TextVectorization è scelta osservando
il grafico di distribuzione della lunghezza dei testi.
Ogni modello conterrà un layer di tipo Embedding. Questo layer prende in ingresso i testi vettorizzati e per ogni parola crea dei vettori che riescono a cogliere l'aspetto semantico, parole
simili saranno rappresentate da vettori simili.
Il primo modello è un modello composto principalmente da layer di tipo Dense. Questo modello è formato da:

- Embedding(input_dim=25001, output_dim=512)
- GlobalMaxPooling1D()
- BatchNormalization()
- Dense(1024, activation='relu')
- BatchNormalization()
- Dropout(0.3)
- Dense(1024, activation='relu')
- BatchNormalization()
- Dense(y_train.shape[1], activation='sigmoid')

Il secondo modello, oltre ai layer Dense ed Embedding, utilizza layers di tipo Conv1D che sono in grado di "osservare" più parole contemporaneamente
a seconda della grandezza del kernel. Il modello convoluzionale è formato da:

- Embedding(input_dim=25001, output_dim=512)
- SpatialDropout1D(0.2)
- Conv1D(filters=256, kernel_size=5, activation='relu', padding='same')
- BatchNormalization()
- Conv1D(filters=256, kernel_size=5, activation='relu', padding='same')
- BatchNormalization()
- GlobalMaxPooling1D()
- BatchNormalization()
- Dense(1024, activation='relu')
- BatchNormalization()
- Dropout(0.5)
- Dense(y_train.shape[1], activation='sigmoid')

Il terzo e ultimo modello invece utilizza un layer di tipo LSTM (Long-Short Term Memory), la rete è quindi composta da:

- Embedding(input_dim=25001, output_dim=512)
- Bidirectional(LSTM(128, retun_sequences=True, dropout=0.2))
- LayerNormalization()
- Bidirectional(LSTM(128, retun_sequences=True, dropout=0.2))
- LayerNormalization()
- GlobalMaxPooling1D()
- BatchNormalization()
- Dense(1024, activation='relu')
- BatchNormalization()
- Dense(y_train.shape[1], activation='sigmoid')

Per tutti i modelli come metriche si utilizzano l'F1-score con average micro, la Precision e la Recall, come optimizer adam e come loss la BinaryCrossentropy.
Durante il training dei modelli sono stati utilizzati i callbacks di EarlyStopping e ReduceLROnPlateau. Entrambi monitorano la funzione di loss
e l'EarlyStopping è settato per impostare i parametri con i quali si è ottenuto il valore migliore.
Il repositorio contiene uno script (train_models.py) che permette di allenare da zero i modelli proposti. Questo script contiene tre funzioni,
una per modello, che possono essere utilizzate in una sessione python fornendo come argomento una stringa.
Sarà quindi possibile allenare i modelli sia sul dataset di default passando alla funzione la stringa "default" così da utilizzare
i file:

- "data/processed/final_article_normalized_optimal_clustering.json"
- "data/processed/final_dataset_binary_classes_optimal_clustering.npy"

mentre per utilizzare i dataset ottenuti dagli script exploratory_data_analysis.py e keywords_binarization.py la stringa dovrà essere "new", per utilizzare
i file:

- "data_new/processed/new_article_clustering.json"
- "data_new/processed/new_dataset_binary_labels.npy"

I modelli allenati possono essere utilizzati per ottenere delle predizioni su nuovi testi. Nello script new_model_predictions.py sono definite diverse funzioni
a seconda dello scopo. Sarà infatti possibile ottenere alcune predizioni sui dati di test, dai quali si ricavano le prestazioni della rete al variare della soglia decisionale
delle metriche. Inoltre è possibile ottenere delle possibili keywords su nuovi articoli.
Se si vuole utilizzare i modelli pre-allenati sarà sufficiente utilizzare lo script model_predictions.py che ha le stesse funzionalità dello script precedente.