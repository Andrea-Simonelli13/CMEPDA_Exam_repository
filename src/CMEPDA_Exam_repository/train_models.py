'''In questo script sono contenute le funzioni utili per allenare i modelli
sia sul dataset di default che su quello ottenuto dagli script exploratory_data_analysis.py
e keywords_binarization.py.
'''

import gc
import json
import pickle
import re
import sys
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import tensorflow
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    LayerNormalization,
    LSTM,
    SpatialDropout1D,
    TextVectorization,
)
from tensorflow.keras.models import Sequential

from CMEPDA_Exam_repository import (
    CMEPDA_EXAM_REPOSITORY_NEW_MODELS,
    CMEPDA_EXAM_REPOSITORY_DATA,
    CMEPDA_EXAM_REPOSITORY_DATA_NEW
    )

def clean_text(text):
    '''Funzione che pulisce il testo da caratteri
    che possono confondere la rete.
    Args:
         text (string): testo da pulire
    Returns:
          " ".join(clean_words) (string): testo pulito
    '''
    # 1. Rimuove il simbolo $ spesso utilizzato in LaTex
    text = text.replace('$', '')
    # 2. Rimuove il simbolo \ che spesso è utilizzato in latex
    # per i simboli come alpha, beta, tau etc.
    text = text.replace('\\', '')
    # 3. Rimuove tutto ciò che è tra due parentesi graffe comprese le graffe
    text = re.sub(r'\{.*?\}', ' ', text)
    # 4. Rimuove tutto ciò che non è una lettera
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # 5. Lowercase e rimozione stop-words
    stop_words = set(stopwords.words('english'))
    words = text.lower().split()
    clean_words = [w for w in words if w not in stop_words and len(w) > 1]

    return " ".join(clean_words)

def data_splitting(texts_file, label_file):
    '''Funzione che divide il dataset (testi e labels) in dataset
    di allenamento, validazione e test.
    Args:
         text_file (string): path al file degli articoli
         label_file (string): path al file delle labels
    Returns:
         X_train (list): lista degli articoli per il training
         X_val (list): lista degli articoli per la validazione
         X_test (list): lista degli articoli per il test
         y_train (nparray): array numpy delle labels per il training
         y_val (nparray): array numpy delle labels per la validazione
         y_test (nparray): array numpy delle labels per il test
    '''
    #carica il dataset
    with open(texts_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    #pulisce il testo con la funzione precedente
    texts = [clean_text(article["text"]) for article in data]
    #cancella dalla memoria il datset
    del data
    gc.collect()
    #crea una lista con la lunghezza dei testi
    #ed esegue il grafico della distribuzione della lunghezza dei testi
    lengths =[len(t.split()) for t in texts]
    freq = Counter(lengths)
    print(f"Lunghezza massima = {max(lengths)}")

    x = list(freq.keys())#lunghezza per sequenza
    y = list(freq.values())#frequenza

    plt.bar(x, y)
    plt.xlabel("Lunghezza testo")
    plt.ylabel("Frequenza")
    plt.title("Distribuzione lunghezze testi")
    plt.show()
    #carica le label binarie
    labels = np.load(label_file).astype(np.int8)

    X_temp, X_test, y_temp, y_test = train_test_split(texts, labels, test_size=0.10, random_state=42)
    del texts, labels
    gc.collect()
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42)
    del X_temp, y_temp
    gc.collect()
    return X_train, X_val, X_test, y_train, y_val, y_test

def plot_training_history_f1score(history):
    """
    Visualizza i grafici di Loss, Precision, Recall e F1-Score per monitorare il training.
    Args:
         history (history): history del training del modello.
    """

    plt.figure(figsize=(16, 4))#20 #5

    # 1. Plot della LOSS
    plt.subplot(1, 4, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.legend()

    # 2. Plot della PRECISION
    plt.subplot(1, 4, 2)
    plt.plot(history.history['precision'], label='Train Precision')
    plt.plot(history.history['val_precision'], label='Val Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision History')
    plt.legend()

    # 3. Plot della RECALL
    plt.subplot(1, 4, 3)
    plt.plot(history.history['recall'], label='Train Recall')
    plt.plot(history.history['val_recall'], label='Val Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall History')
    plt.legend()

     # --- Grafico 4: F1-SCORE MICRO---
    plt.subplot(1, 4, 4)
    plt.plot(history.history['f1_score'], label='Train F1-Score')
    plt.plot(history.history['val_f1_score'], label='Val F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.title('F1-Score History')
    plt.legend()

    plt.tight_layout() # Evita che i titoli si sovrappongano
    plt.show()

def train_model_Dense(dataset="default"):
    '''Funzione per allenare il modello Dense.
    Args:
         dataset (string): dataset da utilizzare, "default" utilizza il dataset di default
                           "new" utilizza il dataset ottenuto dagli script.
    '''
    nltk.download('stopwords')
    if dataset == "default":
        path_json = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_articles_normalized_optimal_clustering.json"
        path_npy = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_dataset_binary_lables_optimal_clustering.npy"
        print("Dataset di default selezionato")

    if dataset == "new":
        path_json = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_article_clustering.json"
        path_npy = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_dataset_binary_lables.npy"
        print("Dataset nuovo selezionato")

    #Si passano i file alla funzione per pulire e dividere il dataset.
    X_train, X_val, X_test, y_train, y_val, y_test = data_splitting(
        path_json,
        path_npy
    )
    print("dataset caricato e splittato")
    #Si definisce il vectorizer e si vettorizzano i testi.
    vectorizer = TextVectorization(
        max_tokens=25000,
        output_mode='int',
        output_sequence_length=200
    )

    vectorizer.adapt(X_train)

    print("Train:", len(X_train))
    print("Validation:", len(X_val))
    print("Test:", len(X_test))
    print("Label shape:", y_train.shape)

    X_train_vect = vectorizer(np.array(X_train)).numpy().astype('int32')
    X_val_vect = vectorizer(np.array(X_val)).numpy().astype('int32')
    X_test_vect = vectorizer(np.array(X_test)).numpy().astype('int32')
    del X_train
    del X_val
    del X_test
    gc.collect()
    #si implementa un early stop in modo tale che i pesi della rete
    #siano quelli dell'epoca con il valore della funzione di loss migliore.
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True, # Ripristina i pesi dell'epoca migliore
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,       # Riduce il LR a 1/5
        patience=4,       # Aspetta 4 epoche di "stallo" prima di intervenire
        min_lr=1e-6,      # Non scende sotto questa soglia
        verbose=1         # Avvisa con un messaggio quando interviene
    )
    #si definisce la struttura del modello.
    model_Dense = Sequential([
        Embedding(input_dim=25001, output_dim = 512),
        GlobalMaxPooling1D(),
        BatchNormalization(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dense(y_train.shape[1], activation='sigmoid')
    ])
    #si compila il modello scegliendo la funzione di loss, l'optimizer e le metriche
    model_Dense.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[
            tensorflow.keras.metrics.Precision(name='precision'),
            tensorflow.keras.metrics.Recall(name='recall'),
            tensorflow.keras.metrics.F1Score(average='micro', threshold=0.5, name='f1_score')
        ]
    )
    #si allena il modello passandogli i dataset necessari e stabilendo il numero
    #di epoche e il batch_size
    history = model_Dense.fit(
        X_train_vect,
        y_train,
        epochs=80,
        batch_size=256,
        validation_data=(X_val_vect, y_val),
        callbacks=[reduce_lr, early_stop]
    )
    #si salvano i pesi del modello per poterli riutilizzare
    model_Dense.save_weights(CMEPDA_EXAM_REPOSITORY_NEW_MODELS / 'new_model_Dense.weights.h5')
    #si graficano le funzione di loss e le metriche per ogni epoca
    plot_training_history_f1score(history=history)

    print("--------------------------------------------TEST-----------------------------------")
    model_Dense.evaluate(X_test_vect, y_test)

    #si salva il vocabolario del vectorizer
    with open(CMEPDA_EXAM_REPOSITORY_NEW_MODELS / 'new_vocabulary_Dense.pkl', "wb") as f:
     pickle.dump(vectorizer.get_vocabulary(), f)

    del X_train_vect, X_val_vect, X_test_vect, y_train, y_val, y_test
    gc.collect()



def train_model_CNN(dataset="default"):
    '''Funzione per allenare il modello CNN.
    Args:
         dataset (string): dataset da utilizzare, "default" utilizza il dataset di default
                           "new" utilizza il dataset ottenuto dagli script.
    '''
    nltk.download('stopwords')

    if dataset == "default":
        path_json = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_articles_normalized_optimal_clustering.json"
        path_npy = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_dataset_binary_lables_optimal_clustering.npy"
        print("Dataset di default selezionato CNN")

    if dataset == "new":
        path_json = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_article_clustering.json"
        path_npy = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_dataset_binary_lables.npy"
        print("Dataset nuovo selezionato CNN")


    #Si passano i file alla funzione per pulire e dividere il dataset.
    X_train, X_val, X_test, y_train, y_val, y_test = data_splitting(
        path_json,
        path_npy
    )
    print("dataset caricato e splittato")
    #Si definisce il vectorizer e si vettorizzano i testi.
    vectorizer = TextVectorization(
        max_tokens=25000,
        output_mode='int',
        output_sequence_length=200
    )

    vectorizer.adapt(X_train)

    print("Train:", len(X_train))
    print("Validation:", len(X_val))
    print("Test:", len(X_test))
    print("Label shape:", y_train.shape)
    
    X_train_vect = vectorizer(np.array(X_train)).numpy().astype('int32')
    X_val_vect = vectorizer(np.array(X_val)).numpy().astype('int32')
    X_test_vect = vectorizer(np.array(X_test)).numpy().astype('int32')
    del X_train
    del X_val
    del X_test
    gc.collect()
    #si implementa un early stop in modo tale che i pesi della rete
    #siano quelli dell'epoca con il valore della funzione di loss migliore.
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True, # Ripristina i pesi dell'epoca migliore
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,       # Riduce il LR a 1/5 (es. da 0.001 a 0.0002)
        patience=4,       # Aspetta 4 epoche di "stallo" prima di intervenire
        min_lr=1e-6,      # Non scende sotto questa soglia
        verbose=1         # Avvisa con un messaggio quando interviene
    )
    #si definisce la struttura del modello.
    model_CNN = Sequential([
        Embedding(input_dim=25001, output_dim = 512),
        SpatialDropout1D(0.2),
        Conv1D(filters=256, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(filters=256, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        GlobalMaxPooling1D(),
        BatchNormalization(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(y_train.shape[1], activation='sigmoid')
    ])
    #si compila il modello scegliendo la funzione di loss, l'optimizer e le metriche
    model_CNN.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[
            tensorflow.keras.metrics.Precision(name='precision'),
            tensorflow.keras.metrics.Recall(name='recall'),
            tensorflow.keras.metrics.F1Score(average='micro', threshold=0.5, name='f1_score')
        ]
    )
    #si allena il modello passandogli i dataset necessari e stabilendo il numero
    #di epoche e il batch_size
    history = model_CNN.fit(
        X_train_vect,
        y_train,
        epochs=100,
        batch_size=256,
        validation_data=(X_val_vect, y_val),
        callbacks=[reduce_lr, early_stop]
    )
    #si salvano i pesi del modello per poterli riutilizzare
    model_CNN.save_weights(CMEPDA_EXAM_REPOSITORY_NEW_MODELS / 'new_model_CNN.weights.h5')
    #si graficano le funzione di loss e le metriche per ogni epoca
    plot_training_history_f1score(history=history)

    print("--------------------------------------------TEST-------------------------------------------------")
    model_CNN.evaluate(X_test_vect, y_test)

    #si salva il vocabolario del vectorizer
    with open(CMEPDA_EXAM_REPOSITORY_NEW_MODELS / 'new_vocabulary_CNN.pkl', "wb") as f:
     pickle.dump(vectorizer.get_vocabulary(), f)

    del X_train_vect, X_val_vect, X_test_vect, y_train, y_val, y_test
    gc.collect()


def train_model_LSTM(dataset="default"):
    '''Funzione per allenare il modello LSTM.
    Args:
         dataset (string): dataset da utilizzare, "default" utilizza il dataset di default
                           "new" utilizza il dataset ottenuto dagli script.
    '''
    nltk.download('stopwords')

    if dataset == "default":
        path_json = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_articles_normalized_optimal_clustering.json"
        path_npy = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_dataset_binary_lables_optimal_clustering.npy"
        print("Dataset di default selezionato LSTM")

    if dataset == "new":
        path_json = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_article_clustering.json"
        path_npy = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_dataset_binary_lables.npy"
        print("Dataset nuovo selezionato LSTM")


    #Si passano i file alla funzione per pulire e dividere il dataset.
    X_train, X_val, X_test, y_train, y_val, y_test = data_splitting(
        path_json,
        path_npy
    )
    print("dataset caricato e splittato")
    #Si definisce il vectorizer e si vettorizzano i testi.
    vectorizer = TextVectorization(
        max_tokens=25000,
        output_mode='int',
        output_sequence_length=200
    )

    vectorizer.adapt(X_train)

    print("Train:", len(X_train))
    print("Validation:", len(X_val))
    print("Test:", len(X_test))
    print("Label shape:", y_train.shape)
    
    X_train_vect = vectorizer(np.array(X_train)).numpy().astype('int32')
    X_val_vect = vectorizer(np.array(X_val)).numpy().astype('int32')
    X_test_vect = vectorizer(np.array(X_test)).numpy().astype('int32')
    del X_train
    del X_val
    del X_test
    gc.collect()
    #si implementa un early stop in modo tale che i pesi della rete
    #siano quelli dell'epoca con il valore della funzione di loss migliore.
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True, # Ripristina i pesi dell'epoca migliore
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,       # Riduce il LR a 1/5
        patience=4,       # Aspetta 4 epoche di "stallo" prima di intervenire
        min_lr=1e-6,      # Non scende sotto questa soglia
        verbose=1         # Avvisa con un messaggio quando interviene
    )
    #si definisce la struttura del modello.
    model_LSTM = Sequential([
        Embedding(input_dim=25001, output_dim = 512),
        Bidirectional(LSTM(128, return_sequences= True, dropout=0.2)),
        LayerNormalization(),
        Bidirectional(LSTM(128, return_sequences= True, dropout=0.2)),
        LayerNormalization(),
        GlobalMaxPooling1D(),
        BatchNormalization(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dense(y_train.shape[1], activation='sigmoid')
    ])
    #si compila il modello scegliendo la funzione di loss, l'optimizer e le metriche
    model_LSTM.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[
            tensorflow.keras.metrics.Precision(name='precision'),
            tensorflow.keras.metrics.Recall(name='recall'),
            tensorflow.keras.metrics.F1Score(average='micro', threshold=0.5, name='f1_score')
        ]
    )
    #si allena il modello passandogli i dataset necessari e stabilendo il numero
    #di epoche e il batch_size
    history = model_LSTM.fit(
        X_train_vect,
        y_train,
        epochs=100,
        batch_size=256,
        validation_data=(X_val_vect, y_val),
        callbacks=[reduce_lr, early_stop]
    )
    #si salvano i pesi del modello per poterli riutilizzare
    model_LSTM.save_weights(CMEPDA_EXAM_REPOSITORY_NEW_MODELS / 'new_model_LSTM.weights.h5')
    #si graficano le funzione di loss e le metriche per ogni epoca
    plot_training_history_f1score(history=history)

    print("--------------------------------------------TEST-------------------------------------------------")
    model_LSTM.evaluate(X_test_vect, y_test)

    #si salva il vocabolario del vectorizer
    with open(CMEPDA_EXAM_REPOSITORY_NEW_MODELS / 'new_vocabulary_LSTM.pkl', "wb") as f:
     pickle.dump(vectorizer.get_vocabulary(), f)

    del X_train_vect, X_val_vect, X_test_vect, y_train, y_val, y_test
    gc.collect()


def help():
    print('Per allenare il modello selezionare uno dei modelli a disposizione: Dense, CNN, LSTM')
    print('Inoltre, è necessario scrivere default se si vuole usare i file di default,' \
    ' new se si vuole utilizzare il nuovo dataset.')
    print('Esempio: python train_models.py Dense new')

def train_model():
    modelli = {
        "Dense": train_model_Dense,
        "CNN": train_model_CNN,
        "LSTM": train_model_LSTM,
        "help": help
    }
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        modelli["help"]()
        return
    modello = sys.argv[1]

    if modello in modelli:
        if modello != "help":
            if len(sys.argv) == 3:
                modelli[modello](sys.argv[2])
            else:
                modelli["help"]()
        else:
            modelli["help"]()
    else:
        modelli["help"]()
        
if __name__ == "__main__":
    train_model()
