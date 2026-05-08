'''Questo script contiene le funzioni per ottenere le predizioni sui dati
di test e su nuovi articoli.
'''
import gc
import json
import pickle
import re
import sys

import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.layers import (
    Bidirectional, BatchNormalization, Conv1D,
    Dropout, Dense, Embedding, GlobalMaxPooling1D,
    LSTM, LayerNormalization, SpatialDropout1D,
    TextVectorization
)
from tensorflow.keras.models import Sequential

from CMEPDA_Exam_repository import (
    CMEPDA_EXAM_REPOSITORY_NN_MODELS,
    CMEPDA_EXAM_REPOSITORY_DATA
)

#-----------------------------Funzione che pulisce i testi-----------------------------------------
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
    # 4. Rimuove tutto ciò che non è una lettera, numeri, punti o virgole
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # 5. Lowercase e rimozione stop-words
    stop_words = set(stopwords.words('english'))
    words = text.lower().split()
    clean_words = [w for w in words if w not in stop_words and len(w) > 1]

    return " ".join(clean_words)
#-----------------Funzione che divide il dataset in training, vlidation e test------------------
def data_splitting_pred(texts_file, label_file):
    '''Funzione che divide il dataset (testi e labels e kewords) in dataset
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
         orig_train (list): lista delle keywords originali per il training
         orig_val (list): lista delle keywords originali per la validazione
         orig_test (list): lista delle keywords originali per il test
    '''
    with open(texts_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [clean_text(article["text"]) for article in data]
    original_keywords = [article["keywords"] for article in data]
    del data
    gc.collect()

    labels = np.load(label_file).astype(np.int8)

    X_temp, X_test, y_temp, y_test, orig_temp, orig_test = train_test_split(texts, labels, original_keywords, test_size=0.10, random_state=42)
    del texts, labels
    gc.collect()
    X_train, X_val, y_train, y_val, orig_train, orig_val = train_test_split(X_temp, y_temp, orig_temp, test_size=0.1111, random_state=42)
    del X_temp, y_temp
    gc.collect()
    return X_train, X_val, X_test, y_train, y_val, y_test, orig_train, orig_val, orig_test
#-----------------------------Predizioni del modello Dense sui dati di test----------
def Dense_model_prediction():
    '''Funzione che stampa le predizioni del modello Dense di default sui dati di test e
    crea i grafici delle metriche in funzione della soglia decisionale.
    '''
    nltk.download('stopwords')

    print("Verranno utilizzati i file presenti nelle cartelle models e data/processed")

    weights_path = CMEPDA_EXAM_REPOSITORY_NN_MODELS / "model_Dense_v32_BC_02_25000.weights.h5"
    vectorizer_path = CMEPDA_EXAM_REPOSITORY_NN_MODELS / "vectorizer_Dense_v32_BC_02_25000.pkl"

    data_path = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_articles_normalized_optimal_clustering.json"
    label_path = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_dataset_binary_lables_optimal_clustering.npy"
    classes_path = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_keyword_binary_classes_optimal_clustering.npy"

    with open(vectorizer_path, 'rb') as f:
        vectorizer_vocab = pickle.load(f)

    X_train, X_val, X_test, y_train, y_val, y_test, orig_train, orig_val, orig_test = data_splitting_pred(
        data_path,
        label_path
    )

    vectorizer = TextVectorization(max_tokens=25000, output_mode='int', output_sequence_length=200)
    vectorizer.set_vocabulary(vectorizer_vocab)

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

    model_Dense.build(input_shape=(None, 200))

    model_Dense.load_weights(weights_path)
    print("Modello caricato con successo tramite pesi!")

    model_Dense.summary()

    classes = np.load(classes_path, allow_pickle=True)
    mlb = MultiLabelBinarizer()
    mlb.classes_ = classes


    X_test_vect = vectorizer(np.array(X_test)).numpy().astype('int32')
    prediction = model_Dense.predict(X_test_vect)

    predicted_keywords = []
    soglie = [0.3, 0.35, 0.40, 0.45, 0.50, 0.55, 0.6]

    for soglia in soglie:
        predicted_keywords.append(mlb.inverse_transform((prediction>soglia).astype(int)))

    for i in range (0, 30):
        print(f"Articolo {i} del dataset di test")
        print(f"Keywords predette con soglia 0.5: {predicted_keywords[4][i]}")
        print(f"Keywords predette con soglia 0.3: {predicted_keywords[0][i]}")
        print(f"Keywords originali : {orig_test[i]}")
        print(' ')

    len_original = [len(o) for o in orig_test]
    print(f"Media delle keywords degli articoli originali : {np.mean(len_original):.2f}")

    contatore = 0
    for id_soglia in range(len(soglie)):
        len_predictions = [len(p) for p in predicted_keywords[id_soglia]]
        print(f'Media keywords predette soglia {soglie[id_soglia]:.2f}: {np.mean(len_predictions):.2f}')
        contatore += 1

    precisions = []
    recalls = []
    f1_scores = []

    for soglia in soglie:
        prediction_soglia = (prediction > soglia).astype(int)
        precision = precision_score(y_test, prediction_soglia, average='micro')
        precisions.append(precision)
        recall = recall_score(y_test, prediction_soglia, average='micro')
        recalls.append(recall)
        f1 = f1_score(y_test, prediction_soglia, average='micro')
        f1_scores.append(f1)
        print(f'Valori delle metriche con threshold {soglia:.2f}: precision = {precision:.4f}, recall = {recall:.4f}, F1-Score = {f1:.4f}')


    plt.figure(figsize=(15, 5))

    # 1. Plot della Precision
    plt.subplot(1, 3, 1)
    plt.plot(soglie, precisions, label='Precision', marker='o')
    plt.xlabel('Soglia')
    plt.ylabel('Precision')
    plt.title('Soglia vs Precision')

    # 2. Plot della Recall
    plt.subplot(1, 3, 2)
    plt.plot(soglie, recalls, label='Recall', marker='o')
    plt.xlabel('Soglia')
    plt.ylabel('Recall')
    plt.title('Soglia vs Recall')

    # 2. Plot del'F1
    plt.subplot(1, 3, 3)
    plt.plot(soglie, f1_scores, label='F1-Score', marker='o')
    plt.xlabel('Soglia')
    plt.ylabel('F1-Score')
    plt.title('Soglia vs F1-Score')

    plt.tight_layout() # Evita che i titoli si sovrappongano
    plt.show()
#---------------------------------------------Predizioni del modello su nuovi testi------------------
def Dense_model_new_prediction(new_text):
    '''Funzione che stampa le predizioni del modello Dense di default sui testi
    di nuovi articoli.
    Args:
         new_text (list): lista dei testi dei nuovi articoli
    '''
    nltk.download('stopwords')

    weights_path = CMEPDA_EXAM_REPOSITORY_NN_MODELS / "model_Dense_v32_BC_02_25000.weights.h5"
    vectorizer_path = CMEPDA_EXAM_REPOSITORY_NN_MODELS / "vectorizer_Dense_v32_BC_02_25000.pkl"
    classes_path = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_keyword_binary_classes_optimal_clustering.npy"

    with open(vectorizer_path, 'rb') as f:
        vectorizer_vocab = pickle.load(f)

    vectorizer = TextVectorization(max_tokens=25000, output_mode='int', output_sequence_length=200)
    vectorizer.set_vocabulary(vectorizer_vocab)

    model_Dense = Sequential([
        Embedding(input_dim=25001, output_dim = 512),
        GlobalMaxPooling1D(),
        BatchNormalization(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dense(567, activation='sigmoid')
    ])

    model_Dense.build(input_shape=(None, 200))

    model_Dense.load_weights(weights_path)
    print("Modello caricato con successo tramite pesi!")

    model_Dense.summary()

    classes = np.load(classes_path, allow_pickle=True)
    mlb = MultiLabelBinarizer()
    mlb.classes_ = classes

    cleaned_texts = [clean_text(text) for text in new_text]
    new_texts_vect = vectorizer(np.array(cleaned_texts)).numpy().astype('int32')
    prediction = model_Dense.predict(new_texts_vect)
    predicted_keywords = mlb.inverse_transform((prediction>0.5).astype(int))
    for predicted in predicted_keywords:
        print(predicted)


def CNN_model_prediction():
    '''Funzione che stampa le predizioni del modello CNN di default sui dati di test e
    crea i grafici delle metriche in funzione della soglia decisionale.
    '''
    nltk.download('stopwords')

    print("Verranno utilizzati i file presenti nelle cartelle models e data/processed")

    weights_path = CMEPDA_EXAM_REPOSITORY_NN_MODELS / "model_CNN_v44_BC_02.weights.h5"
    vectorizer_path = CMEPDA_EXAM_REPOSITORY_NN_MODELS / "vectorizer_CNN_v44_BC_02.pkl"

    data_path = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_articles_normalized_optimal_clustering.json"
    label_path = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_dataset_binary_lables_optimal_clustering.npy"
    classes_path = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_keyword_binary_classes_optimal_clustering.npy"

    with open(vectorizer_path, 'rb') as f:
        vectorizer_vocab = pickle.load(f)

    X_train, X_val, X_test, y_train, y_val, y_test, orig_train, orig_val, orig_test = data_splitting_pred(
        data_path,
        label_path
    )

    vectorizer = TextVectorization(max_tokens=25000, output_mode='int', output_sequence_length=200)
    vectorizer.set_vocabulary(vectorizer_vocab)

    model_CNN = Sequential([
        Embedding(input_dim=25001, output_dim = 512),
        SpatialDropout1D(0.2),
        Conv1D(filters=256, kernel_size=5, activation='relu', padding='same'), #128
        BatchNormalization(),
        Conv1D(filters=256, kernel_size=5, activation='relu', padding='same'), #128
        BatchNormalization(),
        GlobalMaxPooling1D(),
        BatchNormalization(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(y_train.shape[1], activation='sigmoid')
    ])

    model_CNN.build(input_shape=(None, 200))

    model_CNN.load_weights(weights_path)
    print("Modello caricato con successo tramite pesi!")

    model_CNN.summary()


    classes = np.load(classes_path, allow_pickle=True)
    mlb = MultiLabelBinarizer()
    mlb.classes_ = classes


    X_test_vect = vectorizer(np.array(X_test)).numpy().astype('int32')
    prediction = model_CNN.predict(X_test_vect)

    predicted_keywords = []
    soglie = [0.3, 0.35, 0.40, 0.45, 0.50, 0.55, 0.6]

    for soglia in soglie:
        predicted_keywords.append(mlb.inverse_transform((prediction>soglia).astype(int)))

    for i in range (0, 30):
        print(f"Articolo {i} del dataset di test")
        print(f"Keywords predette con soglia 0.5: {predicted_keywords[4][i]}")
        print(f"Keywords predette con soglia 0.3: {predicted_keywords[0][i]}")
        print(f"Keywords originali : {orig_test[i]}")
        print(' ')
    
    len_original = [len(o) for o in orig_test]
    print(f"Media delle keywords degli articoli originali : {np.mean(len_original):.2f}")

    contatore = 0
    for id_soglia in range(len(soglie)):
        len_predictions = [len(p) for p in predicted_keywords[id_soglia]]
        print(f'Media keywords predette soglia {soglie[id_soglia]:.2f}: {np.mean(len_predictions):.2f}')
        contatore += 1

    precisions = []
    recalls = []
    f1_scores = []

    for soglia in soglie:
        prediction_soglia = (prediction > soglia).astype(int)
        precision = precision_score(y_test, prediction_soglia, average='micro')
        precisions.append(precision)
        recall = recall_score(y_test, prediction_soglia, average='micro')
        recalls.append(recall)
        f1 = f1_score(y_test, prediction_soglia, average='micro')
        f1_scores.append(f1)
        print(f'Valori delle metriche con threshold {soglia:.2f}: precision = {precision:.4f}, recall = {recall:.4f}, F1-Score = {f1:.4f}')


    plt.figure(figsize=(15, 5))

    # 1. Plot della Precision
    plt.subplot(1, 3, 1)
    plt.plot(soglie, precisions, label='Precision', marker='o')
    plt.xlabel('Soglia')
    plt.ylabel('Precision')
    plt.title('Soglia vs Precision')

    # 2. Plot della Recall
    plt.subplot(1, 3, 2)
    plt.plot(soglie, recalls, label='Recall', marker='o')
    plt.xlabel('Soglia')
    plt.ylabel('Recall')
    plt.title('Soglia vs Recall')

    # 2. Plot del'F1
    plt.subplot(1, 3, 3)
    plt.plot(soglie, f1_scores, label='F1-Score', marker='o')
    plt.xlabel('Soglia')
    plt.ylabel('F1-Score')
    plt.title('Soglia vs F1-Score')

    plt.tight_layout() # Evita che i titoli si sovrappongano
    plt.show()

#---------------------------------------------Predizioni del modello su nuovi testi------------------
def CNN_model_new_prediction(new_text):
    '''Funzione che stampa le predizioni del modello CNN di default sui testi
    di nuovi articoli.
    Args:
         new_text (list): lista dei testi dei nuovi articoli
    '''
    nltk.download('stopwords')

    weights_path = CMEPDA_EXAM_REPOSITORY_NN_MODELS / "model_CNN_v44_BC_02.weights.h5"
    vectorizer_path = CMEPDA_EXAM_REPOSITORY_NN_MODELS / "vectorizer_CNN_v44_BC_02.pkl"
    classes_path = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_keyword_binary_classes_optimal_clustering.npy"

    with open(vectorizer_path, 'rb') as f:
        vectorizer_vocab = pickle.load(f)

    vectorizer = TextVectorization(max_tokens=25000, output_mode='int', output_sequence_length=200)
    vectorizer.set_vocabulary(vectorizer_vocab)

    model_CNN = Sequential([
        Embedding(input_dim=25001, output_dim = 512),
        SpatialDropout1D(0.2),
        Conv1D(filters=256, kernel_size=5, activation='relu', padding='same'), #128
        BatchNormalization(),
        Conv1D(filters=256, kernel_size=5, activation='relu', padding='same'), #128
        BatchNormalization(),
        GlobalMaxPooling1D(),
        BatchNormalization(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(567, activation='sigmoid')
    ])

    model_CNN.build(input_shape=(None, 200))

    model_CNN.load_weights(weights_path)
    print("Modello caricato con successo tramite pesi!")

    model_CNN.summary()

    classes = np.load(classes_path, allow_pickle=True)
    mlb = MultiLabelBinarizer()
    mlb.classes_ = classes

    cleaned_texts = [clean_text(text) for text in new_text]
    new_texts_vect = vectorizer(np.array(cleaned_texts)).numpy().astype('int32')
    prediction = model_CNN.predict(new_texts_vect)
    predicted_keywords = mlb.inverse_transform((prediction>0.5).astype(int))
    for predicted in predicted_keywords:
        print(predicted)

def LSTM_model_prediction():
    '''Funzione che stampa le predizioni del modello LSTM di default sui dati di test e
    crea i grafici delle metriche in funzione della soglia decisionale.
    '''
    nltk.download('stopwords')

    print("Verranno utilizzati i file presenti nelle cartelle models e data/processed")

    weights_path = CMEPDA_EXAM_REPOSITORY_NN_MODELS / "model_LSTM_v81_BC_02.weights.h5"
    vectorizer_path = CMEPDA_EXAM_REPOSITORY_NN_MODELS / "vectorizer_LSTM_v81_BC_02.pkl"

    data_path = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_articles_normalized_optimal_clustering.json"
    label_path = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_dataset_binary_lables_optimal_clustering.npy"
    classes_path= CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_keyword_binary_classes_optimal_clustering.npy"

    with open(vectorizer_path, 'rb') as f:
        vectorizer_vocab = pickle.load(f)

    X_train, X_val, X_test, y_train, y_val, y_test, orig_train, orig_val, orig_test = data_splitting_pred(
        data_path,
        label_path
    )

    vectorizer = TextVectorization(max_tokens=25000, output_mode='int', output_sequence_length=200)
    vectorizer.set_vocabulary(vectorizer_vocab)

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
        #Dropout(0.3),
        Dense(y_train.shape[1], activation='sigmoid')
    ])

    model_LSTM.build(input_shape=(None, 200))

    model_LSTM.load_weights(weights_path)
    print("Modello caricato con successo tramite pesi!")

    model_LSTM.summary()

    classes = np.load(classes_path, allow_pickle=True)
    mlb = MultiLabelBinarizer()
    mlb.classes_ = classes


    X_test_vect = vectorizer(np.array(X_test)).numpy().astype('int32')
    prediction = model_LSTM.predict(X_test_vect)

    predicted_keywords = []
    soglie = [0.3, 0.35, 0.40, 0.45, 0.50, 0.55, 0.6]

    for soglia in soglie:
        predicted_keywords.append(mlb.inverse_transform((prediction>soglia).astype(int)))

    for i in range (0, 30):
        print(f"Articolo {i} del dataset di test")
        print(f"Keywords predette con soglia 0.50: {predicted_keywords[4][i]}")
        print(f"Keywords predette con soglia 0.35: {predicted_keywords[1][i]}")
        print(f"Keywords originali : {orig_test[i]}")
        print(' ')
    
    len_original = [len(o) for o in orig_test]
    print(f"Media delle keywords degli articoli originali : {np.mean(len_original):.2f}")

    contatore = 0
    for id_soglia in range(len(soglie)):
        len_predictions = [len(p) for p in predicted_keywords[id_soglia]]
        print(f'Media keywords predette soglia {soglie[id_soglia]:.2f}: {np.mean(len_predictions):.2f}')
        contatore += 1

    precisions = []
    recalls = []
    f1_scores = []

    for soglia in soglie:
        prediction_soglia = (prediction > soglia).astype(int)
        precision = precision_score(y_test, prediction_soglia, average='micro')
        precisions.append(precision)
        recall = recall_score(y_test, prediction_soglia, average='micro')
        recalls.append(recall)
        f1 = f1_score(y_test, prediction_soglia, average='micro')
        f1_scores.append(f1)
        print(f'Valori delle metriche con threshold {soglia:.2f}: precision = {precision:.4f}, recall = {recall:.4f}, F1-Score = {f1:.4f}')


    plt.figure(figsize=(15, 5))

    # 1. Plot della Precision
    plt.subplot(1, 3, 1)
    plt.plot(soglie, precisions, label='Precision', marker='o')
    plt.xlabel('Soglia')
    plt.ylabel('Precision')
    plt.title('Soglia vs Precision')

    # 2. Plot della Recall
    plt.subplot(1, 3, 2)
    plt.plot(soglie, recalls, label='Recall', marker='o')
    plt.xlabel('Soglia')
    plt.ylabel('Recall')
    plt.title('Soglia vs Recall')

    # 2. Plot del'F1
    plt.subplot(1, 3, 3)
    plt.plot(soglie, f1_scores, label='F1-Score', marker='o')
    plt.xlabel('Soglia')
    plt.ylabel('F1-Score')
    plt.title('Soglia vs F1-Score')

    plt.tight_layout() # Evita che i titoli si sovrappongano
    plt.show()

#---------------------------------------------Predizioni del modello su nuovi testi------------------
def LSTM_model_new_prediction(new_text):
    '''Funzione che stampa le predizioni del modello LSTM di default sui testi
    di nuovi articoli.
    Args:
         new_text (list): lista dei testi dei nuovi articoli
    '''
    nltk.download('stopwords')

    weights_path = CMEPDA_EXAM_REPOSITORY_NN_MODELS / "model_LSTM_v81_BC_02.weights.h5"
    vectorizer_path = CMEPDA_EXAM_REPOSITORY_NN_MODELS / "vectorizer_LSTM_v81_BC_02.pkl"
    classes_path = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_keyword_binary_classes_optimal_clustering.npy"

    with open(vectorizer_path, 'rb') as f:
        vectorizer_vocab = pickle.load(f)

    vectorizer = TextVectorization(max_tokens=25000, output_mode='int', output_sequence_length=200)
    vectorizer.set_vocabulary(vectorizer_vocab)

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
        #Dropout(0.3),
        Dense(567, activation='sigmoid')
    ])

    model_LSTM.build(input_shape=(None, 200))

    model_LSTM.load_weights(weights_path)
    print("Modello caricato con successo tramite pesi!")

    model_LSTM.summary()

    classes = np.load(classes_path, allow_pickle=True)
    mlb = MultiLabelBinarizer()
    mlb.classes_ = classes

    cleaned_texts = [clean_text(text) for text in new_text]
    new_texts_vect = vectorizer(np.array(cleaned_texts)).numpy().astype('int32')
    prediction = model_LSTM.predict(new_texts_vect)
    predicted_keywords = mlb.inverse_transform((prediction>0.5).astype(int))
    for predicted in predicted_keywords:
        print(predicted)

def help():
    print(" ")
    print("Per ottenere alcuni esempi di predizioni di un modello sui dati di test, inserire dopo il nome dello script uno dei modelli disponibili: Dense, CNN o LSTM.")
    print("esempio: python model_predictions.py Dense")
    print("Se si vuole ottenere delle probabili keywords su un nuovo testo inserire dopo il nome del modello il nuovo testo tra virgolette")
    print("esempio: python model_predictions.py Dense testo")

def main():
    modelli = { "Dense": [Dense_model_prediction, Dense_model_new_prediction],
                "CNN": [CNN_model_prediction, CNN_model_new_prediction],
                "LSTM": [LSTM_model_prediction, LSTM_model_new_prediction],
                "help": help
              }
    if len(sys.argv) < 2:
        modelli["help"]()
        return
    modello = sys.argv[1]
    if len(sys.argv) == 2:
        if modello in modelli:
            if modello != "help":
                print(f'Modello selezionato : {modello}')
                modelli[modello][0]()
            else:
                modelli[modello]()
        else:
            print(f"Modello {modello} non disponibile!")
            modelli["help"]()
    
    if len(sys.argv)>=3:
        new_texts = [arg for arg in sys.argv[2:] ]
        if modello in modelli:
            if modello != "help":
                print(f'Modello selezionato : {modello}')
                modelli[modello][1](new_texts)
            else:
                modelli[modello]()
        else:
            print(f"Modello {modello} non disponibile!")
            modelli["help"]()

if __name__ == "__main__":
    main()
