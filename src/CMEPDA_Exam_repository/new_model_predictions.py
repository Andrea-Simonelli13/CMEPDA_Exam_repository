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
    CMEPDA_EXAM_REPOSITORY_NEW_MODELS,
    CMEPDA_EXAM_REPOSITORY_DATA_NEW,
    CMEPDA_EXAM_REPOSITORY_DATA
                                    )

#-----------------------------Funzione che pulisce i testi-----------------------------------------
def clean_text(text):
    # 1. Rimuove il simbolo $ spesso utilizzato in LaTex
    #text = re.sub(r'\$.*?\$', '', text)
    text = text.replace('$', '')
    # 2. Rimuove il simbolo \ che spesso è utilizzato in latex per i simboli come alpha, beta, tau etc.
    #text = re.sub(r'\\\w+', '', text)
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
#-----------------Funzione che divide il dataset in training, vlidation e test-------------------------------
def data_splitting_pred(texts_file, label_file):
    with open(texts_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [clean_text(article["text"]) for article in data]
    original_keywords = [article["keywords"] for article in data]
    del data
    gc.collect()
    
    labels = np.load(label_file).astype(np.int8)
    indices = [i for i in range(len(texts))]

    X_temp, X_test, y_temp, y_test, orig_temp, orig_test = train_test_split(texts, labels, original_keywords, test_size=0.10, random_state=42)
    del texts, labels
    gc.collect()
    X_train, X_val, y_train, y_val, orig_train, orig_val = train_test_split(X_temp, y_temp, orig_temp, test_size=0.1111, random_state=42)
    del X_temp, y_temp
    gc.collect()
    return X_train, X_val, X_test, y_train, y_val, y_test, orig_train, orig_val, orig_test

#-----------------------------Predizioni del modello Dense sui dati di test---------------------------
def trained_Dense_model_prediction():
    nltk.download('stopwords')
  
    print(f"Verranno utilizzati i file presenti nelle cartelle new_models e data_new/processed")
    weights_path = CMEPDA_EXAM_REPOSITORY_NEW_MODELS / "new_model_Dense.weights.h5"
    vectorizer_path = CMEPDA_EXAM_REPOSITORY_NEW_MODELS / "new_vocabulary_Dense.pkl"
    data_path = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_article_clustering.json"
    label_path = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_dataset_binary_lables.npy"
    classes_path = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_keyword_binary_classes.npy"

    #data_path = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_articles_normalized_optimal_clustering.json"
    #label_path = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_dataset_binary_lables_optimal_clustering.npy"
    #classes_path= CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_keyword_binary_classes_optimal_clustering.npy"

    with open(vectorizer_path, 'rb') as f:
        vectorizer_vocab = pickle.load(f)

    X_train, X_val, X_test, y_train, y_val, y_test, orig_train, orig_val, orig_test = data_splitting_pred(
        data_path,
        label_path
    )

    vectorizer = TextVectorization(max_tokens=25000, output_mode='int', output_sequence_length=200)
    vectorizer.set_vocabulary(vectorizer_vocab)

    #model_Dense = tensorflow.keras.models.load_model(Dense_model_path, compile=False)
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

    #with open (data_path, "r", encoding="utf-8") as f:
        #data = json.load(f)

    #labels = np.load(label_path).astype(np.int8)
    classes = np.load(classes_path, allow_pickle=True)
    mlb = MultiLabelBinarizer()
    mlb.classes_ = classes

    #original_keywords = [article["keywords"] for article in data]
    #texts = [clean_text(article["text"]) for article in data]
    ##

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
    #plt.legend()

    # 2. Plot della Recall
    plt.subplot(1, 3, 2)
    plt.plot(soglie, recalls, label='Recall', marker='o')
    plt.xlabel('Soglia')
    plt.ylabel('Recall')
    plt.title('Soglia vs Recall')
    #plt.legend()

    # 2. Plot del'F1
    plt.subplot(1, 3, 3)
    plt.plot(soglie, f1_scores, label='F1-Score', marker='o')
    plt.xlabel('Soglia')
    plt.ylabel('F1-Score')
    plt.title('Soglia vs F1-Score')
    #plt.legend()

    plt.tight_layout() # Evita che i titoli si sovrappongano
    plt.show()

#---------------------------------------------Predizioni del modello su nuovi testi------------------
def trained_Dense_model_new_prediction(new_text):
    #if isinstance(new_text, str):
        #print(new_text)
    #else:
        #print("Tipo non valido: è necessario passare una stringa")
    nltk.download('stopwords')
    #weights_path = "models/model_Dense_v41_25.weights.h5"
    #vectorizer_path="models/vectorizer_Dense_v41_weights_25.pkl"
    weights_path = CMEPDA_EXAM_REPOSITORY_NEW_MODELS / "new_model_Dense.weights.h5"
    vectorizer_path = CMEPDA_EXAM_REPOSITORY_NEW_MODELS / "new_vocabulary_Dense.pkl"
    classes_path = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_keyword_binary_classes.npy"
    #classes_path= CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_keyword_binary_classes_optimal_clustering.npy"

    with open(vectorizer_path, 'rb') as f:
        vectorizer_vocab = pickle.load(f)

    vectorizer = TextVectorization(max_tokens=25000, output_mode='int', output_sequence_length=200)
    vectorizer.set_vocabulary(vectorizer_vocab)

    classes = np.load(classes_path, allow_pickle=True)

    model_Dense = Sequential([
        Embedding(input_dim=25001, output_dim = 512),
        GlobalMaxPooling1D(),
        BatchNormalization(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dense(len(classes), activation='sigmoid')
    ])

    model_Dense.build(input_shape=(None, 200))

    model_Dense.load_weights(weights_path)
    print("Modello caricato con successo tramite pesi!")

    model_Dense.summary()

    #classes = np.load(classes_path, allow_pickle=True)
    mlb = MultiLabelBinarizer()
    mlb.classes_ = classes

    cleaned_texts = [clean_text(text) for text in new_text]
    new_texts_vect = vectorizer(np.array(cleaned_texts)).numpy().astype('int32')
    prediction = model_Dense.predict(new_texts_vect)
    predicted_keywords = mlb.inverse_transform((prediction>0.5).astype(int))
    for predicted in predicted_keywords:
        print(predicted)

def trained_CNN_model_prediction():
    nltk.download('stopwords')
    
    print(f"Verranno utilizzati i file presenti nelle cartelle new_models e data_new/processed")
    weights_path = CMEPDA_EXAM_REPOSITORY_NEW_MODELS / "new_model_CNN.weights.h5"
    vectorizer_path = CMEPDA_EXAM_REPOSITORY_NEW_MODELS / "new_vocabulary_CNN.pkl"
    #data_path="data/processed/final_articles_normalized_optimal_clustering.json"
    #label_path="data/processed/final_dataset_binary_lables_optimal_clustering.npy"
    #classes_path="data/processed/final_keyword_binary_classes_optimal_clustering.npy"

    data_path = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_article_clustering.json"
    label_path = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_dataset_binary_lables.npy"
    classes_path = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_keyword_binary_classes.npy"

    #data_path = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_articles_normalized_optimal_clustering.json"
    #label_path = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_dataset_binary_lables_optimal_clustering.npy"
    #classes_path= CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_keyword_binary_classes_optimal_clustering.npy"

    with open(vectorizer_path, 'rb') as f:
        vectorizer_vocab = pickle.load(f)

    X_train, X_val, X_test, y_train, y_val, y_test, orig_train, orig_val, orig_test = data_splitting_pred(
        data_path,
        label_path
    )

    vectorizer = TextVectorization(max_tokens=25000, output_mode='int', output_sequence_length=200)
    vectorizer.set_vocabulary(vectorizer_vocab)


    #model_Dense = tensorflow.keras.models.load_model(Dense_model_path, compile=False)
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

    #with open (data_path, "r", encoding="utf-8") as f:
        #data = json.load(f)

    #labels = np.load(label_path).astype(np.int8)
    classes = np.load(classes_path, allow_pickle=True)
    mlb = MultiLabelBinarizer()
    mlb.classes_ = classes

    #original_keywords = [article["keywords"] for article in data]
    #texts = [clean_text(article["text"]) for article in data]
    ##

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
    #plt.legend()

    # 2. Plot della Recall
    plt.subplot(1, 3, 2)
    plt.plot(soglie, recalls, label='Recall', marker='o')
    plt.xlabel('Soglia')
    plt.ylabel('Recall')
    plt.title('Soglia vs Recall')
    #plt.legend()

    # 2. Plot del'F1
    plt.subplot(1, 3, 3)
    plt.plot(soglie, f1_scores, label='F1-Score', marker='o')
    plt.xlabel('Soglia')
    plt.ylabel('F1-Score')
    plt.title('Soglia vs F1-Score')
    #plt.legend()

    plt.tight_layout() # Evita che i titoli si sovrappongano
    plt.show()

#---------------------------------------------Predizioni del modello su nuovi testi------------------
def trained_CNN_model_new_prediction(new_text):
    nltk.download('stopwords')
    #weights_path = "models/model_CNN_v61.weights.h5"
    #vectorizer_path="models/vectorizer_CNN_v61.pkl"
    weights_path = CMEPDA_EXAM_REPOSITORY_NEW_MODELS / "new_model_CNN.weights.h5"
    vectorizer_path = CMEPDA_EXAM_REPOSITORY_NEW_MODELS / "new_vocabulary_CNN.pkl"
    classes_path = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_keyword_binary_classes.npy"
    #classes_path= CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_keyword_binary_classes_optimal_clustering.npy"

    with open(vectorizer_path, 'rb') as f:
        vectorizer_vocab = pickle.load(f)

    vectorizer = TextVectorization(max_tokens=25000, output_mode='int', output_sequence_length=200)
    vectorizer.set_vocabulary(vectorizer_vocab)

    classes = np.load(classes_path, allow_pickle=True)

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
        Dense(len(classes), activation='sigmoid')
    ])

    model_CNN.build(input_shape=(None, 200))

    model_CNN.load_weights(weights_path)
    print("Modello caricato con successo tramite pesi!")

    model_CNN.summary()

    #classes = np.load(classes_path, allow_pickle=True)
    mlb = MultiLabelBinarizer()
    mlb.classes_ = classes

    cleaned_texts = [clean_text(text) for text in new_text]
    new_texts_vect = vectorizer(np.array(cleaned_texts)).numpy().astype('int32')
    prediction = model_CNN.predict(new_texts_vect)
    predicted_keywords = mlb.inverse_transform((prediction>0.5).astype(int))
    for predicted in predicted_keywords:
        print(predicted)

def trained_LSTM_model_prediction():
    nltk.download('stopwords')
    print(f"Verranno utilizzati i file presenti nelle cartelle new_models e data_new/processed")
    weights_path = CMEPDA_EXAM_REPOSITORY_NEW_MODELS / "new_model_LSTM.weights.h5"
    vectorizer_path = CMEPDA_EXAM_REPOSITORY_NEW_MODELS / "new_vocabulary_LSTM.pkl"
    
    #data_path="data/processed/final_articles_normalized_optimal_clustering.json"
    #label_path="data/processed/final_dataset_binary_lables_optimal_clustering.npy"
    #classes_path="data/processed/final_keyword_binary_classes_optimal_clustering.npy"

    data_path = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_article_clustering.json"
    label_path = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_dataset_binary_lables.npy"
    classes_path = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_keyword_binary_classes.npy"

    #data_path = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_articles_normalized_optimal_clustering.json"
    #label_path = CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_dataset_binary_lables_optimal_clustering.npy"
    #classes_path= CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_keyword_binary_classes_optimal_clustering.npy"

    with open(vectorizer_path, 'rb') as f:
        vectorizer_vocab = pickle.load(f)

    X_train, X_val, X_test, y_train, y_val, y_test, orig_train, orig_val, orig_test = data_splitting_pred(
        data_path,
        label_path
    )

    vectorizer = TextVectorization(max_tokens=25000, output_mode='int', output_sequence_length=200)
    vectorizer.set_vocabulary(vectorizer_vocab)

    #model_Dense = tensorflow.keras.models.load_model(Dense_model_path, compile=False)

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

    #with open (data_path, "r", encoding="utf-8") as f:
        #data = json.load(f)

    #labels = np.load(label_path).astype(np.int8)
    classes = np.load(classes_path, allow_pickle=True)
    mlb = MultiLabelBinarizer()
    mlb.classes_ = classes

    #original_keywords = [article["keywords"] for article in data]
    #texts = [clean_text(article["text"]) for article in data]
    ##

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
    #plt.legend()

    # 2. Plot della Recall
    plt.subplot(1, 3, 2)
    plt.plot(soglie, recalls, label='Recall', marker='o')
    plt.xlabel('Soglia')
    plt.ylabel('Recall')
    plt.title('Soglia vs Recall')
    #plt.legend()

    # 2. Plot del'F1
    plt.subplot(1, 3, 3)
    plt.plot(soglie, f1_scores, label='F1-Score', marker='o')
    plt.xlabel('Soglia')
    plt.ylabel('F1-Score')
    plt.title('Soglia vs F1-Score')
    #plt.legend()

    plt.tight_layout() # Evita che i titoli si sovrappongano
    plt.show()

#---------------------------------------------Predizioni del modello su nuovi testi------------------
def trained_LSTM_model_new_prediction(new_text):
    #if isinstance(new_text, str):
        #print(new_text)
    #else:
        #print("Tipo non valido: è necessario passare una stringa")
    nltk.download('stopwords')
    #weights_path = "models/model_LSTM_v81.weights.h5"
    #vectorizer_path="models/vectorizer_LSTM_v81.pkl"
    weights_path = CMEPDA_EXAM_REPOSITORY_NEW_MODELS / "new_model_LSTM.weights.h5"
    vectorizer_path = CMEPDA_EXAM_REPOSITORY_NEW_MODELS / "new_vocabulary_LSTM.pkl"
    classes_path = CMEPDA_EXAM_REPOSITORY_DATA_NEW / "processed/new_keyword_binary_classes.npy"
    #classes_path= CMEPDA_EXAM_REPOSITORY_DATA / "processed/final_keyword_binary_classes_optimal_clustering.npy"
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer_vocab = pickle.load(f)

    vectorizer = TextVectorization(max_tokens=25000, output_mode='int', output_sequence_length=200)
    vectorizer.set_vocabulary(vectorizer_vocab)

    classes = np.load(classes_path, allow_pickle=True)

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
        Dense(len(classes), activation='sigmoid')
    ])

    model_LSTM.build(input_shape=(None, 200))

    model_LSTM.load_weights(weights_path)
    print("Modello caricato con successo tramite pesi!")

    model_LSTM.summary()

    #classes = np.load(classes_path, allow_pickle=True)
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
    print("esempio: python new_model_predictions.py Dense")
    print("Se si vuole ottenere delle probabili keywords su un nuovo testo inserire dopo il nome del modello il nuovo testo tra virgolette")
    print("esempio: python new_model_predictions.py Dense testo")

def main():
    modelli = { "Dense": [trained_Dense_model_prediction, trained_Dense_model_new_prediction],
                "CNN": [trained_CNN_model_prediction, trained_CNN_model_new_prediction],
                "LSTM": [trained_LSTM_model_prediction, trained_LSTM_model_new_prediction],
                "help": help
              }
    if len(sys.argv) < 2:
        modelli["help"]()
        return
    modello = sys.argv[1]
    #nuovo_testo = sys.argv[2]
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