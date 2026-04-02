from tensorflow.keras.layers import TextVectorization, Embedding, Dense, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D, LSTM, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import re 
import json
import nltk
import gc
import tensorflow
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords

def clean_text(text):
    # 1. Rimuove espressioni LaTeX tra $ (es. $\beta$, $\sqrt{s}$)
    #text = re.sub(r'\$.*?\$', '', text)
    text = text.replace('$', '')
    # 2. Rimuove comandi LaTeX che iniziano con \ (es. \begin, \alpha)
    #text = re.sub(r'\\\w+', '', text)
    text = text.replace('\\', '')
    text = re.sub(r'\{.*?\}', ' ', text)
    # 3. Rimuove tutto ciò che non è una lettera (numeri e punteggiatura)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # 4. Lowercase e rimozione stop-words
    stop_words = set(stopwords.words('english'))
    words = text.lower().split()
    clean_words = [w for w in words if w not in stop_words and len(w) > 1]
    
    return " ".join(clean_words)

def data_splitting(texts_file, label_file, test_size=0.15, val_size=0.176, random_state=42):
    with open(texts_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [clean_text(article["text"]) for article in data]
    del data
    gc.collect()
    #lengths =[len(t.split()) for t in texts]
    #freq = Counter(lengths)
    #print(f"Lunghezza massima = {max(lengths)}")

    #x = list(freq.keys())#lunghezza per sequenza
    #y = list(freq.values())#frequenza

    #plt.bar(x, y)
    #plt.xlabel("Lunghezza frase")
    #plt.ylabel("Frequenza")
    #plt.title("Distribuzione lunghezze sequenze")
    #plt.show()

    #with open(label_file, "r", encoding="utf-8") as file:
        #labels = np.array(json.load(file))
    labels = np.load(label_file).astype(np.int8)

    X_temp, X_test, y_temp, y_test = train_test_split(texts, labels, test_size=0.10, random_state=42)
    del texts, labels
    gc.collect()
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42)
    del X_temp, y_temp
    gc.collect()
    return X_train, X_val, X_test, y_train, y_val, y_test 

def plot_training_history(history, metric='accuracy'):
    """
    history: oggetto restituito da model.fit()
    metric: 'accuracy' o un'altra metrica monitorata dal modello
    """
    # plot della loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train loss', marker=None)
    plt.plot(history.history['val_loss'], label='val loss', marker=None)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    
    # plot della metrica (accuracy o altra)
    plt.subplot(1, 2, 2)
    plt.plot(history.history[metric], label=f'train {metric}', marker=None)
    plt.plot(history.history[f'val_{metric}'], label=f'val {metric}', marker=None)
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.title(f'Training vs Validation {metric.capitalize()}')
    plt.legend()
    
    plt.show()   

def model_1():
    #with open("data/processed/articles_normalized.json", "r", encoding="utf-8") as f:
        #data = json.load(f)

    #texts = [article["text"] for article in data]
    #texts = np.array(texts)

    #with open("data/processed/dataset_binary_lables.json", "r", encoding="utf-8") as file:
        #labels = np.array(json.load(file))

    #X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.15, random_state=42)

    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.176, random_state=42)

    X_train, X_val, X_test, y_train, y_val, y_test = data_splitting(
        "data/processed/articles_normalized.json",
        "data/processed/dataset_binary_lables.json"
    )

    print("dataset caricato e splittato")

    vectorizer = TextVectorization(
        max_tokens=5000,
        output_mode='int',
        output_sequence_length=200
    )

    vectorizer.adapt(X_train)

    print("Train:", len(X_train))
    print("Validation:", len(X_val))
    print("Test:", len(X_test))
    print("Label shape:", y_train.shape)
    #inputs = vectorizer(texts)
    X_train_vect = vectorizer(X_train)
    X_val_vect = vectorizer(X_val)
    X_test_vect = vectorizer(X_test)

    model_1 = Sequential([
        Embedding(input_dim=5000, output_dim = 64),
        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        Dense(y_train.shape[1], activation='sigmoid')
    ])

    model_1.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    history = model_1.fit(
        X_train_vect,
        y_train,
        epochs=100, #10
        batch_size=32, #32
        validation_data=(X_val_vect, y_val)
    )

    loss_train = history.history["loss"]
    loss_val = history.history["val_loss"]

    acc_train = history.history["accuracy"]
    acc_val = history.history["val_accuracy"]

    epochs = range(1, len(loss_train) + 1)

    # plot della loss
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(epochs, loss_train, 'b-', label='Training Loss')
    plt.plot(epochs, loss_val, 'r-', label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # plot dell'accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, acc_train, 'b-', label='Training Accuracy')
    plt.plot(epochs, acc_val, 'r-', label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("--------------------------------------------TEST-------------------------------------------------")
    model_1.evaluate(X_test_vect, y_test)

def model_CNN():
    X_train, X_val, X_test, y_train, y_val, y_test = data_splitting(
        "data/processed/articles_normalized_clustering.json",
        "data/processed/dataset_binary_lables_clustering.npy"
    )
    print("dataset caricato e splittato")

    vectorizer = TextVectorization(
        max_tokens=10000,
        output_mode='int',
        output_sequence_length=200
    )

    vectorizer.adapt(X_train)

    print("Train:", len(X_train))
    print("Validation:", len(X_val))
    print("Test:", len(X_test))
    print("Label shape:", y_train.shape)
    #inputs = vectorizer(texts)
    #X_train_vect = vectorizer(X_train)
    #X_val_vect = vectorizer(X_val)
    #X_test_vect = vectorizer(X_test)
    X_train_vect = vectorizer(np.array(X_train)).numpy().astype('int32')
    X_val_vect = vectorizer(np.array(X_val)).numpy().astype('int32')
    X_test_vect = vectorizer(np.array(X_test)).numpy().astype('int32')
    del X_train
    del X_val
    del X_test 
    gc.collect()

    model_CNN = Sequential([
        Embedding(input_dim=10001, output_dim = 128),
        Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'), #128
        BatchNormalization(),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'), #128
        BatchNormalization(),
        GlobalMaxPooling1D(),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(y_train.shape[1], activation='sigmoid')
    ])

    model_CNN.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[
            'accuracy',
            tensorflow.keras.metrics.Precision(name='precision'),
            tensorflow.keras.metrics.Recall(name='recall')
        ]
    )

    history = model_CNN.fit(
        X_train_vect,
        y_train,
        epochs=15,
        batch_size=32, #32
        validation_data=(X_val_vect, y_val)
    )

    plot_training_history(history=history, metric='accuracy')

    print("--------------------------------------------TEST-------------------------------------------------")
    model_CNN.evaluate(X_test_vect, y_test)

def model_lstm():
    X_train, X_val, X_test, y_train, y_val, y_test = data_splitting(
        "data/processed/articles_normalized_clustering.json",
        "data/processed/dataset_binary_lables_clustering.npy"
    )
    print("dataset caricato e splittato")

    vectorizer = TextVectorization(
        max_tokens=10000,
        output_mode='int',
        output_sequence_length=200
    )

    vectorizer.adapt(X_train)

    print("Train:", len(X_train))
    print("Validation:", len(X_val))
    print("Test:", len(X_test))
    print("Label shape:", y_train.shape)
    #inputs = vectorizer(texts)
    X_train_vect = vectorizer(np.array(X_train)).numpy().astype('int32')
    X_val_vect = vectorizer(np.array(X_val)).numpy().astype('int32')
    X_test_vect = vectorizer(np.array(X_test)).numpy().astype('int32')

    del X_train
    del X_val
    del X_test 
    gc.collect()

    model_lstm = Sequential([
        Embedding(input_dim=10001, output_dim = 256, mask_zero=True), #64 #128  #5000 
        Bidirectional(LSTM(128, return_sequences=True)),#64
        GlobalMaxPooling1D(),
        BatchNormalization(),
        Dense(512, activation='relu'),#128
        Dropout(0.4),
        BatchNormalization(),
        #Dense(64, activation='relu'),
        #Dropout(0.2),
        Dense(y_train.shape[1], activation='sigmoid')
    ])

    model_lstm.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[
            #'accuracy',
            tensorflow.keras.metrics.Precision(name='precision'),
            tensorflow.keras.metrics.Recall(name='recall')
        ] #'accuracy'
    )

    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=5,          # Si ferma se la val_loss non migliora per 5 epoche
        restore_best_weights=True
    )

    history = model_lstm.fit(
        X_train_vect,
        y_train,
        epochs=5,
        batch_size=32, #32
        validation_data=(X_val_vect, y_val)#,
        #callbacks=[early_stop]
    )
    plot_training_history(history=history, metric='accuracy')

    print("--------------------------------------------TEST-------------------------------------------------")
    model_lstm.evaluate(X_test_vect, y_test)

if __name__ == "__main__":
    #model_1()
    model_CNN()
    #model_lstm()
