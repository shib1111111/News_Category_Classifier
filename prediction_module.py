import json
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D,Flatten
from sklearn.model_selection import train_test_split
import pickle
import numpy as np


def train_model():
    data = []
    with open('News_Category_Dataset_v3.json', 'r') as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    df = df[['link', 'headline', 'short_description', 'category']]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['headline'] + ' ' + df['short_description'])
    sequences = tokenizer.texts_to_sequences(df['headline'] + ' ' + df['short_description'])
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1

    max_length = 256
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    X = padded_sequences
    y = pd.get_dummies(df['category']).values
    num_classes = y.shape[1]
    categories = df['category'].unique().tolist()  # Obtain list of categories


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    num_epochs = 5
    batch_size = 32
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=num_epochs)

    # Save tokenizer, max_length, categories, and the model
    model.save('trained_model.h5')
    with open('tokenizer.pkl', 'wb') as f:
      pickle.dump(tokenizer, f)
    np.save('categories.npy', categories)
    np.save('max_length.npy',max_length)

    print("training successful")

#train_model()



def load_model_components():
    tokenizer = None
    max_length = None
    categories = None
    model = None

    try:
        with open('tokenizer.pkl', 'rb') as f:                     # Load tokenizer
            tokenizer = pickle.load(f)
        max_length = np.load('max_length.npy', allow_pickle=True)  # Load max_length
        categories = np.load('categories.npy', allow_pickle=True)  # Load categories
        model = load_model('trained_model.h5')                     # Load model
        print("Model components loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except pickle.UnpicklingError as e:
        print(f"Error: Failed to unpickle the tokenizer - {e}")
    except Exception as e:
        print(f"Error: An exception occurred - {e}")

    return tokenizer, max_length, categories, model



def predict(tokenizer, max_length, model, headline, description, categories):
    if tokenizer is None:
        raise ValueError("Tokenizer is not initialized")
    sequence = tokenizer.texts_to_sequences([headline + ' ' + description])
    sequence = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(sequence)
    predicted_class_index = np.argmax(prediction)
    predicted_class = categories[predicted_class_index]
    return predicted_class
