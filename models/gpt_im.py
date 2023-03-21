import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import os
import warnings
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings("ignore")

os.chdir('C:/Users/Jonathan Korn/Desktop/deep.txt.gen.eoe.[v.1]')

# Load the data
data = pd.read_csv('processed/data.csv', sep=',')
text_str = data['text'].str.cat(sep=' ')

# Split the string into train and validation sets
sentences = text_str.split('. ')  # split by sentences
random.shuffle(sentences)  # shuffle the order of sentences
split_index = int(len(sentences) * 0.8)  # 80% train and 20% validation
train_data = sentences[:split_index]
val_data = sentences[split_index:]

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data)
vocab_size = len(tokenizer.word_index) + 1
train_sequences = tokenizer.texts_to_sequences(train_data)
val_sequences = tokenizer.texts_to_sequences(val_data)

# Pad the sequences to a fixed length
max_length = 100
train_sequences = pad_sequences(train_sequences, maxlen=max_length, padding='pre')
val_sequences = pad_sequences(val_sequences, maxlen=max_length, padding='pre')

# Determine the number of classes
num_classes = len(set(text_str.split()))

# Define the model - 
model = tf.keras.Sequential([
  Embedding(vocab_size, 128),
  LSTM(128),
  Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
X_train = np.reshape(train_sequences[:-1], (train_sequences[:-1].shape[0], train_sequences[:-1].shape[1], 1))
y_train = tf.keras.utils.to_categorical(train_sequences[1:], num_classes=num_classes)
X_val = np.reshape(val_sequences[:-1], (val_sequences[:-1].shape[0], val_sequences[:-1].shape[1], 1))
y_val = tf.keras.utils.to_categorical(val_sequences[1:], num_classes=num_classes)

# Generate text
seed_text = 'Life is a journey'
for i in range(20):
    sequence = tokenizer.texts_to_sequences([seed_text])[0]
    sequence = np.array(sequence)
    prediction = model.predict(sequence)[0]
    prediction = np.argmax(prediction)
    output_word = ''
    for word, index in tokenizer.word_index.items():
        if index == prediction:
            output_word = word
            break
    seed_text += ' ' + output_word

print(seed_text)

# Save the model
save_model(model, 'gpt_model.h5')
