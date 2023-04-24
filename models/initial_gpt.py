import tensorflow as tf
from tensorflow import keras
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
from os import walk
from tensorflow.keras.preprocessing.sequence import pad_sequences


warnings.filterwarnings("ignore")

os.chdir('C:/Users/Jonathan Korn/Desktop/deep.txt.gen.eoe.[v.1]')

# Load the data
mypath = './processed/'
filenames = next(walk(mypath), (None, None, []))[2]
rand_num = random.randint(0,len(filenames))
data = pd.read_csv('processed/'+str(filenames[-1]), sep=',')
text = data['text'].str.cat(sep=' ')

# Open the file in read mode
with open('words/words.txt', 'r') as file:
    # Read the contents of the file
    words = file.read()
    
# Define the chunk size
chunk_size = 10000
# Split the words into chunks
chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]

# Tokenize each chunk
tokenizer = Tokenizer()
for chunk in chunks:
    tokenizer.fit_on_texts([chunk])

# Prepare the data
sequences = tokenizer.texts_to_sequences([text])[0]
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")
X = []
y = []
for i in range(1, len(sequences)):
    X.append(sequences[i-1])
    y.append(sequences[i])
X = np.array(X)
idx = np.random.choice(len(X), size=len([X]), replace=False)
X = X[idx]
y = np.array(y)
y = y[idx]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# Build the model
model = keras.Sequential([
    Embedding(vocab_size, 50, input_length=1),
    LSTM(128),
    Dense(vocab_size, activation="softmax")
])
model.compile(loss="categorical_crossentropy", optimizer="adam")

# Train the model
model.fit(X, y, batch_size=128, epochs=10)

# Save the model
model.save('states/gpt_model.h5')
