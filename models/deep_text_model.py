from urllib.request import Request, urlopen
import random
import os
import warnings
import tensorflow as tf
import numpy as np
import os
import pickle
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from string import punctuation

warnings.filterwarnings("ignore")

os.chdir('C:/Users/Jonathan Korn/Desktop/deep.txt.gen.eoe.[v.1]')

model = Sequential([
    LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(n_unique_chars, activation="softmax"),
])

# define the model path
model_weights_path = f"states/individuals/{BASENAME}-{sequence_length}.h5"
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# make results folder if does not exist yet
if not os.path.isdir("states/individuals"):
    os.mkdir("states/individuals")
# train the model
model.fit(ds, steps_per_epoch=(len(encoded_text) - sequence_length) // BATCH_SIZE, epochs=EPOCHS)
# save the model
model.save(model_weights_path)
