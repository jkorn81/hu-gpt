import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import save_model, load_model
import pandas as pd
import numpy as np
import os
import warnings
import random
from os import walk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFGPT2Model
from transformers import TFAutoModelForCausalLM

warnings.filterwarnings("ignore")

os.chdir('C:/Users/Jonathan Korn/Desktop/deep.txt.gen.eoe.[v.1]')

# Load the data
mypath = './processed/'
filenames = next(walk(mypath), (None, None, []))[2]
rand_num = random.randint(0,len(filenames))
data = pd.read_csv('processed/'+str(filenames[-1]), sep=',')
# Initialize an empty string to store the combined text
text = ""

# Iterate over each file
for filename in filenames:
    # Read the file using pandas
    data = pd.read_csv(os.path.join(mypath, filename), sep=',')
    
    # Concatenate the 'text' column of the current file to the 'text' variable
    text += data['text'].str.cat(sep=' ')

# Open the file in read mode
with open('words/words.txt', 'r') as file:
    # Read the contents of the file
    words = file.read()

# Define the chunk size
chunk_size = 100
# Split the words into chunks
chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

# Tokenize each chunk
tokenizer = Tokenizer()
for chunk in chunks:
    tokenizer.fit_on_texts([chunk])

# Prepare the data
sequences = tokenizer.texts_to_sequences([text])[0]
vocab_size = (len(tokenizer.word_index) + 1)/1000
print(f"Vocabulary size: {vocab_size}")
X = []
y = []
for i in range(1, len(sequences)):
    X.append(sequences[i - 1])
    y.append(sequences[i])
X = np.array(X)
idx = np.random.choice(len(X), size=len(X), replace=False)
X = X[idx]
fixed_size = 1000  # Replace with your desired fixed size
X = np.array(X[:fixed_size])
y = np.array(y)
y = y[idx]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)


class CustomRLOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, name="CustomRLOptimizer", **kwargs):
        super(CustomRLOptimizer, self).__init__(name, **kwargs)
        self.learning_rate = learning_rate

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        updates = []
        for param, grad in zip(params, grads):
            update = tf.multiply(self.learning_rate, grad)  # Custom update rule
            new_param = param - update
            updates.append((param, new_param))
        return updates

    def get_config(self):
        config = super(CustomRLOptimizer, self).get_config()
        config.update({"learning_rate": self.learning_rate})
        return config


# Define the GPT-4-like model
class GPT4Model(tf.keras.Model):
    def __init__(self, vocab_size, num_layers=2, num_heads=2, d_model=1, d_ff=1, dropout_rate=0.1):
        super(GPT4Model, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = [
            self.create_transformer_block(num_heads, d_model, d_ff, dropout_rate) for _ in range(num_layers)
        ]
        self.dropout = Dropout(dropout_rate)
        self.final_layer_norm = LayerNormalization(epsilon=1e-6)
        self.dense = Dense(vocab_size, activation="softmax")

    def create_transformer_block(self, num_heads, d_model, d_ff, dropout_rate):
        return keras.Sequential([
            keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model),
            Dense(d_ff, activation="relu"),
            Dropout(dropout_rate),
            Dense(d_model),
            Dropout(dropout_rate),
            LayerNormalization(epsilon=1e-6)
        ])

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        x = self.dropout(x, training=training)
        x = self.final_layer_norm(x)
        x = self.dense(x)
        return x


# Load the data
# ... your code to load and preprocess the data ...

# Define the GPT-4-like model
gpt4_model = GPT4Model(vocab_size)

# Define the number of GPT-2 model instances to stack
num_stacked_models = 1  # Experiment with the number of stacked models

# Stack the GPT-2 models
stacked_models = []
for _ in range(num_stacked_models):
    stacked_models.append(gpt4_model)

# Add more layers after the stacked GPT-2 models
model = keras.Sequential([
    Embedding(vocab_size, 50, input_length=X.shape[1]),  # Adjust input_length to the appropriate value
    keras.layers.TimeDistributed(keras.layers.Flatten()),  # Flatten the 4D tensor to 3D
    *stacked_models,
    Dense(128, activation="relu"),  # Add additional dense layers
    Dense(vocab_size, activation="softmax")
])

# Define the custom optimizer
optimizer = CustomRLOptimizer(learning_rate=0.001)

model.compile(loss="categorical_crossentropy", optimizer=optimizer)

# Train the model
model.fit(X, y, batch_size=2, epochs=10)

# Save the model
model.save('states/gpt_model2.h5')
