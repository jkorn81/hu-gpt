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
from tensorflow.keras.preprocessing.sequence import pad_sequences


warnings.filterwarnings("ignore")

os.chdir('C:/Users/Jonathan Korn/Desktop/deep.txt.gen.eoe.[v.1]')

# Load the data
data = pd.read_csv('processed/data.csv', sep=',')
text = data['text'].str.cat(sep=' ')

# Open the file in read mode
with open('data/words.txt', 'r') as file:
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
idx = np.random.choice(len(X), size=1000, replace=False)
X = X[idx]
y = np.array(y)
y = y[idx]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# Load the Model 
model = load_model('states/gpt_model.h5')
model.compile(loss="categorical_crossentropy", optimizer="adam")

# Train the model
model.fit(X, y, batch_size=128, epochs=10)

# Save the model
model.save('states/gpt_model.h5')

# Generate some text
import openai
# Set up the model and prompt
model_engine = "text-davinci-003"
# Set up the OpenAI API client
openai.api_key = "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"

from urllib.request import Request, urlopen
import random
import os
import warnings

warnings.filterwarnings("ignore")

os.chdir('C:/Users/Jonathan Korn/Desktop/deep.txt.gen.eoe.[v.1]')

prompt = "Provide me a random word."
# Generate a response
completion = openai.Completion.create(
  engine=model_engine,
  prompt=prompt,
  max_tokens=2000,
  n=1,
  stop=None,
  temperature=0.5,
)
response = completion.choices[0].text
rand_words = response[:0].split("\n")


words = range(0, len(rand_words))
for i in words:
  random.shuffle(rand_words)
  prompt = "Provide me a 400 word paragraph or statement or quote or phrase about "+str(rand_words[i])+" with at least 5 sentences or more."
  # Generate a response
  completion = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=2000,
    n=1,
    stop=None,
    temperature=0.5,
    )
  response = completion.choices[0].text
seed_text = str(response)
import random
num_words = random.randint(1, 1000)  # number of words to generate after seed_text
for i in range(num_words):
    sequence = tokenizer.texts_to_sequences([seed_text])[0]
    sequence = np.array(sequence)
    prediction = model.predict(sequence)[0]
    prediction = np.argmax(prediction)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == prediction:
            output_word = word
            break
    seed_text += " " + output_word

print(seed_text)
# Save the generated text to a file
with open("outputs/output.txt", "w", encoding='iso-8859-1') as f:
    f.write(seed_text)
