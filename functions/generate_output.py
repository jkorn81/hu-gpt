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
