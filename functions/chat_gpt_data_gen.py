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
  #open text file
  from datetime import datetime
  from datetime import date
  now = datetime.now()
  current_time = now.strftime("%H:%M:%S")
  current_time = current_time.replace(":","")
  today = date.today()
  time = str(today)+str(current_time)
  text_file = open("./data/data"+time+".txt", "w")
 
  #write string to file
  text_file.write(response)
 
  #close file
  text_file.close()
