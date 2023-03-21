import random
import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

os.chdir('C:/Users/Jonathan Korn/Desktop/deep.txt.gen.eoe.[v.1]')

# Load the generated string from CSV file
generated_text = pd.read_csv('outputs/output.txt')

prompt = str("Is the following text legible? Please provide an answer with a Yes or No only. Do not include any form of text in the response.  Here is the text: ")+str(generated_text)
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
text_file = open("./outputs/rank.txt", "w")
 
#write string to file
text_file.write(response)
 
#close file
text_file.close()
