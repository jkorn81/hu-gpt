# hu-gpt
We are attempting to use Open AI's existing GhatGpt to produce training, validation, and confimration data and serve as an agent in our custom RL optimizer for our custom gpt model. 

## Launch.R
The launch R script is the main script that all other resources are feed into and executed.


# Notes:

(1) Need to revise the initial and retrain modeling to run on multiple text samples randomly selected from the pool of (generated texts, web scrapped texts, etc).
(2) Need to develop custom rl based optimizer using the chatgpt model as an agent using text prompt to produce a binary answer on whether the text is readible to human levels. Need to set up data pipeline to include a confirmation text for the rl agent optimizer to use for its decisions and reward system. 
      - generates a fixed target text per day for the model to use for the RL optimization piece. 
