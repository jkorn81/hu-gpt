#######################################################################################
# - [-] - Set Working Directory ----
setwd("C:/Users/Jonathan Korn/Desktop/deep.txt.gen.eoe.[v.1]")
# - [-] - load libraries & Functions ----
source("functions/libraries.r")
libraries()
source("functions/data.processor.r")
# - [-] -source text samples ----
source_python("functions/chat_gpt_data_gen.py") #Need to loop to collect daily training samples. x amount of txt files.

# Need one sample to be generated to become the daily text we are trying to master. 
# When retrain generates the text, the source text should be the beginning of the one sample test text.

# - [-] - preprocess the data ----
list_of_txt <- list.files(path = "data/.", recursive = TRUE,
                          pattern = "\\.txt$", 
                          full.names = TRUE)
for(i in seq_along(length(list_of_txt))) {
  text = lapply(list_of_txt, readtext::readtext)
}
for(i in seq_along(length(text))) {
  string = rbindlist(text)
}
for(i in 1:length(list_of_txt)) {
  starttime = Sys.time()
  print(paste0("Sample ...",i)) 
  len = i + 0
  text = c(string$text[len])
  data.processor(text)
  data = data.frame("text" = as.character(data.text))
  write.csv(data, paste0("processed/data.csv"))
}
source_python("models/initial_gpt.py")

while (TRUE) {
  # - [-] -source text samples ----
  source_python("functions/chat_gpt_data_gen.py") #This needs to be the daily collected samples
  # - [-] - preprocess the data ----
  list_of_txt <- list.files(path = "data/.", recursive = TRUE,
                            pattern = "\\.txt$", 
                            full.names = TRUE)
  for (i in seq_along(length(list_of_txt))) {
    text = lapply(list_of_txt, readtext::readtext)
  }
  for (i in seq_along(length(text))) {
    string = rbindlist(text)
  }
  for (i in 1:length(list_of_txt)) {
    starttime = Sys.time()
    print(paste0("Sample ...", i)) 
    len = i + 0
    text = c(string$text[len])
    data.processor(text)
    data = data.frame("text" = as.character(data.text))
    write.csv(data, paste0("processed/data.csv"))
  }
  source_python("models/retrain_gpt.py")
  source_python("functions/gpt-confirmation.py") # This needs to add similarity as well to the one sample daily test text. Both readability and similarity need to be met to start sourcing the next samples and retraining with new samples.
  if (readLines("outputs/rank.txt") == "No") {
    break
  }
}

# Need to figure out how to use the gpt-confirmation of legibility to retrain the model. 
# Need a SARSA that can learn to retrain or not based on readability sanctioned by gt_confirmation.
# Need loop to continue while readability and similarity are not met and re-refresh and continue to train if readability and similarity are met.