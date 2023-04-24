#######################################################################################
# - [-] - Set Working Directory ----
setwd("C:/Users/Jonathan Korn/Desktop/deep.txt.gen.eoe.[v.1]")
# - [-] - load libraries & Functions ----
source("functions/libraries.r")
libraries()
source("functions/data.processor.r")
# - [-] -source text samples ----
time = Sys.time()
hour = hour(time)
while(TRUE){
  if(hour <= 6){
    source_python("functions/chat_gpt_data_gen.py") #Need to loop to collect daily training samples. x amount of txt files.
    print("data collection processing running...")
  } else {
    setwd("C:/Users/Jonathan Korn/Desktop/deep.txt.gen.eoe.[v.1]")
    list_of_states <- list.files(path = "states/.", recursive = TRUE,
                                 pattern = "\\.h5$", 
                                 full.names = TRUE)
    if(length(list_of_states)==0){
      setwd("C:/Users/Jonathan Korn/Desktop/deep.txt.gen.eoe.[v.1]")
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
        len = sample(1:length(list_of_txt),1)
        text = c(string$text[len])
        data.processor(text)
        data = data.frame("text" = as.character(data.text))
        write.csv(data, paste0("processed/data", i,".csv"))
      }
      source_python("models/initial_gpt.py")
    } else {
      while (TRUE) {
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
          len = sample(1:length(list_of_txt),1)
          text = c(string$text[len])
          data.processor(text)
          data = data.frame("text" = as.character(data.text))
          write.csv(data, paste0("processed/data", i,".csv"))
        }
        source_python("models/retrain_gpt.py")
        source_python("functions/gpt-confirmation.py") # This needs to add similarity as well to the one sample daily test text. Both readability and similarity need to be met to start sourcing the next samples and retraining with new samples.
        if (readLines("outputs/rank.txt") == "Yes") {
          break
        }
      }
    }
  }
  
  
  # Need to figure out how to use the gpt-confirmation of legibility to retrain the model. 
  # Need a SARSA that can learn to retrain or not based on readability sanctioned by gt_confirmation.
  # Need while loop to continue training on source of texts while readability and similarity are not met and re-refresh source of texts and continue to train  until readability and similarity are met and refresh and continue.
}