# - [0] - Load Required Packages ----
source("functions/libraries.r")
libraries()
# Training & Results ----------------------------------------------------
output.models <- function(x,data.text) {
  model = load_model_hdf5(x, compile = TRUE)
  text =  tokenize_regex(data.text, simplify = TRUE)
  print(sprintf("corpus length: %d", length(text)))
  
  vocab <- gsub("\\s", "", unlist(text)) %>%
    unique() %>%
    sort()
  print(sprintf("total words: %d", length(vocab))) 
  
  sentence <- list()
  next_word <- list()
  list_words <- data.frame(word = unlist(text), stringsAsFactors = F)
  j <- 1
  
  for (i in seq(1, length(list_words$word) - FLAGS$maxlen - 1, by = FLAGS$steps)){
    sentence[[j]] <- as.character(list_words$word[i:(i+FLAGS$maxlen-1)])
    next_word[[j]] <- as.character(list_words$word[i+FLAGS$maxlen])
    j <- j + 1
  }  
  
  sample_mod <- function(preds, temperature = 1){
    preds <- log(preds)/temperature
    exp_preds <- exp(preds)
    preds <- exp_preds/sum(exp(preds))
    
    rmultinom(1, 1, preds) %>% 
      as.integer() %>%
      which.max()
  }
for(diversity in c(0.2, 0.5, 1, 1.2)){
  
  cat(sprintf("diversity: %f ---------------\n\n", diversity))
  
  start_index <- sample(1:(length(text) - FLAGS$maxlen), size = 1)
  sentence <- text[start_index:(start_index + FLAGS$maxlen - 1)]
  generated <- ""
  
  for(i in 1:200){
    
    x <- sapply(vocab, function(x){
      as.integer(x == sentence)
    })
    x <- array_reshape(x, c(1, dim(x)))
    
    preds <- predict(model, x)
    next_index <- sample_mod(preds, diversity)
    nextword <- vocab[next_index]
    
    generated <- str_c(generated, nextword, sep = " ")
    sentence <- c(sentence[-1], nextword)
    
  }
  
  cat(generated)
  cat("\n\n")
  
}
}