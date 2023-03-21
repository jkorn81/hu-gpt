sampling_generator <- function(){
  
  function(){
    
    batch <- sample(all_samples, FLAGS$batch_size)
    all_samples <- all_samples[-batch]
    
    sentences <- sentence[batch]
    next_words <- next_word[batch]
    
    # vectorization
    X <- array(0, dim = c(FLAGS$batch_size, FLAGS$maxlen, length(vocab)))
    y <- array(0, dim = c(FLAGS$batch_size, length(vocab)))
    
    
    for(i in 1:batch_size){
      
      X[i,,] <- sapply(vocab, function(x){
        as.integer(x == sentences[i])
      })
      
      y[i,] <- as.integer(vocab == next_words[i])
      
    }
    
    # return data
    list(X, y)
  }
}