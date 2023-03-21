#######################################################################################
#######################################################################################
# - [-] - Set Working Directory ----
dir = getwd()
setwd(dir)
# - [-] - load libraries & Functions ----
source("functions/libraries.r")
libraries()
source("functions/data.processor.r")
source("functions/sample_generator.r")
source("functions/sample_mod.r")
source("functions/vocab.r")
# [-] - Initial Model State ----
# - load data ----
list_of_txt <- list.files(path = "processed/.", recursive = TRUE,
                          pattern = "\\.txt$", 
                          full.names = TRUE)
len_list1 = 1:length(list_of_txt[1])
for(i in len_list1) {
  starttime = Sys.time()
  print(paste0("Import Sample ...",i)) 
  for(x in c(len_list1)) {
    text = lapply(list_of_txt, readtext::readtext)
  }
  for(y in seq_along(length(text))) {
    string = rbindlist(text)
  }
  len = i
  data.processor(string$text[len])
  # - Set parameters ----
  FLAGS <- flags(
    flag_integer('filters_cnn', 6),
    flag_integer('filters_lstm', 6),
    flag_numeric('reg1', 5e-4),
    flag_numeric('reg2', 5e-4),
    flag_numeric('batch_size',5),
    flag_numeric('maxlen', 10),
    flag_numeric('steps', 50),
    flag_numeric('embedding_dim', 20),
    flag_numeric('kernel', 5),
    flag_numeric('leaky_relu', 0.50),
    flag_numeric('epochs', 10),
    flag_numeric('lr', 0.002)
  )
  # - Prepare Tokenized Forms of Each Text ----
  text =  tokenize_regex(data.text, simplify = TRUE)
  print(sprintf("corpus length: %d", length(text)))

  #vocab <- gsub("\\s", "", unlist(text)) %>%
  #  unique() %>%
  #  sort()
  #print(sprintf("total words: %d", length(vocab))) 
  
  sentence <- list()
  next_word <- list()
  list_words <- data.frame(word = unlist(text), stringsAsFactors = F)
  j <- 1
  
  for (z in seq(1, length(list_words$word) - FLAGS$maxlen - 1, by = FLAGS$steps)){
    sentence[[j]] <- as.character(list_words$word[z:(z+FLAGS$maxlen-1)])
    next_word[[j]] <- as.character(list_words$word[z+FLAGS$maxlen])
    j <- j + 1
  }
  # Model Definition ----
  model <- keras_model_sequential(name = gsub("\\s", "",chartr(":", " ",toString(i)))) %>% 
    layer_conv_1d(input_shape = c(FLAGS$maxlen, length(vocab)),
                  FLAGS$filters_cnn/4, FLAGS$kernel, 
                  kernel_initializer = "VarianceScaling",
                  kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg1, l2 = FLAGS$reg1),
                  padding = "same", strides = 1L
    ) %>%
    layer_activation_leaky_relu(FLAGS$leaky_relu) %>%
    layer_conv_1d(
      FLAGS$filters_cnn, FLAGS$kernel-1, 
      kernel_initializer = "VarianceScaling",
      kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg1, l2 = FLAGS$reg1),
      padding = "same", strides = 1L
    ) %>%  
    layer_activation_leaky_relu(FLAGS$leaky_relu) %>%
    layer_conv_1d(
      FLAGS$filters_cnn/2, FLAGS$kernel-2, 
      kernel_initializer = "VarianceScaling",
      kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg1, l2 = FLAGS$reg1),
      padding = "same", strides = 1L
    ) %>%
    layer_conv_1d(
      FLAGS$filters_cnn/2, FLAGS$kernel-2, 
      kernel_initializer = "VarianceScaling",
      kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg1, l2 = FLAGS$reg1),
      padding = "same", strides = 1L
    ) %>%
    layer_activation_leaky_relu(FLAGS$leaky_relu) %>%
    layer_conv_1d(
      FLAGS$filters_cnn/4, FLAGS$kernel-3, 
      kernel_initializer = "VarianceScaling",
      kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg1, l2 = FLAGS$reg1),
      padding = "same", strides = 1L
    ) %>%
    layer_activation_leaky_relu(FLAGS$leaky_relu) %>%
    layer_batch_normalization()  %>%
    layer_dropout(0.5) %>%
    layer_lstm(FLAGS$filters_lstm, input_shape = c(FLAGS$maxlen, length(vocab))) %>%
    layer_dropout(FLAGS$leaky_relu) %>%
    layer_dense(length(vocab)) %>%
    layer_activation("softmax")
  
  optimizer <- optimizer_nadam(lr = FLAGS$lr)
  
  model %>% compile(
    loss = "categorical_crossentropy", 
    optimizer = optimizer
  )
  # - Training & Results ----
  batch_size <- FLAGS$batch_size
  all_samples <- 1:length(sentence)
  num_steps <- trunc(length(sentence)/batch_size)
  
  for(r in range(1)){
    model %>% fit_generator(generator = sampling_generator(),
                            steps_per_epoch = num_steps,
                            epochs = FLAGS$epochs,
                            view_metrics = getOption("keras.view_metrics",
                                                     default = "auto"))
  }
  
  for(diversity in c(0.2, 0.5, 1, 1.2)){
    
    cat(sprintf("diversity: %f ---------------\n\n", diversity))
    
    start_index <- sample(1:(length(text) - FLAGS$maxlen), size = 1)
    sentence <- text[start_index:(start_index + FLAGS$maxlen - 1)]
    generated <- ""
    
    for(t in 1:200){
      
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
  # Store Model States ----
  model %>% save_model_hdf5(paste0("states/individuals/model",i,".h5"))
}
#######################################################################################
#######################################################################################
# - [-] - Transfer Learning ----
states <- list.files(path = "states/individuals/.", recursive = TRUE,
                          pattern = "\\.h5$", 
                          full.names = TRUE)
len_list1 = 2:length(list_of_txt)
for(i in len_list1) {
  starttime = Sys.time()
  print(paste0("Import Sample ...",i)) 
  for(x in c(len_list1)) {
    text = lapply(list_of_txt, readtext::readtext)
  }
  for(y in seq_along(length(text))) {
    string = rbindlist(text)
  }
  string = string[c(1,11:18,2:10),]
  len = i
  data.processor(string$text[len])
  # - Set parameters ----
  FLAGS <- flags(
    flag_integer('filters_cnn', 6),
    flag_integer('filters_lstm', 6),
    flag_numeric('reg1', 5e-4),
    flag_numeric('reg2', 5e-4),
    flag_numeric('batch_size',5),
    flag_numeric('maxlen', 10),
    flag_numeric('steps', 50),
    flag_numeric('embedding_dim', 20),
    flag_numeric('kernel', 5),
    flag_numeric('leaky_relu', 0.50),
    flag_numeric('epochs', 10),
    flag_numeric('lr', 0.002)
  )
  # - Prepare Tokenized Forms of Each Text ----
  text =  tokenize_regex(data.text, simplify = TRUE)
  print(sprintf("corpus length: %d", length(text)))
  
  #vocab <- gsub("\\s", "", unlist(text)) %>%
  #  unique() %>%
  #  sort()
  #print(sprintf("total words: %d", length(vocab))) 
  
  sentence <- list()
  next_word <- list()
  list_words <- data.frame(word = unlist(text), stringsAsFactors = F)
  j <- 1
  
  for (z in seq(1, length(list_words$word) - FLAGS$maxlen - 1, by = FLAGS$steps)){
    sentence[[j]] <- as.character(list_words$word[z:(z+FLAGS$maxlen-1)])
    next_word[[j]] <- as.character(list_words$word[z+FLAGS$maxlen])
    j <- j + 1
  }
  # Model Definition ----
  model <- keras_model_sequential(name = gsub("\\s", "",chartr(":", " ",toString(i)))) %>% 
    layer_conv_1d(input_shape = c(FLAGS$maxlen, length(vocab)),
                  FLAGS$filters_cnn/4, FLAGS$kernel, 
                  kernel_initializer = "VarianceScaling",
                  kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg1, l2 = FLAGS$reg1),
                  padding = "same", strides = 1L
    ) %>%
    layer_activation_leaky_relu(FLAGS$leaky_relu) %>%
    layer_conv_1d(
      FLAGS$filters_cnn, FLAGS$kernel-1, 
      kernel_initializer = "VarianceScaling",
      kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg1, l2 = FLAGS$reg1),
      padding = "same", strides = 1L
    ) %>%  
    layer_activation_leaky_relu(FLAGS$leaky_relu) %>%
    layer_conv_1d(
      FLAGS$filters_cnn/2, FLAGS$kernel-2, 
      kernel_initializer = "VarianceScaling",
      kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg1, l2 = FLAGS$reg1),
      padding = "same", strides = 1L
    ) %>%
    layer_conv_1d(
      FLAGS$filters_cnn/2, FLAGS$kernel-2, 
      kernel_initializer = "VarianceScaling",
      kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg1, l2 = FLAGS$reg1),
      padding = "same", strides = 1L
    ) %>%
    layer_activation_leaky_relu(FLAGS$leaky_relu) %>%
    layer_conv_1d(
      FLAGS$filters_cnn/4, FLAGS$kernel-3, 
      kernel_initializer = "VarianceScaling",
      kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg1, l2 = FLAGS$reg1),
      padding = "same", strides = 1L
    ) %>%
    layer_activation_leaky_relu(FLAGS$leaky_relu) %>%
    layer_batch_normalization()  %>%
    layer_dropout(0.5) %>%
    layer_lstm(FLAGS$filters_lstm, input_shape = c(FLAGS$maxlen, length(vocab)),
               kernel_initializer = "VarianceScaling",
               kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg2, l2 = FLAGS$reg2)) %>%
    layer_dropout(FLAGS$leaky_relu) %>%
    layer_dense(length(vocab)) %>%
    layer_activation("softmax")
  
  optimizer <- optimizer_nadam(lr = FLAGS$lr)
  
  model %>% compile(
    loss = "categorical_crossentropy", 
    optimizer = optimizer
  )
  # - Training & Results ----
  batch_size <- FLAGS$batch_size
  all_samples <- 1:length(sentence)
  num_steps <- trunc(length(sentence)/batch_size)
  
  weights <- get_weights(load_model_hdf5(toString(last(states,1))))
  set_weights(model, weights)
  
  for(r in range(1)){
    model %>% fit_generator(generator = sampling_generator(),
                            steps_per_epoch = num_steps,
                            epochs = FLAGS$epochs,
                            view_metrics = getOption("keras.view_metrics",
                                                     default = "auto"))
  }
  
  for(diversity in c(0.2, 0.5, 1, 1.2)){
    
    cat(sprintf("diversity: %f ---------------\n\n", diversity))
    
    start_index <- sample(1:(length(text) - FLAGS$maxlen), size = 1)
    sentence <- text[start_index:(start_index + FLAGS$maxlen - 1)]
    generated <- ""
    
    for(t in 1:200){
      
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
  # Store Model States ----
  model %>% save_model_hdf5(paste0("states/individuals/model",i,".h5"))
}