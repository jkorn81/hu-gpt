# - [-] - load libraries & Functions ----
source("functions/libraries.r")
libraries()
# capture the corpus vocab size ----
list_txt <- list.files(path = "processed/.", recursive = TRUE,
                          pattern = "\\.txt$", 
                          full.names = TRUE)
for(i in seq_along(length(list_txt))) {
  corp = lapply(list_txt, readtext::readtext)
}
for(i in seq_along(length(corp))) {
  string_corp = rbindlist(corp)
}
string_corp = string_corp[c(1,11:18,2:10),]
data.processor(string_corp$text)
text_corp =  tokenize_regex(data.text, simplify = TRUE)
vocab <- gsub("\\s", "", unlist(text_corp)) %>%
  unique() %>%
  sort()
assign("vocab",vocab, envir = globalenv())
print(sprintf("total words in vocab: %d", length(vocab)))