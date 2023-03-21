# - [0] - Load Required Packages ----
source("functions/libraries.r")
libraries()
data.processor <- function(x) {
  starttime = Sys.time()
  print("Pre-processing Text ...")
  text = c(lapply(x, as.character))
  data.text = text
  data.text = tolower(data.text)
  data.text = iconv(data.text, "latin1", "ASCII", sub = " ")
  data.text = gsub("^NA| NA ", " ", data.text)
  data.text = gsub("/| / ", " ", data.text)
  data.text = tm::removeNumbers(data.text)
  data.text = tm::stripWhitespace(data.text)
  assign("data.text",data.text, envir = globalenv())
  end.time <- Sys.time()
  time.taken <- end.time - starttime
  print(c("Pre-processing Text Complete ..."))
  print(time.taken)
}