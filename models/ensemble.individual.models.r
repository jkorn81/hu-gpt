# - [0] - Load Required Packages ----
source("functions/libraries.r")
libraries()
# - [-] - Load States ----
states <- list.files(path = "states/individuals/", recursive = TRUE, # import stored h5 files in set output folder as a list
                     pattern = "\\.h5$", 
                     full.names = TRUE)
for(i in seq_along(1:length(states))) {
  assign(paste0("model", i), load_model_hdf5(states[i]))
}
# - [] - Ensemble the Models ----
shared_input <- layer_input(shape=(get_input_shape_at(model1, 1) %>% unlist))
model_list <- c(model1(shared_input))
for(i in seq_along(2:length(states))) {
  model_list <- c(model_list,get(paste0("model",i))(shared_input))
}
main_output  <- layer_average(model_list)
ens_model <- keras_model(
  inputs = c(shared_input), 
  outputs = c(main_output)
)
# - [] - Save the Model ----
ens_model %>% save_model_hdf5(paste0("states/ensembles/ens_model",gsub("\\s", "",chartr(":", " ",toString(Sys.time()))),".h5"))