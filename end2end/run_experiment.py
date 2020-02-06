from end2end.experiment.experiment_setup import train
from end2end.common import dict_to_struct

from end2end.configuration import *


# Make the arguments' dictionary.
configuration = dict()
configuration["tf_records_folder"] = TF_RECORDS_FOLDER
configuration["output_folder"] = OUTPUT_FOLDER

configuration["input_gaussian_noise"] = 0.1
configuration["num_layers"] = 2
configuration["hidden_units"] = 256
configuration["initial_learning_rate"] = 0.001
configuration["train_seq_length"] = 500
configuration["full_seq_length"] = 6000
configuration["train_batch_size"] = 10
configuration["devel_batch_size"] = 4
configuration["test_batch_size"] = 4
configuration["train_size"] = 17
configuration["devel_size"] = 16
configuration["test_size"] = 16
configuration["num_epochs"] = 100
configuration["val_every_n_epoch"] = 5

configuration["GPU"] = 0

configuration = dict_to_struct(configuration)

train(configuration)
