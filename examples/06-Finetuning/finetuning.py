#!/usr/bin/env python
# coding: utf-8

# # How-to Finetune
#
# This tutorial shows how to adapt a pretrained model to a different, eventually much smaller dataset, a concept called finetuning. Finetuning is
#well-established in machine learning and thus nothing new. Generally speaking, the idea is to use a (very) large and diverse dataset to learn a
#general understanding of the underlying problem first and then, in a second step, adapt this general model to the target data. Usually,
#especially if the available target data is limited, pretraining plus finetuning yields (much) better results than only considering the final target data.
#
# The connection to hydrology is the following: Often, researchers or operators are only interested in a single basin. However,
#considering that a Deep Learning (DL) model has to learn all (physical) process understanding from the available training data, it might be
#understandable that the data records of a single basin might not be enough (see e.g. the presentation linked at
#[this](https://meetingorganizer.copernicus.org/EGU2020/EGU2020-8855.html) EGU'20 abstract)
#
# This is were we apply the concept of pretraining and finetuning: First, we train a DL model (e.g. an LSTM) with a large and diverse,
#multi-basin dataset (e.g. CAMELS) and then finetune this model to our basin of interest. Everything you need is available in the `neuralHydrology`
#package and in this notebook we will give you an overview of how to actually do it.
#
# **Note**: Finetuning can be a tedious task and is usually very sensitive to the learning rate as well as the number of epochs used for finetuning.
#One reason is that the pretrained models are usually quite large. In fact, most often they are much larger than what would be possible to train for
#just a single basin. So during finetuning, we have to make sure that this large capacity is not negatively impacting our model results. Common
#approaches are to a) only allow parts of the model to be adapted during finetuning and/or b) to train with a much lower learning rate. So far,
#no publication was published that presents a universally working approach for finetuning in hydrology. So be aware that the results may vary and
#you might need to invest some time before finding a good strategy. However, in our experience it was always possible to get better results _with_
#finetuning than without.
#
# **To summarize**: If you are interested in getting the best-performing Deep Learning model for a single basin, pretraining on a large and diverse
#dataset, followed by finetuning the pretrained model on your target basin is the way to go.

import pickle
from pathlib import Path

import numpy as np

from neuralhydrology.nh_run import start_run, eval_run, finetune

# ## Pretraining
#
# In the first step, we need to pretrain our model on a large and possibly diverse dataset. Our target basin does not necessarily have to be a part of
#this dataset, but usually it should be better to include it.
#
# For the sake of the demonstration, we will train an LSTM on the CAMELS US dataset and then finetune this model to a random basin. Note that it is
#possible to use other inputs during pretraining and finetuning, if additional embedding layers (before the LSTM) are used, which we will ignore for now.
#Furthermore, we will concentrate only on demonstrating the "how-to" rather than striving for best-possible performance. To save time and energy, we will
#only pretrain the model for a small number of epochs. When striving for the best possible performance, you should make sure that you pretrain the model
#as best as possible, before starting to finetune.
#
# We will stick closely to the model and experimental setup from [Kratzert et al.
#(2019)](https://hess.copernicus.org/articles/23/5089/2019/hess-23-5089-2019.html). To summarize:
# - A single LSTM layer with a hidden size of 128.
# - Input sequences are 365 days and the prediction is made at the last timestep.
# - For the sake of this demonstration, we will only consider the 5 meteorological variables from the Maurer forcing data.
# - We will use the same CAMELS attributes, as in the publication mentioned above, as additional inputs at every time step so that the model can learn
#different hydrological behaviors depending on the catchment properties.
#
# For more details, take a look at the config print-out below.

config_file = Path("531_basins.yml")
start_run(config_file=config_file)

# We end with an okay'ish model that should be enough for the purpose of this demonstration. Remember we only train for a limited number of epochs here.
#
# Next, let's look in the `runs/` folder, where the folder of this model is stored to lookup the exact name.

get_ipython().system('ls runs/')

# Next, we'll load the validation results into memory so we can select a basin to demonstrate how to finetune based on the model performance. Here,
#we will select a random basin from the lower 50% of the NSE distribution, i.e. a basin where the NSE is below the median NSE. Usually, you'll see
#better performance gains for basins with lower model performance than for those where the base model is already really good.

# Load validation results for the last epoch
run_dir = Path("runs/cudalstm_maurer_531_basins_1201_213010")
with open(run_dir / "validation" / "model_epoch003" / "validation_results.p", "rb") as fp:
    validation_results = pickle.load(fp)

# Compute the median NSE from all basins, where discharge observations are available for that period
median_nse = np.median([v["1D"]["NSE"] for v in validation_results.values() if "NSE" in v["1D"].keys()])
print(f"Median NSE of the validation period {median_nse:.3f}")

# Select a random basins from the lower 50% of the NSE distribution
basins = []
for k, v in validation_results.items():
    if ("NSE" in v["1D"].keys()) and (v["1D"]["NSE"] < median_nse):
        basins.append(k)
basin = np.random.choice(basins)

print(f"Selected basin: {basin} with an NSE of {validation_results[basin]['1D']['NSE']:.3f}")

# ## Finetuning
#
# Next, we will show how to perform finetuning for the basin selected above, based on the model we just trained. The function to use is `finetune` from
#`neuralhydrology.nh_run` if you want to train from within a script or notebook. If you want to start finetuning from the command line, you can also call the
#`nh-run` utility with the `finetune` argument, instead of e.g. `train` or `evaluate`.
#
# The only thing required, similar to the model training itself, is a config file. This config however has slightly different requirements to a normal model
#config and works slightly different:
# - The config has to contain the following two arguments:
#     - `base_run_dir`: The path to the directory of the pre-trained model.
#     - `finetune_modules`: Which parts of the pre-trained model you want to finetune. Check the documentation of each model class for a list of all
#possible parts. Often only parts, e.g. the output layer, are trained during finetuning and the rest is kept fixed. There is no general rule of thumb
#and most likely you will have to try both.
# - Any additional argument contained in this config will overwrite the config argument of the pre-trained model. Everything _not_ specified will be taken
#from the pre-trained model. That is, you can e.g. specify a new basin file in the finetuning config (by `train_basin_file`) to finetune the pre-trained
#model on a different set of basins, or even just a single basin as we will do in this notebook. You can also change the learning rate, loss function,
#evaluation metrics and so on. The only thing you can not change are arguments that change the model architecture (e.g. `model`, `hidden_size` etc.),
#because this leads to errors when you try to load the pre-trained weights into the initialized model.
#
# Let's have a look at the `finetune.yml` config that we prepared for this tutorial.

get_ipython().system('cat finetune.yml')

# So out of the two arguments that are required, `base_run_dir` is still missing. We will add the argument from here and point at the directory of
#the model we just trained. Furthermore, we point to a new file for training, validation and testing, called `finetune_basin.txt`, which does not yet
#exist. We will create this file and add the basin we selected above as the only basin we want to use here. The rest are some changes to the learning
#rate and the number of training epochs as well as a new name. Also note that here, we train the full model, by selecting all model parts available for
#the `CudaLSTM` under `finetune_modules`.

# Add the path to the pre-trained model to the finetune config
with open("finetune.yml", "a") as fp:
    fp.write(f"base_run_dir: {run_dir.absolute()}")

# Create a basin file with the basin we selected above
with open("finetune_basin.txt", "w") as fp:
    fp.write(basin)

# With that, we are ready to start the finetuning. As mentioned above, we have two options to start finetuning:
# 1. Call the `finetune()` function from a different Python script or a Jupyter Notebook with the path to the config.
# 2. Start the finetuning from the command line by calling
#
# ```bash
# nh-run finetune --config-file /path/to/config.yml
# ```
#
# Here, we will use the first option.

finetune(Path("finetune.yml"))

# Looking at the validation result, we can see an increase of roughly 0.1 NSE.
#
# Last but not least, we will compare the pre-trained and the finetuned model on the test period. For this, we will make use of the `eval_run` function
#from `neuralhydrolgy.nh_run`. Alternatively, you could evaluate both runs from the command line by calling
#
# ```bash
# nh-run evaluate --run-dir /path/to/run_directory/
# ```

eval_run(run_dir, period="test")

# Next we check for the full name of the finetuning run (which we could also extract from the log output above)

get_ipython().system('ls runs/')

# Now we can call the `eval_run()` function as above, but pointing to the directory of the finetuned run. By default, this function evaluates the last
#checkpoint, which can be changed with the `epoch` argument. Here however, we use the default.

finetune_dir = Path("runs/cudalstm_maurer_531_basins_finetuned_1201_215500")
eval_run(finetune_dir, period="test")

# Now let's look at the test period results of the pre-trained base model and the finetuned model for the basin that we chose above.

# load test results of the base run
with open(run_dir / "test/model_epoch003/test_results.p", "rb") as fp:
    base_model_results = pickle.load(fp)

# load test results of the finetuned model
with open(finetune_dir / "test/model_epoch010/test_results.p", "rb") as fp:
    finetuned_results = pickle.load(fp)

# extract basin performance
base_model_nse = base_model_results[basin]['1D']['NSE']
finetune_nse = finetuned_results[basin]["1D"]["NSE"]
print(f"Basin {basin} base model performance: {base_model_nse:.3f}")
print(f"Performance after finetuning: {finetune_nse:.3f}")

# So we see roughly the same performance increase in the test period, which is great. However, note that a) our base model was not optimally trained
#(we stopped quite early) but also b) the finetuning settings were chosen rather randomly. From our experience so far, you can almost always get
#performance increases for individual basins with finetuning, but it is difficult to find settings that are universally applicable. However, this
#tutorial was just a showcase of how easy it actually is to finetune models with the `neuralHydrology` library. Now it is up to you to experiment with it.
