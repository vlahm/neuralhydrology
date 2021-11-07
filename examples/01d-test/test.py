#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, eval_run

### Train a model for a single config file
start_run(config_file=Path("m_basin.yml"))

### Evaluate run on test set
# The run directory that needs to be specified for evaluation is printed in the output log above.
run_dir = Path("/home/mike/git/macrosheds/qa_experimentation/outside_resources/neuralhydrology/examples/01b-multisite/runs/test_run_1608_182636")
eval_run(run_dir=run_dir, period="test")

### Load and inspect model predictions
#Next, we load the results file and compare the model predictions with observations.
#The results file is always a pickled dictionary with one key per basin (even for a single basin).
#The next-lower dictionary level is the temporal resolution of the predictions.
#In this case, we trained a model only on daily data ('1D'). Within the temporal resolution,
#the next-lower dictionary level are `xr`(an xarray Dataset that contains observations
#and predictions), as well as one key for each metric that was specified in the config file.
with open(run_dir / "test" / "model_epoch050" / "test_results.p", "rb") as fp:
    results = pickle.load(fp)

results.keys()

# The data variables in the xarray Dataset are named according to the name of the
#target variables, with suffix `_obs` for the observations and suffix `_sim` for the simulations.
results['01022500']['1D']['xr']

# Let's plot the model predictions vs. the observations
# extract observations and simulations
qobs = results['01022500']['1D']['xr']['QObs(mm/d)_obs']
# any([x == -999 for x in qobs])
qsim = results['01022500']['1D']['xr']['QObs(mm/d)_sim']

fig, ax = plt.subplots(figsize=(16,10))
ax.plot(qobs['date'], qobs)
ax.plot(qsim['date'], qsim)
ax.set_ylabel("Discharge (mm/d)")
ax.set_title(f"Test period - NSE {results['01022500']['1D']['NSE']:.3f}")

# Next, we are going to compute all metrics that are implemented in the neuralHydrology package.
#You will find additional hydrological signatures implemented in `neuralhydrology.evaluation.signatures`.
values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1))
for key, val in values.items():
    print(f"{key}: {val:.3f}")
