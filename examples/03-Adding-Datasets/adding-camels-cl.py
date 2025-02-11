#!/usr/bin/env python
# coding: utf-8

# # Adding a New Dataset
# 
# There exist two different options to use a different dataset within the `neuralHydrology` library.
# 
# 1. Preprocess your data to use the `GenericDataset` in `neuralhydrology.datasetzoo.genericdataset`.
# 2. Implement a new dataset class, inheriting from `BaseDataset` in `neuralhydrology.datasetzoo.basedataset`.
# 
# Using the `GenericDataset` is recommended and does not require you to add/change a single line of code, while writing a new dataset gives you more freedom to do whatever you want.
# 
# ## Using the GenericDataset
# 
# With the release of version 0.9.6-beta, we added a `GenericDataset`. This class can be used with any data, as long as the data is preprocessed in the following way:
# 
# - The data directory (config argument `data_dir`) must contain a folder 'time_series' and (if static attributes are used) a folder 'attributes'.
# - The folder 'time_series' contains one netcdf file (.nc or .nc4) per basin, named '<basin_id\>.nc/nc4'. The netcdf file has to have one coordinate called `date`, containing the datetime index.
# - The folder 'attributes' contains one or more comma-separated file (.csv) with static attributes, indexed by basin id. Attributes files can be divided into groups of basins or groups of features (but not both).
# 
# If you prepare your data set following these guidelines, you can simply set the config argument `dataset` to `generic` and set the `data_dir` to the path of your preprocessed data directory.
# 
# ## Adding a Dataset Class
# 
# The rest of this tutorial will show you how to add a new dataset class to the `neuralhydrology.datasetzoo`. As an example, we will use the [CAMELS-CL](https://hess.copernicus.org/articles/22/5817/2018/) dataset.

# In[3]:


import sys
from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import xarray
from tqdm.notebook import tqdm

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


# ### Template
# Every dataset has its own file in `neuralhydrology.datasetzoo` and follows a common template. The template can be found [here](https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/datasetzoo/template.py). 
# 
# The most important points are:
# - All dataset classes have to inherit from `BaseDataset` implemented in `neuralhydrology.datasetzoo.basedataset`.
# - All dataset classes have to accept the same inputs upon initialization (see below)
# - Within each dataset class, you have to implement two methods:
#   - `_load_basin_data()`: This method loads the time series data for a single basin of the dataset (e.g. meteorological forcing data and streamflow) into a time-indexed pd.DataFrame.
#   - `_load_attributes()`: This method loads the catchment attributes for all basins in the dataset and returns a basin-indexed pd.DataFrame with attributes as columns.
#   
# 
# `BaseDataset` is a map-style [Pytorch Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) that implements the core logic for all data sets. It takes care of multiple temporal resolutions, data fusion, normalizations, sanity checks, etc. and also implements the required methods `__len__` (returns number of total training samples) and `__getitem__` (returns a single training sample for a given index), which PyTorch data loaders use, e.g., to create mini-batches for training. However, all of this is not important if you just want to add another dataset to the `neuralHydrology` library.
# 
# ### Preprocessing CAMELS-CL
# Because the CAMELS-CL dataset comes in a rather unusual file structure, we added a function to create per-basin csv files with all timeseries features. You can find the function `preprocess_camels_cl_dataset` in `neuralhydrology.datasetzoo.camelscl`, which will create a subfolder called `preprocessed` containing the per-basin files. For the remainder of this tutorial, we assume that this folder and the per-basin csv files exist.
# 
# ### Class skeleton
# For the sake of this tutorial, we will omit doc-strings. However, when adding your dataset class we highly encourage you to add extensive doc-strings, as we did for all dataset classes in this package. We use [Python type annotations](https://docs.python.org/3/library/typing.html) everywhere, which facilitates code development with any modern IDE as well as makes it easier to understand what is happening inside a function or class.
# 
# The class skeleton looks like this:

# In[7]:


class CamelsCL(BaseDataset):
    
    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        
        # Initialize `BaseDataset` class
        super(CamelsCL, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       basin=basin,
                                       additional_features=additional_features,
                                       id_to_int=id_to_int,
                                       scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load timeseries data of one specific basin"""
        raise NotImplementedError

    def _load_attributes(self) -> pd.DataFrame:
        """Load catchment attributes"""
        raise NotImplementedError


# ### Data loading functions
# 
# For all datasets, we implemented the actual data loading (e.g., from the txt or csv files) in separate functions outside of the class so that these functions are usable everywhere. This is useful for example when you want to inspect or visualize the discharge of a particular basin or do anything else with the basin data. These functions are implemented within the same file (since they are specific to each data set) and we use those functions from within the class methods.
# 
# So let's start by implementing a function that reads a single basin file of time series data for a given basin identifier.

# In[5]:


def load_camels_cl_timeseries(data_dir: Path, basin: str) -> pd.DataFrame:
    preprocessed_dir = data_dir / "preprocessed"
    
    # make sure the CAMELS-CL data was already preprocessed and per-basin files exist.
    if not preprocessed_dir.is_dir():
        msg = [
            f"No preprocessed data directory found at {preprocessed_dir}. Use preprocessed_camels_cl_dataset ", 
             "in neuralhydrology.datasetzoo.camelscl to preprocess the CAMELS CL data set once into ",
             "per-basin files."
        ]
        raise FileNotFoundError("".join(msg))
        
    # load the data for the specific basin into a time-indexed dataframe
    basin_file = preprocessed_dir / f"{basin}.csv"
    df = pd.read_csv(basin_file, index_col='date', parse_dates=['date'])
    return df


# Most of this should be easy to follow. First we check that the data was already preprocessed and if it wasn't, we throw an appropriate error message. Then we proceed to load the data into a pd.DataFrame and we make sure that the index is converted into a datetime format.
# 
# Next, we need a function to load the attributes, which are stored in a file called `1_CAMELScl_attributes.txt`. We assume that this file exist in the root directory of the dataset (such information is useful to add to the docstring!). The dataframe that this function has to return must be basin-indexed with attributes as columns. Furthermore, we accept an optional argument `basins`, which is a list of strings. This list can specify basins of interest and if passed, we only return the attributes for said basins.

# In[6]:


def load_camels_cl_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    
    # load attributes into basin-indexed dataframe
    attributes_file = data_dir / '1_CAMELScl_attributes.txt'
    df = pd.read_csv(attributes_file, sep="\t", index_col="gauge_id").transpose()

    # convert all columns, where possible, to numeric
    df = df.apply(pd.to_numeric, errors='ignore')

    # convert the two columns specifying record period start and end to datetime format
    df["record_period_start"] = pd.to_datetime(df["record_period_start"])
    df["record_period_end"] = pd.to_datetime(df["record_period_end"])

    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError('Some basins are missing static attributes.')
        df = df.loc[basins]

    return df


# ### Putting everything together
# 
# Now we have all required pieces and can finish the dataset class. Notice that we have access to all class attributes from the parent class in all methods (such as the config, which is stored in `self.cfg`). In the `_load_attributes` method, we simply defer to the attribute loading function we implemented above. The BaseDataset will take care of removing all attributes that are not specified as input features. It will also check for missing attributes in the `BaseDataset`, so you don't have to take care of this here.

# In[8]:


class CamelsCL(BaseDataset):
    
    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        
        # Initialize `BaseDataset` class
        super(CamelsCL, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       basin=basin,
                                       additional_features=additional_features,
                                       id_to_int=id_to_int,
                                       scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load timeseries data of one specific basin"""
        return load_camels_cl_timeseries(data_dir=self.cfg.data_dir, basin=basin)

    def _load_attributes(self) -> pd.DataFrame:
        """Load catchment attributes"""
        return load_camels_cl_attributes(self.cfg.data_dir, basins=self.basins)


# ### Integrating the dataset class into neuralHydrology
# 
# With these few lines of code, you are ready to use a new dataset within the `neuralHydrology` framework. The only thing missing is to link the new dataset in the `get_dataset()` function, implemented in `neuralhydrology.datasetzoo.__init__.py`. Again, we removed the doc-string for brevity ([here](https://neuralhydrology.readthedocs.io/en/latest/api/neuralhydrology.datasetzoo.html#neuralhydrology.datasetzoo.get_dataset) you can find the documentation), but the code of this function is as simple as this:

# In[11]:


from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.datasetzoo.camelscl import CamelsCL
from neuralhydrology.datasetzoo.camelsgb import CamelsGB
from neuralhydrology.datasetzoo.camelsus import CamelsUS
from neuralhydrology.datasetzoo.hourlycamelsus import HourlyCamelsUS
from neuralhydrology.utils.config import Config


def get_dataset(cfg: Config,
                is_train: bool,
                period: str,
                basin: str = None,
                additional_features: list = [],
                id_to_int: dict = {},
                scaler: dict = {}) -> BaseDataset:
    
    # check config argument and select appropriate data set class
    if cfg.dataset == "camels_us":
        Dataset = CamelsUS
    elif cfg.dataset == "camels_gb":
        Dataset = CamelsGB
    elif cfg.dataset == "hourly_camels_us":
        Dataset = HourlyCamelsUS
    elif cfg.dataset == "camels_cl":
        Dataset = CamelsCL
    else:
        raise NotImplementedError(f"No dataset class implemented for dataset {cfg.dataset}")
    
    # initialize dataset
    ds = Dataset(cfg=cfg,
                 is_train=is_train,
                 period=period,
                 basin=basin,
                 additional_features=additional_features,
                 id_to_int=id_to_int,
                 scaler=scaler)
    return ds


# Now, by settig `dataset: camels_cl` in the config file, you are able to train a model on the CAMELS-CL data set. 
# 
# The available time series features are:
# - tmax_cr2met
# - precip_mswep
# - streamflow_m3s
# - tmin_cr2met
# - pet_8d_modis
# - precip_chirps
# - pet_hargreaves
# - streamflow_mm
# - precip_cr2met
# - swe
# - tmean_cr2met
# - precip_tmpa
# 
# For a list of available attributes, look at the `1_CAMELScl_attributes.txt` file or make use of the above implemented function to load the attributes into a pd.DataFrame.
