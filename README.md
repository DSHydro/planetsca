# PlanetSCA Python Library

### This is a python library for mapping snow covered areas (SCA) from high-resolution PlanetScope images using a Random Forest model.

Originally modeled from Kehan Yang:
[Kehan's Project](https://github.com/KehanGit/High_resolution_snow_cover_mapping/blob/main/01_download_planetscope_images.ipynb)

[TestPyPi](https://test.pypi.org/project/planetsca/)

[Pre-Trained Model](https://huggingface.co/geo-smart/planetsca_models)

[Sample Data](https://huggingface.co/datasets/geo-smart/planetsca_datasets)

This requires you to have an account with Planet and an
[API key](https://developers.planet.com/quickstart/apis/#find-your-api-key) for
data access.

![PyPlanetSCA](https://raw.githubusercontent.com/DSHydro/PyPlanetSCA-Python-Library/main/additional_assets/PyPlanetSCA_Image.png)

#### Table of Contents <a name="table_of_contents"></a>

1. [Installation](#installation)
2. [Running the Code](#running)
3. [Functions](#functions)
4. [Virtual Environments](#VENV)

<br></br>

## Installation <a name="installation"></a>

Python must already be installed in your current working environment. See the
bottom of this README for instructions to create a new conda environment from
the terminal, or with VENV in VSCode.

To install the python package, use:

```bash
pip install --extra-index-url https://test.pypi.org/simple planetsca
```

## Development

1. Clone the repository

   ```bash
   git clone https://github.com/DSHydro/planetsca.git
   ```

2. Go to the repository

   ```bash
   cd planetsca
   ```

3. Install the package

   ```bash
   pip install -e ".[dev]"
   ```

4. Setup pre-commit

   ```bash
   pre-commit install
   ```

5. Now you are ready to use and develop the package!

## Running the code <a name="running"></a>

1. Import modules

```
from planetsca import data_gathering as dg
from planetsca import data_preparation as dp
from planetsca import model_training as mt
from planetsca import prediction_evaluation as pe
import numpy as np
```

2a. Data Gathering Variable Setup

```
# enter the Planet user API key
domain = '551'
apiKey = '_________'
item_type = "PSScene"
asset_type = "ortho_analytic_4b_sr"
bundle_type = "analytic_sr_udm2"
out_direc = '_________' #Replace with folder containing environment
```

2b. Data Gathering Path 1

```
# data download location
domain_geometry = dg.domain_shape()
print(domain_geometry)

result = dg.search_API_request_object(item_type, apiKey, domain)
id_list, geom_list = dg.prep_ID_geometry_lists(result, domain)
order_urls = dg.prepare_submit_orders(id_list, item_type, bundle_type, apiKey, domain)
dg.save_data_to_csv(order_urls)
dg.download_ready_orders(order_urls, out_direc, apiKey)
#dg.display_image(fp)
```

2c. Data Gathering Path 2

```
dg.retrieve_dataset(out_direc, file)
```

3a. Data Preparation Variable Setup

```
#Set up Paths
dir_ROI = ""
dir_raster = ""
dir_ROIraster = ""
dir_samples_root = ""
dir_samples = ""
```

3b. Data Preparation Path 1

```
df_train = data_training_existing(dir_samples)
```

3c. Data Preparation Path 2

```
data_training_new(dir_ROI, dir_raster, dir_ROIraster, dir_samples_root)
```

4a. Model Training Variable Setup

```
#Setup Paths
dir_model = ""
dir_score = ""

#Model Parameters
n_estimators =
max_dpeth =
max_features =
random_state =
n_splits =
n_repeats =
df_train =
```

4b. Model Training Path 1

```
mt.train_model(dir_model, dir_score, n_estimators, max_depth, max_features, random_state, n_splits, n_repeats, df_train)
```

4c. Model Training Path 2

```
mt.retrieve_model(out_direc, file)
```

5a. Prediction Evaluation Variable Setup

```
#Set up directory paths of file locations
dir_raster = ""
dir_out = ""
model = ""

nodata_flag = 9
```

6. Prediction Evaluation

```
pe.run_sca_prediction(dir_raster, dir_out, nodata_flag, model)
```

<br></br>

## Functions <a name="functions"></a>

Please see the [website](https://dshydro.github.io/planetsca/) for more
information and documentation on functions.

<br></br>

## Setting up your environment

### Setting up a Virtual Environment (VENV) on VSCode <a name="venv"></a>

Creating a VENV is recommended for this project as it ensures that there are no
package conflicts and that troubleshooting is much easier. The following
instructions are summarized from
[here](https://code.visualstudio.com/docs/python/environments).

1. Open Command Palette (Ctrl + Shift + P)
2. Select Venv
3. Select Desired Interpreter Path (I use 3.12.2, minimum version is 3.8)
4. Notification should show up on the bottom right corner titled "Creating
   environment (Show logs): Creating venv...
5. The venv is set up and activated if you see " _Python Version_
   (.'venv':venv)"
6. _NOTE_ After setting up a venv once, VScode will automatically start up the
   virtual environment alongside with VScode. There is no need to repeat these
   steps unless you do not see step 5.

### Setting up a conda environment in the terminal:

Find detailed instructions
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

1. Open a terminal
2. Create a new environment called "planetenv" with python version 3.8 or
   greater: `conda create -n planetenv python=3.9`
3. Activate your new environment: `conda activate planetenv`
4. Install planetsca from pip
   `pip install --extra-index-url https://test.pypi.org/simple planetsca`

To use jupyter notebooks with this conda environment:

1. Activate your new environment: `conda activate planetenv`
2. Install ipykernel: `pip install --user ipykernel`
3. Connect this environment to notebooks:
   `python -m ipykernel install --user --name=planetenv`
4. When you start a jupyter notebook, you can now select the `planetenv`
   environment kernel
