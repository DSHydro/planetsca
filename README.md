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

To install the python package, use(Ensure Python is downloaded, check bottom of
README for VENV instructions):

```bash
pip install -i https://test.pypi.org/simple/ planetsca
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

2. Set Variables and folder locations

```
domainID = '551'
# enter the Planet user API key
apiKey = '_________'
item_type = "PSScene"
asset_type = "ortho_analytic_4b_sr"
bundle_type = "analytic_sr_udm2"
```

3. Data Gathering

```

# data download location
out_direc = '_________' #Replace with folder containing environment
domain_geometry = dg.domain_shape()
print(domain_geometry)
result = dg.api_search(item_type, apiKey)
geojson_data, gdf = dg.downloadable_PlanetIDs(result, domain_geometry)
id_list = dg.id_gemoetry_lists(geojson_data, gdf)
order_urls = dg.submit_orders(id_list, item_type, bundle_type, apiKey)
dg.save_data_to_csv(order_urls)
dg.download_orders(order_urls, out_direc, apiKey)
#dg.display_image(fp)
```

4. Data Preparation

```
df_train = dp.data_labeling("","","", "", 'sample_174k.csv')
dir_model = 'random_forest_20240116_binary_174K.joblib'
dir_score = 'random_forest_20240116_binary_174K_scores.csv'
```

5a. Model Training

```
mt.train_model(dir_model, dir_score, 10, 10, 4, 1, df_train)
```

5b. Retrieve Pre-Made Model from Hugging Faces

```
mt.retrieve_model(out_direc):
```

6. Prediction Evaluation

```
dir_raster = '20180528_181110_1025_3B_AnalyticMS_SR_clip.tiff'
dir_out = 'SCA'
pe.single_image_evaluation(dir_raster, dir_model, dir_out)
```

<br></br>

## Functions <a name="functions"></a>

Please see the FUNCTIONS.md file for additional documentation of functions:
[FUNCTIONS.md](https://github.com/DSHydro/planetsca/blob/main/FUNCTIONS.md)

<br></br>

## Setting up a Virtual Environment (VENV) on VSCode <a name="venv"></a>

### Creating a VENV is recommended for this project as it ensures that there are no package conflicts and that troubleshooting is much easier.

### Instructions are summarized from [here](https://code.visualstudio.com/docs/python/environments)

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
