import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pandas import DataFrame
from rasterio import features
from rasterio.enums import MergeAlg

warnings.filterwarnings("ignore")


def vector_rasterize(dir_vector, dir_raster, dir_out, flag_output):
    vector = gpd.read_file(dir_vector)
    # Get list of geometries for all features in vector file
    list(vector.geometry)

    # Open example raster
    raster = rasterio.open(dir_raster)

    # reproject vector to raster
    vector = vector.to_crs(raster.crs)

    # create tuples of geometry, value pairs, where value is the attribute value you want to burn
    geom_value = (
        (geom, value) for geom, value in zip(vector.geometry, vector["label"])
    )

    # Rasterize vector using the shape and transform of the raster
    rasterized = features.rasterize(
        geom_value,
        out_shape=raster.shape,
        transform=raster.transform,
        all_touched=True,
        fill=9,  # background value
        merge_alg=MergeAlg.replace,
        dtype=np.float32,
    )

    if flag_output:
        with rasterio.open(
            dir_out,
            "w",
            driver="GTiff",
            transform=raster.transform,
            dtype=rasterio.float32,
            count=1,
            width=raster.width,
            height=raster.height,
        ) as dst:
            dst.write(rasterized, indexes=1)
    return rasterized


def data_training_existing(dir_samples: str) -> DataFrame:
    """
    Creates training data from a csv file

    Parameters:
    - dir_samples: File location of the csv file

    Returns:
    - DataFrame: DataFrame of training data
    """
    df_train = pd.read_csv(dir_samples)
    return df_train


def data_training_new(
    dir_ROI: str, dir_raster: str, dir_ROIraster: str, dir_samples_root: str
) -> DataFrame:
    """
    Creates training data from scratch

    Parameters:
    - dir_ROI: Directory path to regions of interest
    - dir_raster: Directory to Planet image for training
    - dir_ROIraster: Directory path to a ROI converted to shape mask
    - dir_samples_root: Directory path to csv of training data extracted from images

    Returns:
    - DataFrame: DataFrame of training data
    """

    flag_output = True

    # rasterize ROI
    ROI = vector_rasterize(
        dir_vector=dir_ROI,
        dir_raster=dir_raster,
        dir_out=dir_ROIraster,
        flag_output=flag_output,
    )

    # save surface reflectance and label to csv file
    N_scale = 10000.0
    img = rasterio.open(dir_raster)
    ROI = rasterio.open(dir_ROIraster)
    img_read = img.read() / N_scale
    df_img = pd.DataFrame(img_read.reshape([4, -1]).T)
    df_label = pd.DataFrame(ROI.read().reshape([1, -1]).T)
    df_train = pd.concat([df_img, df_label], axis=1)
    df_train.columns = ["blue", "green", "red", "nir", "label"]
    df_train["ndvi"] = (df_train["nir"] - df_train["red"]) / (
        df_train["nir"] + df_train["red"]
    )
    df_train = df_train[df_train.label != 9]
    df_train.label = np.where(df_train.label > 0, 1, 0)
    dir_samples = dir_samples_root + str(int(len(df_train.index) / 1000)) + "k.csv"
    df_train.to_csv(dir_samples, index=False)

    return df_train
