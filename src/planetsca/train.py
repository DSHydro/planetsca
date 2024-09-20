import time
import warnings
from typing import Optional

import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.enums import MergeAlg
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")


def vector_rasterize(
    labeled_polygons_filepath: str,
    training_image_filepath: str,
    rasterized_mask_output_filepath: str = None,
):
    """
    Helper function for converting vector file to a raster file

    Parameters
    ----------
        labeled_polygons_filepath: str
            File path to shapefile or geojson file with labeled polygons
        training_image_filepath: str
            File path to Planet Scope image
        rasterized_mask_output_filepath: Optional[str]
            Optional: file path to output the rasterized labeled polygons to a geotiff file (defaults to None)

    Returns
    -------
        rasterized: np.array
            Rasterized version of the vector file
    """

    vector = gpd.read_file(labeled_polygons_filepath)
    # Get list of geometries for all features in vector file
    list(vector.geometry)

    # Open example raster
    raster = rasterio.open(training_image_filepath)

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

    if isinstance(rasterized_mask_output_filepath, str):
        print(
            f"Saving rasterized labeled polygons to: {rasterized_mask_output_filepath}"
        )
        with rasterio.open(
            rasterized_mask_output_filepath,
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


def data_training_new(
    labeled_polygons_filepath: str,
    training_image_filepath: str,
    training_data_filepath: Optional[str] = None,
    rasterized_mask_output_filepath: Optional[str] = None,
    ndvi: Optional[bool] = False,
):
    """
    Creates training data from scratch

    Parameters
    ----------
        labeled_polygons_filepath: str
            File path to shapefile or geojson file with labeled polygons
        training_image_filepath: str
            File path to Planet Scope image
        training_data_filepath: Optional[str]
            Optional: file path to output training data dataframe as a csv file (defaults to None)
        rasterized_mask_output_filepath: Optional[str]
            Optional: file path to output the rasterized labeled polygons to a geotiff file (defaults to None)
        ndvi: Optional[bool]
            Optional: Set to True to compute the Normalized Difference Vegetation Index (NDVI) and add to training data DataFrame

    Returns
    -------
        training_data_df: DataFrame
            pandas DataFrame of training data
    """

    # rasterize labeled polygons (our Regions of Interest, or ROI)
    ROI = vector_rasterize(
        labeled_polygons_filepath=labeled_polygons_filepath,
        training_image_filepath=training_image_filepath,
        rasterized_mask_output_filepath=rasterized_mask_output_filepath,
    )

    # save surface reflectance and label to csv file
    N_scale = 10000.0
    img = rasterio.open(training_image_filepath)
    img_read = img.read() / N_scale
    df_img = pd.DataFrame(img_read.reshape([4, -1]).T)
    df_label = pd.DataFrame(ROI.reshape([1, -1]).T)
    training_data_df = pd.concat([df_img, df_label], axis=1)
    training_data_df.columns = ["blue", "green", "red", "nir", "label"]
    if ndvi:
        training_data_df["ndvi"] = (
            training_data_df["nir"] - training_data_df["red"]
        ) / (training_data_df["nir"] + training_data_df["red"])
    training_data_df = training_data_df[training_data_df.label != 9]
    training_data_df.label = np.where(
        training_data_df.label > 0, 1, 0
    )  # any labels with a value > 0 is set to 1
    if isinstance(training_data_filepath, str):
        print(f"Saving training data DataFrame to: {training_data_filepath}")
        training_data_df.to_csv(training_data_filepath, index=False)

    return training_data_df.reset_index(drop=True)


def train_model(
    df_train: pd.DataFrame,
    new_model_filepath: str,
    new_model_score_filepath: str,
    n_estimators: int = 10,
    max_depth: int = 10,
    max_features: int = 4,
    random_state: Optional[int] = None,
    n_splits: int = 2,
    n_repeats: int = 2,
) -> RandomForestClassifier:
    """
    Trains and creates a new model with custom parameters

    Parameters
    ----------
        df_train: pd.DataFrame
            Dataframe containing training data, must have feature columns 'blue', 'green', 'red', 'nir' and target column 'label'
        new_model_filepath: str
            Filepath to save the model as a joblib file
        new_model_score_filepath: str
            Filepath to save the model score information as a csv file
        n_estimators: int
            Number of trees in the forest, defaults to 10
        max_depth: int
            Maximum depth of the tree, defaults to 10
        max_features: int
            Number of features to consider when looking for the best split, defaults to 4
        random_state: int
            Seed to ensure reproducibility, defaults to None
        n_splits: int
            Number of folds in the cross-validation, defaults to 2
        n_repeats: int
            Number of times cross-validator needs to be repeated, defaults to 2

    Returns
    -------
        model: RandomForestClassifier
            The newly trained model
    """

    starttime = time.process_time()
    X = df_train[["blue", "green", "red", "nir"]]
    y = df_train["label"]

    # pre-process ndvi value to -1.0 to 1.0; fill nan to finite value
    # X[X['ndvi']< -1.0]['ndvi'] = -1.0
    # X[X['ndvi']> 1.0]['ndvi'] = 1.0
    # X[np.isfinite(X['ndvi']) == False]['ndvi'] = np.nan
    # define the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=random_state,
    )
    # evaluate the model
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    n_accuracy = cross_val_score(
        model, X, y, scoring="accuracy", cv=cv, n_jobs=-1, error_score="raise"
    )
    n_f1 = cross_val_score(
        model, X, y, scoring="f1", cv=cv, n_jobs=-1, error_score="raise"
    )
    n_balanced_accuracy = cross_val_score(
        model,
        X,
        y,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        error_score="raise",
    )
    # report performance
    plt.hist(n_f1)
    print("Repeat times:".format(), len(n_f1))
    print("F1-score: %.5f (%.5f)" % (n_f1.mean(), n_f1.std()))
    print(
        "Balanced Accuracy: %.5f (%.5f)"
        % (n_balanced_accuracy.mean(), n_balanced_accuracy.std())
    )
    print("Accuracy: %.5f (%.5f)" % (n_accuracy.mean(), n_accuracy.std()))

    # fit model with all observations
    model.fit(X, y)
    # save model
    joblib.dump(model, new_model_filepath)
    print(f"Model saved to {new_model_filepath}")
    # save accuracy
    scores = pd.DataFrame()
    scores["accuracy"] = n_accuracy
    scores["f1"] = n_f1
    scores["balanced_accuracy"] = n_balanced_accuracy
    scores.to_csv(new_model_score_filepath, index=False)
    print(f"Model scores saved to {new_model_score_filepath}")
    print("Total time used:".format(), round(time.process_time() - starttime, 1))

    return model
