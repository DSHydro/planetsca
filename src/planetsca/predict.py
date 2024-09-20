import glob
import os
from typing import List, Union

import joblib
import numpy as np
import onnx
import pandas as pd
import rasterio
from onnxruntime import InferenceSession
from sklearn.ensemble import RandomForestClassifier


def check_inputs(
    planet_path: Union[str, List[str]],
    model: Union[str, RandomForestClassifier],
    output_dirpath: str = "",
) -> int:
    """
    Check the inputs for the predict_sca function

    Parameters
    ----------
        planet_path: str or List[str]
            file path to a single PlanetScope surface reflectance (SR) image, a list of file paths, or path to a directory containing multiple SR images
        model: Union[str, RandomForestClassifier]
            file path to a model joblib file, or an sklearn.ensemble RandomForestClassifier model object
        output_dirpath: str
            the directory where output snow cover images will be stored

    Returns
    ----------
        file_list: List[str]
            a list of filepaths to PlanetScope surface reflectance (SR) images
        model: RandomForestClassifier
            an sklearn.ensemble RandomForestClassifier model object
        output_dirpath: str
            the directory where output snow cover images will be stored
    """

    # if output directory is not empty
    if output_dirpath != "":
        # check if the directory already exists
        if not os.path.exists(output_dirpath):
            # create the output directory if it does not already exist
            os.mkdir(output_dirpath)

    # check if planet_path is a list
    if isinstance(planet_path, list):
        # make sure that each file exists
        if all(os.path.isfile(filepath) for filepath in planet_path):
            # then the file list is what was provided in planet_path
            file_list = planet_path
    # otherwise planet_path should be a string
    elif isinstance(planet_path, str):
        # if planet_path is a directory, then find all images with 'SR' flag, meaning surface reflectance data
        if os.path.isdir(planet_path):
            file_list = glob.glob(planet_path + "/**/*SR*.tif", recursive=True)
        # otherwise we are working with a single planet image
        elif os.path.isfile(planet_path):
            file_list = [planet_path]

    # if provided with a filepath to a model file
    if isinstance(model, str) and os.path.isfile(model):
        # open the model
        print(f"Reading model from file: {model}")
        model = joblib.load(model)
    # otherwise "model" is already our RandomForestClassifier model

    return file_list, model, output_dirpath


def predict_sca(
    planet_path: Union[str, List[str]],
    model: Union[str, RandomForestClassifier],
    output_dirpath: str = "",
    nodata_flag: int = 9,
) -> Union[str, List[str]]:
    """
    This function predicts binary snow cover from PlanetScope satellite images using an ONNX random forest model

    Parameters
    ----------
        planet_path: str or List[str]
            file path to a single PlanetScope surface reflectance (SR) image, a list of file paths, or path to a directory containing multiple SR images
        model: Union[str, RandomForestClassifier]
            file path to a model joblib file, or an sklearn.ensemble RandomForestClassifier model object
        output_dirpath: str
            the directory where output snow cover images will be stored
        nodata_flag: int
            the value used to represent no data in the predicted snow cover image, default value is 9

    Returns
    ----------
        sca_image_paths: List[str]
            list of file paths to the SCA images produced
    """

    #
    file_list, model, output_dirpath = check_inputs(planet_path, model, output_dirpath)

    # make an empty list to populate with finished sca image filepaths
    sca_image_paths = []

    # open and apply the model to each image in the list
    for f in file_list:
        print("Start to predict:".format(), os.path.basename(f))

        with rasterio.open(f, "r") as ds:
            arr = ds.read()  # read all raster values
            if arr.shape[0] > 4:  # if we have more than 4 bands
                print(
                    "Input image has more than the expected 4 bands (blue, green, red, NIR). \
                This function will continue running using the first four bands in the input image."
                )
                # TODO: use UserWarning, warnings, or logging module to handle messages like this
                arr = arr[:4, :, :]  # use only the first four
                # TODO: add functionality for other cases where we have more than 4 bands (e.g. using NDVI or pseudo-NDSI)

        print("Image dimension:".format(), arr.shape)
        X_img = pd.DataFrame(arr.reshape([4, -1]).T)
        X_img.columns = ["blue", "green", "red", "nir"]

        X_img = X_img / 10000  # scale surface reflectance to 0-1

        X_img["nodata_flag"] = np.where(
            X_img["blue"] == 0, -1, 1
        )  # wherever blue band is zero, set to nodata value of -1

        # run model prediction
        y_img = model.predict(X_img.iloc[:, 0:4])

        out_img = pd.DataFrame()
        out_img["label"] = y_img
        out_img["nodata_flag"] = X_img["nodata_flag"]
        out_img["label"] = np.where(
            out_img["nodata_flag"] == -1, nodata_flag, out_img["label"]
        )  # where we set to -1, set to new nodata_flag value
        # Reshape our classification map
        img_prediction = out_img["label"].to_numpy().reshape(arr[0, :, :].shape)

        # save the resulting SCA image out as a geotiff
        file_out = os.path.join(
            output_dirpath, os.path.splitext(os.path.basename(f))[0] + "_SCA.tif"
        )
        print("Save SCA map to: ".format(), file_out)
        with rasterio.open(
            file_out,
            "w",
            driver="GTiff",
            transform=ds.transform,
            dtype=rasterio.uint8,
            count=1,
            crs=ds.crs,
            width=ds.width,
            height=ds.height,
            nodata=nodata_flag,
        ) as dst:
            dst.write(img_prediction, indexes=1, masked=True)

        sca_image_paths.append(file_out)

    return sca_image_paths


def predict_with_onnxruntime(
    model: onnx.onnx_ml_pb2.ModelProto, X: np.array
) -> np.array:
    """
    Run a prediction with an ONNX model

    Parameters
    ----------
        model: onnx.onnx_ml_pb2.ModelProto
            an onnx.onnx_ml_pb2.ModelProto model object
        X: np.array
            an array of input data of shape (n_samples, 4)

    Returns
    ----------
        predictions: np.array
            an array of predicted labels for snow (1) or no snow (0) of shape (n_samples, 4)
    """
    sess = InferenceSession(model.SerializeToString())
    input_name = sess.get_inputs()[0].name
    res = sess.run(None, {input_name: X.astype(np.float32)})
    predictions = res[0]
    return predictions


def check_inputs_onnx(
    planet_path: Union[str, List[str]],
    model: Union[str, onnx.onnx_ml_pb2.ModelProto],
    output_dirpath: str = "",
) -> int:
    """
    Check the inputs for the predict_sca_onnx function

    Parameters
    ----------
        planet_path: str or List[str]
            file path to a single PlanetScope surface reflectance (SR) image, a list of file paths, or path to a directory containing multiple SR images
        model: Union[str, onnx.onnx_ml_pb2.ModelProto]
            file path to a model onnx file, or an onnx.onnx_ml_pb2.ModelProto model object
        output_dirpath: str
            the directory where output snow cover images will be stored

    Returns
    ----------
        file_list: List[str]
            a list of filepaths to PlanetScope surface reflectance (SR) images
        model: onnx.onnx_ml_pb2.ModelProto
            an onnx.onnx_ml_pb2.ModelProto model object
        output_dirpath: str
            the directory where output snow cover images will be stored
    """

    # if output directory is not empty
    if output_dirpath != "":
        # check if the directory already exists
        if not os.path.exists(output_dirpath):
            # create the output directory if it does not already exist
            os.mkdir(output_dirpath)

    # check if planet_path is a list
    if isinstance(planet_path, list):
        # make sure that each file exists
        if all(os.path.isfile(filepath) for filepath in planet_path):
            # then the file list is what was provided in planet_path
            file_list = planet_path
    # otherwise planet_path should be a string
    elif isinstance(planet_path, str):
        # if planet_path is a directory, then find all images with 'SR' flag, meaning surface reflectance data
        if os.path.isdir(planet_path):
            file_list = glob.glob(planet_path + "/**/*SR*.tif", recursive=True)
        # otherwise we are working with a single planet image
        elif os.path.isfile(planet_path):
            file_list = [planet_path]

    # if provided with a filepath to a model file
    if isinstance(model, str) and os.path.isfile(model):
        # open the model
        print(f"Reading model from file: {model}")
        model = onnx.load(model)
    # otherwise "model" is already our RandomForestClassifier model

    return file_list, model, output_dirpath


def predict_sca_onnx(
    planet_path: Union[str, List[str]],
    model: Union[str, onnx.onnx_ml_pb2.ModelProto],
    output_dirpath: str = "",
    nodata_flag: int = 9,
) -> Union[str, List[str]]:
    """
    This function predicts binary snow cover from PlanetScope satellite images using an ONNX random forest model

    Parameters
    ----------
        planet_path: str or List[str]
            file path to a single PlanetScope surface reflectance (SR) image, a list of file paths, or path to a directory containing multiple SR images
        model: Union[str, onnx.onnx_ml_pb2.ModelProto]
            file path to a model onnx file, or an onnx.onnx_ml_pb2.ModelProto model object
        output_dirpath: str
            the directory where output snow cover images will be stored
        nodata_flag: int
            the value used to represent no data in the predicted snow cover image, default value is 9

    Returns
    ----------
        sca_image_paths: List[str]
            list of file paths to the SCA images produced
    """

    #
    file_list, model, output_dirpath = check_inputs_onnx(
        planet_path, model, output_dirpath
    )

    # make an empty list to populate with finished sca image filepaths
    sca_image_paths = []

    # open and apply the model to each image in the list
    for f in file_list:
        print("Start to predict:".format(), os.path.basename(f))

        with rasterio.open(f, "r") as ds:
            arr = ds.read()  # read all raster values
            if arr.shape[0] > 4:  # if we have more than 4 bands
                print(
                    "Input image has more than the expected 4 bands (blue, green, red, NIR). \
                This function will continue running using the first four bands in the input image."
                )
                # TODO: use UserWarning, warnings, or logging module to handle messages like this
                arr = arr[:4, :, :]  # use only the first four
                # TODO: add functionality for other cases where we have more than 4 bands (e.g. using NDVI or pseudo-NDSI)

        print("Image dimension:".format(), arr.shape)
        X_img = pd.DataFrame(arr.reshape([4, -1]).T)
        X_img.columns = ["blue", "green", "red", "nir"]

        X_img = X_img / 10000  # scale surface reflectance to 0-1

        X_img["nodata_flag"] = np.where(
            X_img["blue"] == 0, -1, 1
        )  # wherever blue band is zero, set to nodata value of -1

        # run model prediction with onnxruntime
        y_img = predict_with_onnxruntime(model, X_img.iloc[:, 0:4].to_numpy())

        out_img = pd.DataFrame()
        out_img["label"] = y_img
        out_img["nodata_flag"] = X_img["nodata_flag"]
        out_img["label"] = np.where(
            out_img["nodata_flag"] == -1, nodata_flag, out_img["label"]
        )  # where we set to -1, set to new nodata_flag value
        # Reshape our classification map
        img_prediction = out_img["label"].to_numpy().reshape(arr[0, :, :].shape)

        # save the resulting SCA image out as a geotiff
        file_out = os.path.join(
            output_dirpath, os.path.splitext(os.path.basename(f))[0] + "_SCA.tif"
        )
        print("Save SCA map to: ".format(), file_out)
        with rasterio.open(
            file_out,
            "w",
            driver="GTiff",
            transform=ds.transform,
            dtype=rasterio.uint8,
            count=1,
            crs=ds.crs,
            width=ds.width,
            height=ds.height,
            nodata=nodata_flag,
        ) as dst:
            dst.write(img_prediction, indexes=1, masked=True)

        sca_image_paths.append(file_out)

    return sca_image_paths
