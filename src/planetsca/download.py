import json
import os
import pathlib
import time
from typing import List, Optional

import joblib
import onnx
import requests
from huggingface_hub import hf_hub_download
from requests.auth import HTTPBasicAuth
from sklearn.ensemble import RandomForestClassifier

from planetsca import search


def order(
    api_key: str,
    id_list: List[str],
    filter: dict,
    item_type: str = "PSScene",
    bundle_type: str = "analytic_sr_udm2",
) -> str:
    """
    Builds payload for Planet API and submits an order.

    Parameters
    ----------
        api_key: str
            Planet API key
        id_list: List[str]
            Item id that contains date and location information
        filter: dict
            Dictionary containing data filter information
        item_type: str
            Class of spacecraft and/or processing level of an item, defaults to PSScene. See https://developers.planet.com/docs/apis/data/items-assets/
        bundle_type: str
            Groups of assets for an item and contain metadata, defaults to analytic_sr_udm2. See https://developers.planet.com/apis/orders/product-bundles-reference/#surface-reflectance-4b

    Returns
    -------
        order_url: str
            URL from which to download image from Planet API
    """

    # build payload
    payload = build_payload(
        id_list,
        search.get_filter(filter, "GeometryFilter")["config"]["coordinates"],
        item_type,
        bundle_type,
    )

    # submit payload and get order url
    order_url = order_now(api_key, payload)

    return order_url


def build_payload(
    item_ids: List[str], aoi_coordinates: List[float], item_type: str, bundle_type: str
) -> dict:
    """
    Helper function building payload for the Planet API

    Parameters
    ----------
        item_ids: List[str]
            Item id that contains date and location information
        aoi_coordinates: List[float]
            Area of interest coordinates
        item_type: str
            Class of spacecraft and/or processing level of an item, defaults to PSScene. See https://developers.planet.com/docs/apis/data/items-assets/
        bundle_type: str
            Groups of assets for an item and contain metadata, defaults to analytic_sr_udm2. See https://developers.planet.com/apis/orders/product-bundles-reference/#surface-reflectance-4b

    Returns
    -------
        payload: dict
            Dictionary containing all necessary information for the Planet API
    """

    payload = {
        "name": item_ids[0],  # use the first item id as a name for this order payload
        "source_type": "scenes",
        "products": [
            {
                "item_ids": item_ids,
                "item_type": item_type,
                "product_bundle": bundle_type,
            }
        ],
        "tools": [
            {"clip": {"aoi": {"type": "Polygon", "coordinates": aoi_coordinates}}}
        ],
    }
    return payload


def order_now(api_key, payload):
    """
    Helper function for ordering data from Planet

    Parameters
    ----------
        api_key: str
            Planet API key
        payload: dict
            Dictionary containing all necessary information for the Planet API

    Returns
    ----------
        order_url: str
            URL from which to download image from Planet API
    """

    orders_url = "https://api.planet.com/compute/ops/orders/v2"
    response = requests.post(
        orders_url,
        data=json.dumps(payload),
        auth=HTTPBasicAuth(api_key, ""),
        headers={"Content-Type": "application/json"},
    )
    # print(response)

    if response.status_code == 202:
        order_id = response.json()["id"]
        url = f"https://api.planet.com/compute/ops/orders/v2/{order_id}"
        # feature_check = requests.get(url, auth=(PLANET_API_KEY, ""))
        feature_check = requests.get(url, auth=HTTPBasicAuth(api_key, ""))
        if feature_check.status_code == 200:
            print(
                f"Submitted a total of {len(feature_check.json()['products'][0]['item_ids'])} image ids: accepted a total of {len(feature_check.json()['products'][0]['item_ids'])} ids"
            )
            order_url = f"https://api.planet.com/compute/ops/orders/v2/{order_id}"
            print(f"Order URL: {order_url}")
            return order_url
    else:
        print(f"Failed with Exception code : {response.status_code}")
        return response


def download(
    api_key: str, order_url: str, out_dirpath: str, overwrite: bool = False
) -> None:
    """
    Helper function for downloading the ordered data from Planet, makes a download request every 60 seconds until data is ready to download

    Parameters
    ----------
        api_key: str
            Planet API key
        order_url: str
            Order urls created from prepare_submit_orders()
        out_dirpath: str
            Path to output directory
        overwrite: bool
            Whether or not to overwrite existing files, defaults to False

    Returns
    ----------
        None
    """

    print("Attempting to download")
    request_fufilled = True
    counter = 1
    while request_fufilled:
        r = requests.get(order_url, auth=HTTPBasicAuth(api_key, ""))
        try:
            if r.status_code == 200:
                response = r.json()
                results = response["_links"]["results"]
                results_urls = [r["location"] for r in results]
                results_names = [r["name"] for r in results]
                print("{} items to download".format(len(results_urls)))

                for url, name in zip(results_urls, results_names):
                    path = pathlib.Path(os.path.join(out_dirpath, name))

                    if overwrite or not path.exists():
                        print("downloading {} to {}".format(name, path))
                        r = requests.get(url, allow_redirects=True)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        open(path, "wb").write(r.content)
                    else:
                        print("{} already exists, skipping {}".format(path, name))
            else:
                print(f"Failed with response {r.status_code}")
            request_fufilled = False
        # Data isn't ready yet
        except Exception:
            print("data not ready yet, this was attempt number {}".format(counter))
            print("will automatically try again in 60 seconds")
            # TODO: use warning or logging to handle these cases
            counter += 1
        r.close()
        time.sleep(60)
    print("Completed downloads")
    return None


def retrieve_dataset(filename: str, out_dirpath: Optional[str] = ".") -> str:
    """
    Downloads sample datasets for PlanetSCA model from Hugging Face

    Parameters
    ----------
        filename: str
            Filename of one of the PlanetSCA files on Hugging Face, see list of files here: https://huggingface.co/datasets/geo-smart/planetsca_datasets/tree/main
        out_dirpath: Optional[str]
            Path to directory to save file

    Returns
    ---------
        filepath: str
            Path to local file
    """

    filepath = hf_hub_download(
        repo_id="geo-smart/planetsca_datasets",
        repo_type="dataset",
        filename=filename,
        local_dir=out_dirpath,
    )

    return filepath


def retrieve_model(out_dirpath: Optional[str] = None) -> RandomForestClassifier:
    """
    Downloads pre-trained PlanetSCA model from Hugging Face

    Parameters:
        out_dirpath: Optional[str]
            Path to directory to save pre-trained model file

    Returns
    ----------
        model: RandomForestClassifier
            The trained PlanetSCA model
    """

    # download model file from Hugging Face
    filepath = hf_hub_download(
        repo_id="geo-smart/planetsca_models",
        filename="random_forest_20240116_binary_174K.joblib",
        local_dir=out_dirpath,
    )

    # open the model that was just downloaded
    model = joblib.load(filepath)

    return model


def retrieve_model_onnx(
    out_dirpath: Optional[str] = None,
) -> onnx.onnx_ml_pb2.ModelProto:
    """
    Downloads pre-trained PlanetSCA model from Hugging Face (ONNX format)

    Parameters:
        out_dirpath: Optional[str]
            Path to directory to save pre-trained model file

    Returns
    ----------
        model: onnx.onnx_ml_pb2.ModelProto
            The trained PlanetSCA model (ONNX format)
    """

    # download model file from Hugging Face
    filepath = hf_hub_download(
        repo_id="geo-smart/planetsca_models",
        filename="random_forest_20240116_binary_174K.onnx",
        local_dir=out_dirpath,
    )

    # open the model that was just downloaded
    model = onnx.load(filepath)

    return model
