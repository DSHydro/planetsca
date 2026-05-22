import os
from datetime import datetime
from typing import List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import xarray
from rioxarray.exceptions import NoDataInBounds


def calc_rgb(ds) -> Tuple:
    """
    Extract and normalize RGB and NIR bands from a 4-band PlanetScope xarray Dataset.

    Parameters
    ----------
        ds: xarray.DataArray
            4-band PlanetScope image with bands ordered blue, green, red, NIR

    Returns
    -------
        red_band: np.ndarray
            Raw red band values
        green_band: np.ndarray
            Raw green band values
        blue_band: np.ndarray
            Raw blue band values
        nir_band: np.ndarray
            Raw NIR band values
        rgb_image: np.ndarray
            Normalized RGB image of shape (y, x, 3)
    """
    blue_band = ds.isel(band=0)
    green_band = ds.isel(band=1)
    red_band = ds.isel(band=2)
    nir_band = ds.isel(band=3)

    maxval = green_band.max().values
    minval = green_band.min().values
    red_norm = (red_band - minval) / (maxval - minval)
    green_norm = (green_band - minval) / (maxval - minval)
    blue_norm = (blue_band - minval) / (maxval - minval)
    green_norm = green_norm.where(red_norm <= 1, 1)
    blue_norm = blue_norm.where(red_norm <= 1, 1)
    red_norm = red_norm.where(red_norm <= 1, 1)

    red_band = red_band.values
    green_band = green_band.values
    blue_band = blue_band.values
    nir_band = nir_band.values

    rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=-1)
    return red_band, green_band, blue_band, nir_band, rgb_image


def dem_info(DEM: str, basin: gpd.GeoDataFrame, crs: str = "EPSG:32611") -> Tuple:
    """
    Open and clip a DEM to a basin geometry and return summary elevation statistics.

    Parameters
    ----------
        DEM: str
            File path to a DEM GeoTIFF
        basin: geopandas.GeoDataFrame
            Basin boundary used to clip the DEM
        crs: str
            Target CRS for reprojection, defaults to EPSG:32611

    Returns
    -------
        dem: xarray.DataArray
            Clipped DEM raster
        dem_mean: float
            Mean elevation within the basin
        dem_max: float
            Maximum elevation within the basin
        dem_min: float
            Minimum elevation within the basin
    """
    dem = (
        rioxarray.open_rasterio(DEM, all_touched=False, drop=True, masked=True)
        .rio.reproject(crs, inplace=True)
        .rio.clip(basin.geometry.values, all_touched=False, drop=True)
    )
    dem.values = np.where(dem.values == dem.rio.nodata, np.nan, dem.values)
    dem.rio.write_nodata(np.nan, inplace=True)
    dem_mean = np.nanmean(dem.values)
    dem_max = np.nanmax(dem.values)
    dem_min = np.nanmin(dem.values)
    return dem, dem_mean, dem_max, dem_min


def create_binary_chm(chm: str, basin: gpd.GeoDataFrame) -> Tuple:
    """
    Create a binary canopy height mask from a canopy height model (CHM) raster.
    Pixels with canopy height < 2m are classified as open (1); >= 2m as forest (0).

    Parameters
    ----------
        chm: str
            File path to the canopy height model GeoTIFF
        basin: geopandas.GeoDataFrame
            Basin boundary used to clip the CHM

    Returns
    -------
        chm_mask: xarray.DataArray
            Binary mask raster (1 = open, 0 = forest)
        mean: float
            Mean canopy height within the basin
        max: float
            Maximum canopy height within the basin
        fcan: float
            Forest canopy fraction (percentage of pixels classified as forest)
    """
    chm_mask = rioxarray.open_rasterio(
        chm, all_touched=False, drop=True, masked=True
    ).rio.clip(basin.geometry.values, all_touched=False, drop=True)
    mean = np.nanmean(chm_mask.values)
    max = np.nanmax(chm_mask.values)
    chm_mask.values = np.where(chm_mask < 2, 1, chm_mask)
    chm_mask.values = np.where(chm_mask >= 2, 0, chm_mask)
    one_cnt = (chm_mask.values == 1).sum()
    zero_cnt = (chm_mask.values == 0).sum()
    fcan = zero_cnt / (one_cnt + zero_cnt) * 100
    return chm_mask, mean, max, fcan


def extract_dates(
    aso_tif: List[str], ps_sca_tif: List[str]
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Extract datetime objects from ASO and PlanetScope SCA filenames.

    Parameters
    ----------
        aso_tif: List[str]
            List of file paths to ASO GeoTIFF files
        ps_sca_tif: List[str]
            List of file paths to PlanetScope SCA GeoTIFF files

    Returns
    -------
        aso_dates: List[Tuple[datetime, str]]
            List of (datetime, filepath) tuples for ASO files
        ps_dates: List[Tuple[datetime, str]]
            List of (datetime, filepath) tuples for PlanetScope SCA files
    """
    aso_dates = []
    ps_dates = []
    for aso_file in aso_tif:
        filename = os.path.basename(aso_file)
        date_str = filename.split("_")[2]
        year = date_str[:4]
        month = date_str[4:7]
        day = date_str[7:]
        if "-" in day:
            day = day.split("-")[0]
        date_obj = datetime.strptime(f"{year} {month} {day}", "%Y %b %d")
        aso_dates.append((date_obj, aso_file))

    for ps_file in ps_sca_tif:
        filename = os.path.basename(ps_file)
        date_str = filename.split("_")[0]
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        ps_dates.append((date_obj, ps_file))

    return aso_dates, ps_dates


def create_ps_df(
    ps_sca_tif: List[str],
    basin: gpd.GeoDataFrame,
    chm_mask,
    name: str,
    rgi_path: Optional[str] = None,
    wbd_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a per-date summary DataFrame of PlanetScope SCA pixel counts within a basin.
    Rows with unobserved fraction > 2% are excluded.

    Parameters
    ----------
        ps_sca_tif: List[str]
            List of file paths to PlanetScope SCA GeoTIFF files
        basin: geopandas.GeoDataFrame
            Basin boundary GeoDataFrame
        chm_mask: xarray.DataArray
            Binary canopy height mask (unused directly; reserved for future CHM filtering)
        name: str
            Basin name label added to the output DataFrame
        rgi_path: Optional[str]
            File path to the RGI glacier shapefile. If provided, glacier pixels are masked out.
        wbd_path: Optional[str]
            File path to the NHD waterbody shapefile. If provided, waterbody pixels are masked out.

    Returns
    -------
        ps_sca_year: pd.DataFrame
            DataFrame with columns: fn, date, basin, snow, unobs, nosnow, fsca_obs, f_unobs, f_nosnow
    """
    rgi_mask = gpd.read_file(rgi_path).to_crs("EPSG:32611") if rgi_path else None
    wbd_mask = gpd.read_file(wbd_path).to_crs("EPSG:32611") if wbd_path else None

    ps_sca_df = []
    for file in ps_sca_tif:
        filename = os.path.basename(file)
        date = filename.split(".")[0].split("_")[0]

        ps_sca = rioxarray.open_rasterio(file, masked=True)
        raster_crs = ps_sca.rio.crs
        basin_clip = basin.to_crs(raster_crs).geometry.values
        ps_sca = ps_sca.rio.clip(basin_clip, crs=basin.crs, drop=True)
        ps_sca.values = np.where(np.isnan(ps_sca), 0, ps_sca)

        if rgi_mask is not None:
            ps_sca = ps_sca.rio.clip(rgi_mask.geometry.values, invert=True)
        if wbd_mask is not None:
            ps_sca = ps_sca.rio.clip(wbd_mask.geometry.values, invert=True)

        counts = np.unique(ps_sca.values, return_counts=True)
        count_dict = dict(zip(counts[0], counts[1]))
        zero_cnt = count_dict.get(0, 0)
        one_cnt = count_dict.get(1, 0)
        two_cnt = count_dict.get(2, 0)
        total_cnt = zero_cnt + one_cnt + two_cnt

        ps_sca_df.append(
            {
                "fn": filename.split(".")[0],
                "date": date,
                "basin": name,
                "snow": one_cnt,
                "unobs": two_cnt,
                "nosnow": zero_cnt,
                "fsca_obs": one_cnt / total_cnt if total_cnt else 0,
                "f_unobs": two_cnt / total_cnt if total_cnt else 0,
                "f_nosnow": zero_cnt / total_cnt if total_cnt else 0,
            }
        )

    ps_sca_year = pd.DataFrame(ps_sca_df)
    ps_sca_year = ps_sca_year[ps_sca_year["f_unobs"] < 0.02]
    ps_sca_year = ps_sca_year.sort_values("date").reset_index(drop=True)
    ps_sca_year["date"] = pd.to_datetime(ps_sca_year["date"], format="%Y%m%d")
    return ps_sca_year


def create_aso_df(
    aso_tif: List[str],
    basin: gpd.GeoDataFrame,
    name: str,
    chm_mask,
    rgi_path: Optional[str] = None,
    wbd_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a per-date summary DataFrame of ASO SCA pixel counts within a basin.

    Parameters
    ----------
        aso_tif: List[str]
            List of file paths to ASO SWE GeoTIFF files
        basin: geopandas.GeoDataFrame
            Basin boundary GeoDataFrame
        name: str
            Basin name label added to the output DataFrame
        chm_mask: xarray.DataArray
            Binary canopy height mask (reserved for future CHM filtering)
        rgi_path: Optional[str]
            File path to the RGI glacier shapefile. If provided, glacier pixels are masked out.
        wbd_path: Optional[str]
            File path to the NHD waterbody shapefile. If provided, waterbody pixels are masked out.

    Returns
    -------
        aso_sca_year: pd.DataFrame
            DataFrame with columns: fn, basin, date, snow, unobs, nosnow, fsca_obs, f_unobs, f_nosnow
    """
    rgi_mask = gpd.read_file(rgi_path).to_crs("EPSG:32611") if rgi_path else None
    wbd_mask = gpd.read_file(wbd_path).to_crs("EPSG:32611") if wbd_path else None

    aso_sca_df = []
    for file in aso_tif:
        filename = os.path.basename(file)
        date_str = filename.split("_")[2]
        year = date_str[:4]
        month = date_str[4:7]
        day = date_str[7:]
        if "-" in day:
            day = day.split("-")[0]
        date_obj = datetime.strptime(f"{year} {month} {day}", "%Y %b %d")
        date = date_obj.strftime("%Y-%m-%d")

        rio_aso = rioxarray.open_rasterio(file, mask_and_scale=True, crs="EPSG:32611")
        rio_aso = rio_aso.rio.clip(basin.geometry.values, crs="EPSG:32611", drop=True)
        rio_proj = rio_aso.rio.reproject("EPSG:32611", nodata=np.nan)
        rio_proj.values = np.where(np.isnan(rio_proj), 2, rio_proj.values)
        rio_proj.values = np.where(rio_proj.values > 0.1, 1, rio_proj.values)
        rio_proj.values = np.where(rio_proj.values <= 0.1, 0, rio_proj.values)
        rio_proj = rio_proj.rio.clip(basin.geometry.values, crs=basin.crs, drop=True)

        if rgi_mask is not None:
            rio_proj = rio_proj.rio.clip(rgi_mask.geometry, invert=True)
        if wbd_mask is not None:
            rio_proj = rio_proj.rio.clip(wbd_mask.geometry, invert=True)

        zero_cnt = (rio_proj.values == 0).sum()
        one_cnt = (rio_proj.values == 1).sum()
        two_cnt = (rio_proj.values == 2).sum()
        cnt = zero_cnt + one_cnt + two_cnt

        aso_sca_df.append(
            {
                "fn": filename.split(".")[0],
                "basin": name,
                "date": date,
                "snow": one_cnt,
                "unobs": two_cnt,
                "nosnow": zero_cnt,
                "fsca_obs": one_cnt / cnt if cnt else 0,
                "f_unobs": two_cnt / cnt if cnt else 0,
                "f_nosnow": zero_cnt / cnt if cnt else 0,
            }
        )

    aso_sca_year = pd.DataFrame(aso_sca_df)
    aso_sca_year["date"] = pd.to_datetime(aso_sca_year["date"], format="%Y-%m-%d")
    aso_sca_year = aso_sca_year.sort_values(by="date")
    return aso_sca_year


def closest_date(ps_dates: List[Tuple], aso_date: datetime) -> datetime:
    """
    Find the PlanetScope date closest to a given ASO date.

    Parameters
    ----------
        ps_dates: List[Tuple[datetime, str]]
            List of (datetime, filepath) tuples for PlanetScope SCA files
        aso_date: datetime
            The ASO acquisition date to match against

    Returns
    -------
        closest: datetime
            The PlanetScope date closest to aso_date
    """
    ps_date_list = [date for date, _ in ps_dates]
    sorted_list = sorted(ps_date_list)
    previous_date = sorted_list[-1]
    for date in sorted_list:
        if date >= aso_date:
            if abs((date - aso_date).days) < abs((previous_date - aso_date).days):
                return date
            else:
                return previous_date
        previous_date = date
    return sorted_list[-1]


def closest_dates(ps_dates: List[Tuple], aso_date: datetime) -> List[datetime]:
    """
    Find all PlanetScope dates equally closest to a given ASO date.

    Parameters
    ----------
        ps_dates: List[Tuple[datetime, str]]
            List of (datetime, filepath) tuples for PlanetScope SCA files
        aso_date: datetime
            The ASO acquisition date to match against

    Returns
    -------
        closest: List[datetime]
            List of PlanetScope dates with the minimum distance to aso_date
    """
    ps_date_list = [date for date, _ in ps_dates]
    sorted_list = sorted(ps_date_list)
    closest_distance = None
    result = []
    for date in sorted_list:
        distance = abs((date - aso_date).days)
        if closest_distance is None or distance < closest_distance:
            closest_distance = distance
            result = [date]
        elif distance == closest_distance:
            result.append(date)
    return result


def plot_model_comparison(
    model1_files: List[str],
    model2_files: List[str],
    ps_dates: List[Tuple],
    aso_dates: List[Tuple],
    basin: gpd.GeoDataFrame,
    rgi_path: Optional[str] = None,
    wbd_path: Optional[str] = None,
) -> None:
    """
    Plot side-by-side SCA maps for two model predictions and the corresponding ASO reference,
    matched to each ASO acquisition date by closest PlanetScope date.

    Parameters
    ----------
        model1_files: List[str]
            File paths to the first model's SCA GeoTIFF predictions
        model2_files: List[str]
            File paths to the second model's SCA GeoTIFF predictions
        ps_dates: List[Tuple[datetime, str]]
            List of (datetime, filepath) tuples for PlanetScope SCA files
        aso_dates: List[Tuple[datetime, str]]
            List of (datetime, filepath) tuples for ASO files
        basin: geopandas.GeoDataFrame
            Basin boundary GeoDataFrame used for clipping
        rgi_path: Optional[str]
            File path to the RGI glacier shapefile. If provided, glacier pixels are masked out.
        wbd_path: Optional[str]
            File path to the NHD waterbody shapefile. If provided, waterbody pixels are masked out.

    Returns
    -------
        None
    """
    rgi_mask = gpd.read_file(rgi_path).to_crs("EPSG:32611") if rgi_path else None
    wbd_mask = gpd.read_file(wbd_path).to_crs("EPSG:32611") if wbd_path else None

    for aso_date, aso_file in sorted(aso_dates):
        closest = closest_date(ps_dates, aso_date)
        closest_str = closest.strftime("%Y%m%d")

        model1_file = next(
            (f for f in model1_files if closest_str in os.path.basename(f)), None
        )
        model2_file = next(
            (f for f in model2_files if closest_str in os.path.basename(f)), None
        )

        if model1_file and model2_file:

            def _load_clip(filepath):
                sca = rioxarray.open_rasterio(
                    filepath, all_touched=False, drop=True, masked=True
                )
                sca.values = np.where(np.isnan(sca.values), 0, sca.values)
                sca = sca.rio.clip(basin.geometry.values, crs=basin.crs, drop=True)
                if rgi_mask is not None:
                    sca = sca.rio.clip(rgi_mask.geometry, invert=True)
                if wbd_mask is not None:
                    sca = sca.rio.clip(wbd_mask.geometry, invert=True)
                return np.squeeze(sca.values)

            m1_sca = _load_clip(model1_file)
            m2_sca = _load_clip(model2_file)
            aso_sca = _load_clip(aso_file)

            fig, ax = plt.subplots(1, 3, figsize=(14, 6))
            ax[0].imshow(m1_sca, vmin=0, vmax=2, interpolation="none")
            ax[0].set_title("V1")
            ax[1].imshow(m2_sca, vmin=0, vmax=2, interpolation="none")
            ax[1].set_title("V2")
            ax[2].imshow(aso_sca, vmin=0, vmax=2, interpolation="none")
            ax[2].set_title("ASO")
            plt.suptitle(
                f"Snow Map for ASO Date: {aso_date.strftime('%Y-%m-%d')}, "
                f"Closest Model Date: {closest_str}"
            )
            plt.tight_layout()
            plt.show()
        else:
            print(f"No matching files found for the closest date {closest_str}")


def validation_tif_4class(
    aso_tif: str,
    chm_tif,
    basin: gpd.GeoDataFrame,
    output_tif: str,
) -> None:
    """
    Create a 4-class validation raster by combining a binary ASO snow map and a binary
    canopy mask: 0 = no snow/no canopy, 1 = no snow/canopy, 2 = snow/canopy, 3 = snow/no canopy.

    Parameters
    ----------
        aso_tif: str
            File path to an ASO snow cover GeoTIFF (0 = no snow, 1 = snow)
        chm_tif: xarray.DataArray
            Binary canopy height mask (0 = forest, 1 = open), aligned to aso_tif
        basin: geopandas.GeoDataFrame
            Basin boundary GeoDataFrame used to clip the ASO raster
        output_tif: str
            File path for the output 4-class GeoTIFF

    Returns
    -------
        None
    """
    aso_binary = (
        rioxarray.open_rasterio(aso_tif, mask_and_scale=True)
        .rio.clip(basin.geometry.values, crs=basin.crs, drop=True)
        .squeeze()
    )
    chm_binary = chm_tif.squeeze()

    assert (
        aso_binary.shape == chm_binary.shape
    ), "ASO and CHM rasters must have the same shape"
    assert (
        aso_binary.rio.crs == chm_binary.rio.crs
    ), "ASO and CHM rasters must have the same CRS"
    assert (
        aso_binary.rio.transform() == chm_binary.rio.transform()
    ), "ASO and CHM rasters must have the same transform"

    combined = np.full(aso_binary.shape, np.nan, dtype=np.float32)
    combined[(aso_binary == 0) & (chm_binary == 1)] = 0
    combined[(aso_binary == 0) & (chm_binary == 0)] = 1
    combined[(aso_binary == 1) & (chm_binary == 0)] = 2
    combined[(aso_binary == 1) & (chm_binary == 1)] = 3

    combined_da = xarray.DataArray(
        combined, coords=aso_binary.coords, dims=aso_binary.dims
    )
    combined_da.rio.write_nodata(np.nan, inplace=True)
    combined_da.rio.to_raster(output_tif, driver="GTiff")


def validation_tif_binary(
    aso_tif: List[str],
    basin: gpd.GeoDataFrame,
    year: str,
    name: str,
    output_dir: str,
) -> None:
    """
    Convert ASO SWE rasters to binary snow/no-snow GeoTIFFs for a given year and basin.
    Pixels with SWE > 0.1 are classified as snow (1); others as no snow (0).

    Parameters
    ----------
        aso_tif: List[str]
            List of file paths to ASO SWE GeoTIFF files
        basin: geopandas.GeoDataFrame
            Basin boundary GeoDataFrame used to clip the ASO raster
        year: str
            Year string used to filter filenames (e.g. '2022')
        name: str
            Basin name used as a subdirectory within output_dir
        output_dir: str
            Root directory where binary GeoTIFFs will be saved

    Returns
    -------
        None
    """
    for file in aso_tif:
        if year in file:
            filename = file.split("/")[-1].split(".")[0]
            print(f"Processing file: {filename}")
            try:
                aso = rioxarray.open_rasterio(file, mask_and_scale=True)
                aso = aso.rio.clip(basin.geometry.values, crs=basin.crs, drop=True)
                aso = aso.rio.reproject("EPSG:32611", nodata=np.nan)
                aso.values = np.where(aso.values > 0.1, 1, aso.values)
                aso.values = np.where(aso.values <= 0.1, 0, aso.values)
                output_path = os.path.join(output_dir, name, f"{filename}_binary.tif")
                aso.rio.to_raster(output_path)
                print(f"Saved binary TIFF to: {output_path}")
            except NoDataInBounds:
                print(f"No data found in bounds for file: {filename}")


def create_validation_dataset(
    ps_tif: str,
    aso_tif: str,
    basin: gpd.GeoDataFrame,
    rgi_path: Optional[str] = None,
    wbd_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a pixel-level DataFrame of PlanetScope predictions paired with ASO observations
    for a single date, suitable for computing evaluation metrics.

    Parameters
    ----------
        ps_tif: str
            File path to a PlanetScope SCA GeoTIFF (predicted)
        aso_tif: str
            File path to a binary ASO snow cover GeoTIFF (observed)
        basin: geopandas.GeoDataFrame
            Basin boundary GeoDataFrame used to clip both rasters
        rgi_path: Optional[str]
            File path to the RGI glacier shapefile. If provided, glacier pixels are masked out.
        wbd_path: Optional[str]
            File path to the NHD waterbody shapefile. If provided, waterbody pixels are masked out.

    Returns
    -------
        df: pd.DataFrame
            DataFrame with columns 'x', 'y', 'obs' (ASO), and 'predict' (PlanetScope)
    """
    rgi_mask = gpd.read_file(rgi_path).to_crs("EPSG:32611") if rgi_path else None
    wbd_mask = gpd.read_file(wbd_path).to_crs("EPSG:32611") if wbd_path else None

    def _load_and_clip(filepath, drop=True):
        with rioxarray.open_rasterio(filepath, masked=True) as ds:
            ds.values = np.where(np.isnan(ds.values), 0, ds.values)
            ds_clipped = ds.rio.clip(basin.geometry, drop=drop)
            if rgi_mask is not None:
                ds_clipped = ds_clipped.rio.clip(rgi_mask.geometry, invert=True)
            if wbd_mask is not None:
                ds_clipped = ds_clipped.rio.clip(wbd_mask.geometry, invert=True)
            height, width = ds_clipped.shape[1], ds_clipped.shape[2]
            points = [
                (x, y, ds_clipped.values[0, y, x])
                for y in range(height)
                for x in range(width)
            ]
        return points

    ps_points = _load_and_clip(ps_tif, drop=True)
    aso_points = _load_and_clip(aso_tif, drop=False)

    ps_df = pd.DataFrame(ps_points, columns=["x", "y", "predict"])
    aso_df = pd.DataFrame(aso_points, columns=["x", "y", "obs"])
    return pd.merge(aso_df, ps_df, on=["x", "y"])


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate binary classification evaluation metrics from a prediction-observation DataFrame.

    Parameters
    ----------
        df: pd.DataFrame
            DataFrame with columns 'predict' and 'obs', each containing binary snow labels (0 or 1)

    Returns
    -------
        out: pd.DataFrame
            Single-row DataFrame with columns: precision, recall, f1, sensitivity, specificity,
            balanced_accuracy, accuracy, kappa, TP, TN, FP, FN
    """
    from sklearn.metrics import cohen_kappa_score

    empty = pd.DataFrame(
        data={
            "precision": [0],
            "recall": [0],
            "f1": [0],
            "sensitivity": [0],
            "specificity": [0],
            "balanced_accuracy": [0],
            "accuracy": [0],
            "kappa": [0],
            "TP": [0],
            "TN": [0],
            "FP": [0],
            "FN": [0],
        }
    )

    correct = df[df.predict == df.obs]
    TP = len(correct[correct.predict == 1])
    TN = len(correct[correct.predict == 0])
    incorrect = df[df.predict != df.obs]
    FN = len(incorrect[incorrect.predict == 0])
    FP = len(incorrect[incorrect.predict == 1])

    if (TP + FP) == 0 or (TP + FN) == 0 or (TN + FP) == 0:
        return empty

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    sensitivity = recall
    specificity = TN / (TN + FP)
    balanced_accuracy = (sensitivity + specificity) / 2
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    kappa = cohen_kappa_score(df.predict, df.obs)

    return pd.DataFrame(
        data={
            "precision": [precision],
            "recall": [recall],
            "f1": [f1],
            "sensitivity": [sensitivity],
            "specificity": [specificity],
            "balanced_accuracy": [balanced_accuracy],
            "accuracy": [accuracy],
            "kappa": [kappa],
            "TP": [TP],
            "TN": [TN],
            "FP": [FP],
            "FN": [FN],
        }
    )
