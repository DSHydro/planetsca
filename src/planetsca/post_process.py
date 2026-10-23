import datetime
import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import rasterio as rio
from scipy.ndimage import median_filter
from scipy.signal import convolve2d


def fill_nodata(stacked_data: np.ndarray) -> np.ndarray:
    """
    Fill nodata pixels (value 2) using temporal carry-forward from the previous timestep.
    At the first timestep, nodata pixels are set to 1 (snow present).

    Parameters
    ----------
        stacked_data: np.ndarray
            3D array of shape (time, y, x) with values 0 (no snow), 1 (snow), or 2 (nodata)

    Returns
    -------
        stacked_data: np.ndarray
            3D array with nodata pixels filled
    """
    t, y, x = stacked_data.shape
    for tt in np.arange(t):
        tt_date = stacked_data[tt, :, :]
        if tt == 0:
            tt_date[tt_date == 2] = 1
        else:
            tt_ref = stacked_data[tt - 1, :, :]
            tt_date[tt_date == 2] = tt_ref[tt_date == 2]
        stacked_data[tt, :, :] = tt_date
    return stacked_data


def calculate_snow_cover_percentage(stacked_data: np.ndarray) -> np.ndarray:
    """
    Calculate the fractional snow cover percentage for each timestep.

    Parameters
    ----------
        stacked_data: np.ndarray
            3D array of shape (time, y, x) with snow (1) and no-snow (0) values

    Returns
    -------
        snow_cover_percentage: np.ndarray
            1D array of snow cover percentages per timestep
    """
    snow_count = np.nansum(stacked_data == 1, axis=(1, 2))
    total_count = np.sum(~np.isnan(stacked_data), axis=(1, 2))
    return (snow_count / total_count) * 100


def create_circular_kernel(radius: int) -> np.ndarray:
    """
    Create a circular binary kernel of a given radius for spatial convolution.

    Parameters
    ----------
        radius: int
            Radius of the circular kernel in pixels

    Returns
    -------
        kernel: np.ndarray
            2D binary array of shape (2*radius+1, 2*radius+1)
    """
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center = radius
    for x in range(size):
        for y in range(size):
            if np.sqrt((x - center) ** 2 + (y - center) ** 2) <= radius:
                kernel[x, y] = 1
    return kernel


def masked_convolution(
    stacked_data: np.ndarray,
    mask: np.ndarray,
    kernel: np.ndarray,
    threshold_percent: float,
) -> np.ndarray:
    """
    Apply a neighborhood-based snow infilling to forested pixels (mask == 0).
    A no-snow forested pixel is reclassified as snow if enough neighboring pixels
    are snow-covered, based on a threshold percentage of the kernel area.

    Parameters
    ----------
        stacked_data: np.ndarray
            3D array of shape (time, y, x) with snow (1) and no-snow (0) values
        mask: np.ndarray
            2D canopy height mask where 0 = forest, 1 = open
        kernel: np.ndarray
            2D convolution kernel (e.g. from create_circular_kernel)
        threshold_percent: float
            Fraction of kernel pixels that must be snow to reclassify a forested pixel

    Returns
    -------
        convolved_stack: np.ndarray
            3D array with forested no-snow pixels infilled where appropriate
    """
    time, x, y = stacked_data.shape
    convolved_stack = np.zeros_like(stacked_data)
    for t in range(time):
        valid_mask = mask == 0
        convolved = convolve2d(stacked_data[t], kernel, mode="same", boundary="symm")
        threshold = int(threshold_percent * np.sum(kernel == 1))
        adjusted_snow_image = np.where(convolved >= threshold, 1, 0)
        convolved_stack[t] = np.copy(stacked_data[t])
        combined_mask = valid_mask & (stacked_data[t] == 0) & (adjusted_snow_image == 1)
        convolved_stack[t][combined_mask] = 1
    return convolved_stack


def calculate_pixel_buffer(
    buff_raster_path: str, smaller_raster_path: str
) -> Tuple[int, int, int, int]:
    """
    Calculate the pixel buffer between a larger buffered raster and a smaller raster
    by comparing their spatial extents.

    Parameters
    ----------
        buff_raster_path: str
            File path to the larger buffered raster
        smaller_raster_path: str
            File path to the smaller (unbuffered) raster

    Returns
    -------
        buffer_left_px: int
            Buffer size in pixels on the left side
        buffer_right_px: int
            Buffer size in pixels on the right side
        buffer_top_px: int
            Buffer size in pixels on the top side
        buffer_bottom_px: int
            Buffer size in pixels on the bottom side
    """
    with rio.open(buff_raster_path) as buff_src:
        buff_transform = buff_src.transform
        buff_bounds = buff_src.bounds

    with rio.open(smaller_raster_path) as smaller_src:
        smaller_bounds = smaller_src.bounds

    buffer_left_px = int(
        round(abs((buff_bounds[0] - smaller_bounds[0]) / buff_transform[0]))
    )
    buffer_bottom_px = int(
        round(abs((buff_bounds[1] - smaller_bounds[1]) / buff_transform[4]))
    )
    buffer_right_px = int(
        round(abs((buff_bounds[2] - smaller_bounds[2]) / buff_transform[0]))
    )
    buffer_top_px = int(
        round(abs((buff_bounds[3] - smaller_bounds[3]) / buff_transform[4]))
    )

    return buffer_left_px, buffer_right_px, buffer_top_px, buffer_bottom_px


def basin_masking(
    arr: np.ndarray,
    basin_mask: np.ndarray,
    buff_raster_path: str,
    smaller_raster_path: str,
) -> np.ndarray:
    """
    Apply a basin mask to a stacked array and crop the buffer pixels from each edge.

    Parameters
    ----------
        arr: np.ndarray
            3D array of shape (time, y, x)
        basin_mask: np.ndarray
            2D binary mask where 1 = inside basin, 0 = outside
        buff_raster_path: str
            File path to the larger buffered raster (used to compute pixel buffer)
        smaller_raster_path: str
            File path to the smaller raster (used to compute pixel buffer)

    Returns
    -------
        cropped_stack: np.ndarray
            3D array with basin mask applied and buffer pixels removed
    """
    buffer_left, buffer_right, buffer_top, buffer_bottom = calculate_pixel_buffer(
        buff_raster_path, smaller_raster_path
    )
    t, y, x = arr.shape
    new_y = y - buffer_top - buffer_bottom
    new_x = x - buffer_left - buffer_right
    assert new_y > 0 and new_x > 0, "Cropping would result in negative dimensions."
    cropped_stack = np.zeros((t, new_y, new_x), dtype=arr.dtype)
    for i in range(t):
        masked = np.where(basin_mask == 1, arr[i], np.nan)
        cropped_stack[i] = masked[
            buffer_top : y - buffer_bottom, buffer_left : x - buffer_right
        ]
    return cropped_stack


def temporal_median_filter(
    arr: np.ndarray, mask: np.ndarray, code: int, window_size: int
) -> np.ndarray:
    """
    Apply a temporal median filter to pixels matching a specific mask code.

    Parameters
    ----------
        arr: np.ndarray
            3D array of shape (time, y, x)
        mask: np.ndarray
            2D mask array
        code: int
            Mask value identifying pixels to filter (e.g. 0 for open, 1 for forest)
        window_size: int
            Temporal window size for the median filter (must be odd)

    Returns
    -------
        result: np.ndarray
            3D array with temporal median filter applied to masked pixels
    """
    assert window_size % 2 == 1, "window_size must be odd"
    filtered = median_filter(arr, size=(window_size, 1, 1), mode="nearest")
    valid_mask = np.broadcast_to((mask == code), arr.shape)
    result = arr.copy()
    result[valid_mask] = filtered[valid_mask]
    return result


def qaqc_spatial_temp(
    stacked_data: np.ndarray,
    chm_mask: np.ndarray,
    radius: int,
    threshold_percent: float,
    basin_mask: np.ndarray,
    buff_raster_fn: str,
    small_raster_fn: str,
) -> np.ndarray:
    """
    Apply combined spatial (canopy infilling) and temporal (median filter + nodata fill)
    quality control to a stacked SCA dataset.

    Parameters
    ----------
        stacked_data: np.ndarray
            3D array of shape (time, y, x) with snow classification values
        chm_mask: np.ndarray
            3D canopy height mask array (first band used); 0 = forest, 1 = open
        radius: int
            Radius of the circular kernel for spatial convolution
        threshold_percent: float
            Fraction of kernel pixels that must be snow to reclassify a forested pixel
        basin_mask: np.ndarray
            2D binary basin mask where 1 = inside basin
        buff_raster_fn: str
            File path to the buffered raster for pixel buffer calculation
        small_raster_fn: str
            File path to the smaller raster for pixel buffer calculation

    Returns
    -------
        stacked_data: np.ndarray
            QA/QC-corrected 3D array clipped to basin extent
    """
    stacked_data = masked_convolution(
        stacked_data, chm_mask[0], create_circular_kernel(radius), threshold_percent
    )
    stacked_data = temporal_median_filter(stacked_data, chm_mask[0], 1, window_size=5)
    stacked_data = temporal_median_filter(stacked_data, chm_mask[0], 0, window_size=5)
    stacked_data = fill_nodata(stacked_data)
    stacked_data = basin_masking(
        stacked_data, basin_mask, buff_raster_fn, small_raster_fn
    )
    return stacked_data


def qaqc_spatial(
    stacked_data: np.ndarray,
    chm_mask: np.ndarray,
    radius: int,
    threshold_percent: float,
    basin_mask: np.ndarray,
    buff_raster_fn: str,
    small_raster_fn: str,
) -> np.ndarray:
    """
    Apply spatial-only (canopy infilling) quality control to a stacked SCA dataset.

    Parameters
    ----------
        stacked_data: np.ndarray
            3D array of shape (time, y, x) with snow classification values
        chm_mask: np.ndarray
            3D canopy height mask array (first band used); 0 = forest, 1 = open
        radius: int
            Radius of the circular kernel for spatial convolution
        threshold_percent: float
            Fraction of kernel pixels that must be snow to reclassify a forested pixel
        basin_mask: np.ndarray
            2D binary basin mask where 1 = inside basin
        buff_raster_fn: str
            File path to the buffered raster for pixel buffer calculation
        small_raster_fn: str
            File path to the smaller raster for pixel buffer calculation

    Returns
    -------
        stacked_data: np.ndarray
            Spatially corrected 3D array clipped to basin extent
    """
    stacked_data = masked_convolution(
        stacked_data, chm_mask[0], create_circular_kernel(radius), threshold_percent
    )
    stacked_data = basin_masking(
        stacked_data, basin_mask, buff_raster_fn, small_raster_fn
    )
    return stacked_data


def qaqc_temp(
    stacked_data: np.ndarray,
    chm_mask: np.ndarray,
    basin_mask: np.ndarray,
    buff_raster_fn: str,
    small_raster_fn: str,
) -> np.ndarray:
    """
    Apply temporal-only (median filter + nodata fill) quality control to a stacked SCA dataset.

    Parameters
    ----------
        stacked_data: np.ndarray
            3D array of shape (time, y, x) with snow classification values
        chm_mask: np.ndarray
            3D canopy height mask array (first band used); 0 = forest, 1 = open
        basin_mask: np.ndarray
            2D binary basin mask where 1 = inside basin
        buff_raster_fn: str
            File path to the buffered raster for pixel buffer calculation
        small_raster_fn: str
            File path to the smaller raster for pixel buffer calculation

    Returns
    -------
        stacked_data: np.ndarray
            Temporally corrected 3D array clipped to basin extent
    """
    stacked_data = temporal_median_filter(stacked_data, chm_mask[0], 1, window_size=5)
    stacked_data = temporal_median_filter(stacked_data, chm_mask[0], 0, window_size=5)
    stacked_data = fill_nodata(stacked_data)
    stacked_data = basin_masking(
        stacked_data, basin_mask, buff_raster_fn, small_raster_fn
    )
    return stacked_data

def find_last_consecutive_ones_after_zero(arr: np.ndarray) -> list:
    """
    Find the index of the last date in the longest run of consecutive snow-present
    observations that follows a snow-absent observation.

    Parameters
    ----------
        arr: np.ndarray
            1D array of snow values (0 = no snow, 1 = snow) for a single pixel

    Returns
    -------
        max_consecutive_indices: list
            List containing the index of the last date in the longest qualifying run
    """
    max_consecutive_indices = []
    n = len(arr)
    i = 0
    while i < n - 1:
        if arr[i] == 0 and arr[i + 1] == 1:
            start_index = i + 1
            while (
                start_index < n - 1
                and arr[start_index] == 1
                and arr[start_index + 1] == 1
            ):
                start_index += 1
            consecutive_indices = [start_index]
            if len(consecutive_indices) > len(max_consecutive_indices):
                max_consecutive_indices = consecutive_indices
            i = start_index + 1
        else:
            i += 1
    return max_consecutive_indices


def fill_DSDdates(
    daydiff: np.ndarray, dsd: np.ndarray, nodata_ref: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the day-since-disappearance (DSD) date and its uncertainty, accounting
    for nodata pixels at the DSD timestep by walking back to the nearest valid observation.

    Parameters
    ----------
        daydiff: np.ndarray
            1D array of day-of-water-year values corresponding to each timestep
        dsd: np.ndarray
            2D array of per-pixel DSD timestep indices
        nodata_ref: np.ndarray
            3D array of shape (time, y, x) with snow classification values (2 = nodata)

    Returns
    -------
        dsd_date: np.ndarray
            2D array of DSD dates (day of water year), set to midpoint between
            last snow-present and first snow-absent observation
        dsd_uncertain: np.ndarray
            2D array of uncertainty in DSD date (days)
    """
    dsd_date = daydiff[dsd].astype(float)
    temp = np.empty(dsd.shape)
    # loop through the dsd array
    # determine the snow classification on the dsd date
    for j in range(dsd.shape[0]):
        for i in range(dsd.shape[1]):
            time_index = dsd[j, i]
            temp[j, i] = nodata_ref[time_index, j, i]
    
    # determine where/when dsd occurred on a bad observation pixel and account for that in the uncertainty
    filled_y, filled_x = np.where(temp == 2)
    if len(filled_y) > 0:
        for j, i in zip(filled_y, filled_x):
            time_index = dsd[j, i]
            temp = nodata_ref[time_index, j, i]
            while temp == 2 and time_index > 1 - nodata_ref.shape[0]:
                time_index = time_index - 1
                temp = nodata_ref[time_index, j, i]
            # fill the dsd with the last good date
            dsd[j, i] = time_index

    temp = daydiff[dsd - 1].astype(float)
    temp[dsd < 0] = np.nan
    dsd_uncertain = dsd_date - temp
    dsd_date = np.ceil((dsd_date + temp) / 2)
    return dsd_date, dsd_uncertain

def DSD_index(year: int, files: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the days from 1 October for each image and create the snow cover
    data cube ``(time, y, x)``.

    Parameters
    ----------
        year: int
            Water year being processed (reference date is set to ``year - 1``, Oct 1)
        files: List[str]
            Ordered list of file paths to ``*SCA.tif`` rasters with ``YYYYMMDD_``
            filename prefixes

    Returns
    -------
        stacked_data: np.ndarray
            3D float array of shape ``(time, y, x)`` with snow cover classifications
            (0 = no snow, 1 = snow, 2 = nodata/artifact)
        daydiff: np.ndarray
            1D integer array of days elapsed since 1 October of ``year - 1``
    """
    # instantiate
    data_list, daydiff = [], []
    # set the reference date to the start of the water year
    ref_date = datetime.date(year - 1, 10, 1)

    # loop through the snow cover observations
    for _fCount, file in enumerate(files):
        # determine/log date
        breakFile = os.path.basename(file).split("_")[0]
        dater = datetime.date(
            int(breakFile[0:4]), int(breakFile[4:6]), int(breakFile[6:8])
        )

        # determine how many days into the water year the PS obs is from
        daydiff.append((dater - ref_date).days)

        # read in and append the data
        with rio.open(file) as src:
            data = src.read(1)
            # Switches nodata values to 2 from default of 9 to match fill_nodata
            if src.nodata is not None:
                data[data == src.nodata] = 2
            data_list.append(data)

    # create stacked dataframe for snow cover
    stacked_data = np.stack(data_list, axis=0).astype(float)
    daydiff = np.array(daydiff)
    # print(daydiff)
    return stacked_data, daydiff

def find_spurious_0(arr: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Find single no-snow observations (0) preceded and followed by snow cover (1).

    Parameters
    ----------
        arr: np.ndarray
            3D array of shape ``(time, y, x)`` with snow classification values

    Returns
    -------
        locations: List[Tuple[int, int, int]]
            List of ``(t, i, j)`` index tuples where ``arr[t, i, j] == 0`` and
            ``arr[t - 1, i, j] == 1`` and ``arr[t + 1, i, j] == 1``
    """
    locations = []
    time, x, y = arr.shape
    for t in range(1, time - 1):
        for i in range(x):
            for j in range(y):
                if (
                    arr[t - 1, i, j] == 1
                    and arr[t, i, j] == 0
                    and arr[t + 1, i, j] == 1
                ):
                    locations.append((t, i, j))
    return locations


# function for correcting snow dates preceded and followed by no snow cover (SEASONAL SNOW ONLY!!)
def find_spurious_1(arr: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Find single snow observations (1) preceded and followed by no snow cover (0).

    Parameters
    ----------
        arr: np.ndarray
            3D array of shape ``(time, y, x)`` with snow classification values

    Returns
    -------
        locations: List[Tuple[int, int, int]]
            List of ``(t, i, j)`` index tuples where ``arr[t, i, j] == 1`` and
            ``arr[t - 1, i, j] == 0`` and ``arr[t + 1, i, j] == 0``
    """
    locations = []
    time, x, y = arr.shape
    for t in range(1, time - 1):
        for i in range(x):
            for j in range(y):
                if (
                    arr[t - 1, i, j] == 0
                    and arr[t, i, j] == 1
                    and arr[t + 1, i, j] == 0
                ):
                    locations.append((t, i, j))
    return locations


# fill spurious values using the defined "fill_value"
def fill_locations(
    arr: np.ndarray, locations: List[Tuple[int, int, int]], fill_value: float
) -> np.ndarray:
    """
    Fill specified ``(t, x, y)`` locations in a 3D array with ``fill_value``.

    Parameters
    ----------
        arr: np.ndarray
            3D array of shape ``(time, y, x)`` to modify
        locations: List[Tuple[int, int, int]]
            List of ``(t, x, y)`` index tuples to overwrite
        fill_value: float
            Replacement value written at each index in ``locations``

    Returns
    -------
        arr: np.ndarray
            3D array with the specified locations set to ``fill_value``

    """
    for loc in locations:
        t, x, y = loc
        arr[t, x, y] = fill_value
    return arr


# perform data QA and QC using the functions from above (SEASONAL SNOW ONLY!!)
def DSD_qaqc(stacked_data: np.ndarray) -> np.ndarray:
    """
    Perform seasonal-snow temporal QA/QC on a nodata-filled snow cover stack.

    Replaces isolated ``1 -> 0 -> 1`` transitions with ``1``, replaces isolated
    ``0 -> 1 -> 0`` transitions with ``0``, and backfills all timesteps prior to
    the end of the first ``0 -> 1`` snow run with ``1``.

    Parameters
    ----------
        stacked_data: np.ndarray
            3D array of shape ``(time, y, x)`` with binary snow values (0 or 1)

    Returns
    -------
        stacked_data: np.ndarray
            QA/QC-filtered 3D snow cover array

    """
    locations_1 = find_spurious_0(stacked_data)
    stacked_data = fill_locations(stacked_data, locations_1, 1)
    locations_2 = find_spurious_1(stacked_data)
    stacked_data = fill_locations(stacked_data, locations_2, 0)
    _, y, x = stacked_data.shape
    for j in range(y):
        for i in range(x):
            location_3 = find_last_consecutive_ones_after_zero(stacked_data[:, j, i])
            if len(location_3) > 0:
                stacked_data[: location_3[0], j, i] = 1
    return stacked_data


def process_DSD(
    years: List[int],
    data_direc: str,
    starting_array: Optional[List[int]] = None,
    year_select: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate per-pixel Date of Snow Disappearance (DSD), domain-normalized DSD
    anomaly, and DSD uncertainty across one or more water years from ``*SCA.tif`` rasters.

    Parameters
    ----------
        years: List[int]
            List of years to process
        data_direc: str
            Directory containing ``<year>*SCA.tif`` snow cover rasters
        starting_array: Optional[List[int]]
            Per-year index of the first SCA image to include when slicing the sorted
            file list (``0`` uses all snow cover maps for that year; in seasonal-snow
            domains, chosen near or after peak winter accumulation so the melt window
            begins with continuous snow cover). Defaults to all zeros when ``None``.
        year_select: Optional[int]
            Focus year for which ``DSD_stacked.npy`` and ``DSD_daydiff.npy`` are saved
            under ``<data_direc>/processed_<year_select>/``. Nothing is saved when ``None``.

    Returns
    -------
        dsd_saver: np.ndarray
            3D array of shape ``(len(years), y, x)`` containing per-pixel DSD in days
            since 1 October of ``year - 1``
        dsd_anomaly_saver: np.ndarray
            3D array of shape ``(len(years), y, x)`` containing domain median-normalized
            DSD anomalies (``dsd - np.nanmedian(dsd)``) in days
        dsd_uncertain_saver: np.ndarray
            3D array of shape ``(len(years), y, x)`` containing per-pixel DSD
            uncertainty in days
    """
    if starting_array is None:
        starting_array = [0] * len(years)
    # loop through each testing year
    for year_index, testing_year in enumerate(years):
        # load the processed snow cover files
        files = sorted(
            glob.glob(os.path.join(data_direc, str(testing_year) + "*SCA.tif"))
        )[
            starting_array[year_index] :
        ]

        stacked_data, daydiff = DSD_index(testing_year, files)

        # fill instances with no-data classifications (from image artifacts)
        nodata_ref = stacked_data.copy()
        stacked_data = fill_nodata(stacked_data)

        # do data filtering and post-processing (SEASONAL SNOW ONLY!!)
        stacked_data = DSD_qaqc(stacked_data)

        # save the data if this is a modeling year
        if testing_year == year_select:
            processed_direc = os.path.join(
                data_direc, "processed_" + str(year_select)
            )
            os.makedirs(processed_direc, exist_ok=True)
            np.save(os.path.join(processed_direc, "DSD_stacked.npy"), stacked_data)
            np.save(os.path.join(processed_direc, "DSD_daydiff.npy"), daydiff)

        # calculate the date of snow disappearance
        dsd = np.argmin(stacked_data, axis=0)

        # determine the actual dsd date and uncertainty
        dsd_masked = dsd.copy()
        dsd, dsd_uncertain = fill_DSDdates(daydiff, dsd, nodata_ref)
        # mask values where snow was never observed
        dsd[dsd_masked == 0] = np.nan
        dsd_uncertain[dsd_masked == 0] = np.nan

        # calculate the domain annual-normalized pattern
        dsd_anomaly = dsd - np.nanmedian(dsd)

        # append the data
        if year_index == 0:
            shpp = dsd_anomaly.shape
            dsd_saver = np.empty((len(years), shpp[0], shpp[1]))
            dsd_anomaly_saver = np.empty((len(years), shpp[0], shpp[1]))
            dsd_uncertain_saver = np.empty((len(years), shpp[0], shpp[1]))
        dsd_saver[year_index, :, :] = dsd
        dsd_anomaly_saver[year_index, :, :] = dsd_anomaly
        dsd_uncertain_saver[year_index, :, :] = dsd_uncertain

    return dsd_saver, dsd_anomaly_saver, dsd_uncertain_saver
