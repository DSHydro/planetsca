from typing import Tuple

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
    for j in range(dsd.shape[0]):
        for i in range(dsd.shape[1]):
            time_index = dsd[j, i]
            temp[j, i] = nodata_ref[time_index, j, i]

    filled_y, filled_x = np.where(temp == 2)
    if len(filled_y) > 0:
        for j, i in zip(filled_y, filled_x):
            time_index = dsd[j, i]
            val = nodata_ref[time_index, j, i]
            while val == 2:
                time_index = dsd[j, i] - 1
                val = nodata_ref[time_index, j, i]
            dsd[j, i] = time_index

    temp = daydiff[dsd - 1].astype(float)
    temp[dsd < 0] = np.nan
    dsd_uncertain = dsd_date - temp
    dsd_date = np.ceil((dsd_date + temp) / 2)
    return dsd_date, dsd_uncertain
