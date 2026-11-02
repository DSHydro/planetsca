from typing import List, Optional, Tuple

import numpy as np


def get_SWEdist(D: np.ndarray, meanSWE: float, CV: float) -> np.ndarray:
    """
    Compute a discretized lognormal snow water equivalent (SWE) probability
    distribution over candidate SWE values ``D``, normalized to sum to 1.

    Given the arithmetic mean ``meanSWE`` and coefficient of variation ``CV``
    (standard deviation divided by mean), the log-space standard deviation is
    ``zeta = sqrt(ln(1 + CV^2))`` and the log-space mean is
    ``lam = ln(meanSWE) - 0.5 * zeta^2``.

    Parameters
    ----------
        D: np.ndarray
            1D array of candidate SWE values (m), all > 0
        meanSWE: float
            Spatial mean SWE (m) of the distribution
        CV: float
            Spatial coefficient of variation of SWE (standard deviation / mean)

    Returns
    -------
        fD: np.ndarray
            1D array of the same shape as ``D`` giving the normalized probability
            weight at each SWE value
    """
    # convert arithmetic mean and CV to lognormal distribution parameters (zeta and lambda)
    zeta = np.sqrt(np.log(1 + (CV**2)))
    lam = np.log(meanSWE) - (0.5 * (zeta**2))
    # evaluate the lognormal probability density at each SWE value in D
    fD = (1 / (D * zeta * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((np.log(D) - lam) / zeta) ** 2
    )
    # normalize discrete weights to sum to 1
    fD = fD / np.sum(fD)
    return fD


def simulate_fSCA(
    D: np.ndarray,
    meanSWE: float,
    CV: float,
    SW: np.ndarray,
    MD: int,
    MF: float,
    offset: float,
) -> np.ndarray:
    """
    Simulate daily fractional snow-covered area (fSCA) by uniformly melting a
    lognormal initial SWE distribution using cumulative shortwave radiation.

    An initial spatial SWE distribution is generated across ``D`` from ``meanSWE``
    and ``CV``. Starting on day index ``MD``, daily cumulative melt is calculated as
    ``cumulative_SW * MF + offset`` and subtracted uniformly from ``D``. Each day's
    simulated fSCA is the fraction of the initial distribution where remaining SWE
    is still greater than zero (``D - melt > 0``).

    Parameters
    ----------
        D: np.ndarray
            1D array of candidate SWE values (m), all > 0
        meanSWE: float
            Mean SWE (m) at the start of the melt simulation
        CV: float
            SWE coefficient of variation
        SW: np.ndarray
            1D array of daily-mean incoming shortwave radiation (W/m^2) indexed
            from the start of the water year (1 October)
        MD: int
            Melt-start day index into ``SW`` (days since 1 October); radiation
            before this index is set to 0 prior to accumulating melt
        MF: float
            Shortwave melt factor (m of SWE melt per W/m^2 per day)
        offset: float
            Uniform SWE shift (m) added to the cumulative melt to account for
            potential bias in ``meanSWE``

    Returns
    -------
        fsca_out: np.ndarray
            1D array of simulated daily fSCA values (0 to 1), one per day of ``SW``

    Notes
    -----
    ``SW`` is modified in place by zeroing entries before ``MD`` (``SW[:MD] = 0``).
    """
    # generate the initial lognormal SWE distribution from mean SWE and CV
    fD = get_SWEdist(D, meanSWE, CV)
    # zero out radiation prior to the melt start date and compute cumulative radiation
    SW[:MD] = 0
    SW = np.cumsum(SW)
    # simulate daily fSCA as cumulative radiation melts the SWE distribution
    fsca_out = []
    for sw_sel in SW:
        # compute total melt to date and subtract it uniformly across the SWE distribution
        melt = (sw_sel * MF) + offset
        newSWE = D - melt
        # fSCA is the fraction of the initial distribution with remaining SWE > 0
        fsca_out.append(np.sum(fD[newSWE > 0]))
    fsca_out = np.array(fsca_out)
    return fsca_out


def process_fSCA(
    dsdCube: np.ndarray,
    daydiff: np.ndarray,
    snow_classes: np.ndarray,
    dsd: np.ndarray,
) -> Tuple[np.ndarray, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the domain-wide fractional snow-covered area (fSCA) time series and
    truncate the observation record to the active snowmelt fitting window.

    Pixels outside the valid snow-class domain (where ``snow_classes`` is NaN) are
    masked out in both ``dsdCube`` and ``dsd``. To prevent late-season snow-free
    dates from dominating melt-factor optimization, the fitting window ends 10
    observations after domain fSCA first drops to 1% or less (``<= 0.01``).

    Parameters
    ----------
        dsdCube: np.ndarray
            3D float array of shape ``(time, y, x)`` containing QA/QC-filtered
            binary snow cover (1 = snow, 0 = no snow)
        daydiff: np.ndarray
            1D integer array of days since 1 October for each observation in ``dsdCube``
        snow_classes: np.ndarray
            2D array of snow class labels (e.g., 0-3), with NaN outside the valid
            modeling domain
        dsd: np.ndarray
            2D array of per-pixel Date of Snow Disappearance (days since 1 October)

    Returns
    -------
        fsca: np.ndarray
            1D array of domain-wide fSCA (0 to 1) for each observation date
        idx_start: int
            Start index of the fitting window (always 0)
        idx_end: int
            End index (exclusive) of the fitting window, set to 10 observations
            after ``fsca`` first drops to ``<= 0.01``
        daydiff_filt: np.ndarray
            ``daydiff`` sliced to the fitting window ``[idx_start:idx_end]``
        fsca_filt: np.ndarray
            ``fsca`` sliced to the fitting window ``[idx_start:idx_end]``
        dsdCube_filt: np.ndarray
            ``dsdCube`` sliced to the fitting window ``[idx_start:idx_end]``
        dsd_filt: np.ndarray
            Copy of ``dsd`` with NaN outside the valid ``snow_classes`` domain
    """
    # mask out pixels in the binary snow cover stack that fall outside the valid snow-class domain
    classes_stacked = np.tile(snow_classes, [len(daydiff), 1, 1])
    dsdCube[np.isnan(classes_stacked)] = np.nan
    # count the total number of valid grid cells in the modeling domain
    total_area = len(snow_classes[~np.isnan(snow_classes)])
    # compute domain-wide fractional snow-covered area (fSCA) at each observation timestep
    fsca = np.nansum(dsdCube, axis=(1, 2)) / total_area

    # truncate the time series shortly after domain fSCA drops to <= 1% to avoid
    # overweighting snow-free dates when fitting melt factors
    idx_start = 0
    idx_end = np.where(fsca <= 0.01)
    if len(idx_end) >= 1:
        try:
            idx_end = idx_end[0][0] + 10
        except IndexError:
            idx_end = idx_end[0][0] + 1
    else:
        idx_end = len(fsca) + 1
    daydiff_filt = daydiff[idx_start:idx_end]
    fsca_filt = fsca[idx_start:idx_end]
    dsdCube_filt = dsdCube[idx_start:idx_end, :, :]
    dsd_filt = dsd.copy()
    # mask out snow disappearance dates outside the valid snow-class domain
    dsd_filt[np.isnan(snow_classes)] = np.nan

    return fsca, idx_start, idx_end, daydiff_filt, fsca_filt, dsdCube_filt, dsd_filt


def reconstruct_pillow_SWE(
    dsd: np.ndarray,
    dsd_filt: np.ndarray,
    daydiff_filt: np.ndarray,
    dailySW: np.ndarray,
    MF: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct initial spring SWE on ``daydiff_filt[0]`` from per-pixel snow
    disappearance dates and a snow-pillow linear shortwave melt model.

    Assuming negligible snowfall during the spring melt window, a pixel's SWE on
    the initial date ``t0 = daydiff_filt[0]`` equals the cumulative melt between
    ``t0`` and the pixel's Date of Snow Disappearance (``dsd``):
    ``SWE = MF[0] * sum(dailySW[t0 : dsd + 1]) + MF[1]``.
    Pixels where snow disappeared prior to ``t0`` are set to 0, and pixels outside
    the valid domain (where ``dsd_filt`` is NaN) remain NaN.

    Parameters
    ----------
        dsd: np.ndarray
            2D array of per-pixel Date of Snow Disappearance (days since 1 October)
        dsd_filt: np.ndarray
            2D array of per-pixel DSD masked to NaN outside the valid snow-class
            domain (from ``process_fSCA``)
        daydiff_filt: np.ndarray
            1D array of observation dates (days since 1 October) in the fitting
            window; ``daydiff_filt[0]`` defines the initial reconstruction date
        dailySW: np.ndarray
            1D array of daily-mean incoming shortwave radiation (W/m^2) from 1 October
        MF: np.ndarray
            1D array or sequence ``[slope, intercept]`` of snow-pillow linear melt
            model parameters: ``MF[0]`` is the melt factor (m of SWE per W/m^2 per
            day) and ``MF[1]`` is the linear SWE intercept (m)

    Returns
    -------
        SWE_map_base: np.ndarray
            2D array of reconstructed SWE (m) on ``daydiff_filt[0]``, with 0 where
            ``dsd < daydiff_filt[0]`` and NaN outside the valid domain

    Notes
    -----
    - Cumulative shortwave radiation is summed from ``daydiff_filt[0]`` through
      ``int(dsd_sel)`` inclusive (``np.cumsum(swsum)[int(dsd_sel)]``).
    - Reconstructed SWE scales linearly with cumulative shortwave radiation, so
      uncertainty in ``dsd`` propagates into SWE at roughly ``MF[0] * dailySW``
      per day.
    """
    # find all unique valid snow disappearance dates across the domain
    dsd_unique = np.unique(dsd)
    dsd_unique = dsd_unique[~np.isnan(dsd_unique)]

    # initialize the output SWE map with NaN
    SWE_map_base = np.empty(dsd.shape)
    SWE_map_base[:] = np.nan
    # compute initial SWE for each unique snow disappearance date
    for dsd_sel in dsd_unique:
        # if snow disappeared before the initial modeling date, SWE on that date is 0
        if dsd_sel < daydiff_filt[0]:
            SWE_map_base[dsd_filt == dsd_sel] = 0
        else:
            # sum daily shortwave radiation from the initial modeling date through the disappearance date
            swsum = dailySW.copy()
            swsum[: daydiff_filt[0]] = 0
            swsum = np.cumsum(swsum)[int(dsd_sel)]
            # convert cumulative shortwave radiation to SWE using the snow-pillow linear melt relationship
            SWE_map_base[dsd_filt == dsd_sel] = (swsum * MF[0]) + MF[1]

    return SWE_map_base


def fit_domain_melt_factor(
    SWE_map_base: np.ndarray,
    dailySW: np.ndarray,
    daydiff_filt: np.ndarray,
    fsca_filt: np.ndarray,
    MF: np.ndarray,
    D: Optional[np.ndarray] = None,
    C_array: Optional[np.ndarray] = None,
    B_array: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, List[float]]:
    """
    Calibrate a single domain-wide melt factor, SWE coefficient of variation, and
    SWE offset by fitting simulated fSCA to the observed domain-wide fSCA evolution.

    Performs a grid search over candidate SWE coefficients of variation (``C_array``),
    20 uniform SWE offsets spanning +-10% of the domain-mean initial SWE
    (``np.nanmean(SWE_map_base)``), and candidate shortwave melt factors (``B_array``).
    For each parameter combination, daily fSCA is simulated with ``simulate_fSCA``
    and evaluated against ``fsca_filt`` on the observation dates ``daydiff_filt``
    using mean squared error (MSE).

    Parameters
    ----------
        SWE_map_base: np.ndarray
            2D array of baseline initial SWE (m) on ``daydiff_filt[0]`` (from
            ``reconstruct_pillow_SWE``); its spatial mean sets the mean of the
            lognormal SWE distribution
        dailySW: np.ndarray
            1D array of daily-mean incoming shortwave radiation (W/m^2) from 1 October
        daydiff_filt: np.ndarray
            1D array of observation dates (days since 1 October) in the fitting window
        fsca_filt: np.ndarray
            1D array of observed domain-wide fSCA in the fitting window (from
            ``process_fSCA``)
        MF: np.ndarray
            Snow-pillow melt model parameters ``[slope, intercept]``; ``MF[0]`` is
            used to center the default ``B_array`` search range
        D: Optional[np.ndarray]
            Candidate SWE values (m) for the lognormal distribution. Defaults to
            ``np.linspace(0.0001, 7, 2000)``
        C_array: Optional[np.ndarray]
            Candidate SWE coefficients of variation. Defaults to
            ``np.linspace(0.05, 0.5, 20)``
        B_array: Optional[np.ndarray]
            Candidate shortwave melt factors (m per W/m^2 per day). Defaults to
            ``np.linspace(MF[0] * 0.5, MF[0] * 1.5, 20)``

    Returns
    -------
        B_single: float
            Best-fitting domain-wide melt factor (m per W/m^2 per day)
        C_single: float
            Best-fitting domain-wide SWE coefficient of variation
        offset_single: float
            Best-fitting uniform SWE offset (m)
        MSE: List[float]
            Mean squared error for every evaluated parameter combination, ordered
            by ``C_array`` -> ``offsets`` -> ``B_array``

    Notes
    -----
    - If multiple parameter combinations tie for the minimum MSE, the first
      combination in grid-search order is selected.
    - ``dailySW`` is modified in place by ``simulate_fSCA`` (entries prior to
      ``daydiff_filt[0]`` are set to 0).
    """
    # default grid of candidate SWE values (m)
    if D is None:
        D = np.linspace(0.0001, 7, 2000)
    # default grid of candidate SWE coefficients of variation
    if C_array is None:
        C_array = np.linspace(0.05, 0.5, 20)
    # default grid of candidate melt factors: +-50% around the snow-pillow melt factor
    if B_array is None:
        B_array = np.linspace(MF[0] * 0.5, MF[0] * 1.5, 20)

    # initialize lists to track parameter combinations and their mean squared errors
    MSE = []
    C_saver, B_saver, offset_saver = [], [], []

    # grid search over SWE coefficients of variation, SWE offsets, and melt factors
    for C in C_array:
        # evaluate the initial lognormal SWE distribution for the domain-mean SWE and current CV
        _temp_dist = get_SWEdist(D, np.nanmean(SWE_map_base), C)
        # shift the SWE distribution uniformly by 20 offsets spanning +-10% of domain-mean SWE
        offsets = np.linspace(
            -0.1 * np.nanmean(SWE_map_base), 0.1 * np.nanmean(SWE_map_base), 20
        )
        for offset in offsets:
            for B in B_array:
                # record the current parameter combination
                C_saver.append(C)
                B_saver.append(B)
                offset_saver.append(offset)
                # simulate daily fSCA evolution for this parameter combination
                test_output = simulate_fSCA(
                    D,
                    np.nanmean(SWE_map_base),
                    C,
                    dailySW,
                    daydiff_filt[0],
                    B,
                    offset,
                )
                # compute mean squared error between simulated and observed fSCA on image dates
                MSE.append(np.mean((test_output[daydiff_filt] - fsca_filt) ** 2))

    # select the parameter combination that minimizes mean squared error
    best_idx = np.where(MSE == np.nanmin(MSE))[0][0]
    B_single = B_saver[best_idx]
    C_single = C_saver[best_idx]
    offset_single = offset_saver[best_idx]
    print("best model parameters")
    print("Melt Factor: ", B_single)
    print("SWE CoV: ", C_single)
    print("Initial SWE offset: ", offset_single)

    return B_single, C_single, offset_single, MSE


def fit_class_melt_factors(
    SWE_map_base: np.ndarray,
    dailySW: np.ndarray,
    daydiff_filt: np.ndarray,
    dsdCube: np.ndarray,
    daydiff: np.ndarray,
    snow_classes: np.ndarray,
    idx_start: int,
    idx_end: int,
    MF: np.ndarray,
    D: Optional[np.ndarray] = None,
    C_array: Optional[np.ndarray] = None,
    B_array: Optional[np.ndarray] = None,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calibrate a separate melt factor, SWE coefficient of variation, and SWE offset
    for each snow class by fitting simulated fSCA to class-specific observed fSCA.

    For each unique non-NaN class in ``snow_classes``, the observed fSCA time series
    is computed over pixels belonging to that class and sliced to the fitting window
    ``[idx_start:idx_end]``. A grid search over ``C_array``, 20 uniform SWE offsets,
    and ``B_array`` is then performed using the class-mean initial SWE
    (``np.nanmean(SWE_map_base[snow_classes == class_no])``), selecting the parameter
    set with the lowest mean squared error (MSE) for each class.

    Parameters
    ----------
        SWE_map_base: np.ndarray
            2D array of baseline initial SWE (m) on ``daydiff_filt[0]`` (from
            ``reconstruct_pillow_SWE``)
        dailySW: np.ndarray
            1D array of daily-mean incoming shortwave radiation (W/m^2) from 1 October
        daydiff_filt: np.ndarray
            1D array of observation dates (days since 1 October) in the fitting window
        dsdCube: np.ndarray
            3D binary snow cover array of shape ``(time, y, x)``
        daydiff: np.ndarray
            1D array of observation dates (days since 1 October) for all images in
            ``dsdCube``
        snow_classes: np.ndarray
            2D array of snow class labels (e.g., 0-3), with NaN outside the valid
            modeling domain
        idx_start: int
            Start index of the fitting window (from ``process_fSCA``)
        idx_end: int
            End index (exclusive) of the fitting window (from ``process_fSCA``)
        MF: np.ndarray
            Snow-pillow melt model parameters ``[slope, intercept]``; ``MF[0]`` is
            used to center the default ``B_array`` search range
        D: Optional[np.ndarray]
            Candidate SWE values (m) for the lognormal distribution. Defaults to
            ``np.linspace(0.0001, 7, 2000)``
        C_array: Optional[np.ndarray]
            Candidate SWE coefficients of variation. Defaults to
            ``np.linspace(0.05, 0.5, 20)``
        B_array: Optional[np.ndarray]
            Candidate shortwave melt factors (m per W/m^2 per day). Defaults to
            ``np.linspace(MF[0] * 0.5, MF[0] * 1.5, 20)``

    Returns
    -------
        B_class: List[float]
            Best-fitting melt factor (m per W/m^2 per day) for each snow class, in
            ascending class order
        C_class: List[float]
            Best-fitting SWE coefficient of variation for each snow class, in
            ascending class order
        offset_class: List[float]
            Best-fitting uniform SWE offset (m) for each snow class, in ascending
            class order

    Notes
    -----
    - Candidate SWE offsets for each class span +-10% of the full-domain mean initial
      SWE (``np.nanmean(SWE_map_base)``), rather than the class-specific mean SWE.
    - ``dailySW`` is modified in place by ``simulate_fSCA`` (entries prior to
      ``daydiff_filt[0]`` are set to 0).
    """
    # default grids for candidate SWE values, coefficients of variation, and melt factors
    if D is None:
        D = np.linspace(0.0001, 7, 2000)
    if C_array is None:
        C_array = np.linspace(0.05, 0.5, 20)
    if B_array is None:
        B_array = np.linspace(MF[0] * 0.5, MF[0] * 1.5, 20)
    # identify all unique non-NaN snow class labels in ascending order
    class_numbers = np.unique(snow_classes)
    class_numbers = class_numbers[~np.isnan(class_numbers)]

    # initialize lists to store the best-fitting parameters per snow class
    B_class, C_class, offset_class = [], [], []
    # calibrate melt factor, SWE coefficient of variation, and SWE offset independently for each snow class
    for class_no in class_numbers:
        # initialize lists to store parameter combinations and MSE for the current snow class
        MSE = []
        C_saver, B_saver, offset_saver = [], [], []

        # mask the binary snow cover stack to retain only pixels in the current snow class
        classes_stacked = np.tile(snow_classes, [len(daydiff), 1, 1])
        dsdCube_classes = dsdCube.copy()
        dsdCube_classes[classes_stacked != class_no] = np.nan
        # compute observed fSCA time series for this snow class within the fitting window
        total_area = len(snow_classes[snow_classes == class_no])
        fsca_class = np.nansum(dsdCube_classes, axis=(1, 2)) / total_area
        fsca_class = fsca_class[idx_start:idx_end]

        # grid search over SWE coefficients of variation, SWE offsets, and melt factors
        for C in C_array:
            # evaluate the initial lognormal SWE distribution using the mean SWE of the current snow class
            _temp_dist = get_SWEdist(
                D, np.nanmean(SWE_map_base[snow_classes == class_no]), C
            )
            # shift the SWE distribution uniformly by 20 offsets spanning +-10% of domain-mean SWE
            offsets = np.linspace(
                -0.1 * np.nanmean(SWE_map_base), 0.1 * np.nanmean(SWE_map_base), 20
            )
            for offset in offsets:
                for B in B_array:
                    # record the current parameter combination
                    C_saver.append(C)
                    B_saver.append(B)
                    offset_saver.append(offset)

                    # simulate daily fSCA evolution for this snow class and parameter combination
                    test_output = simulate_fSCA(
                        D,
                        np.nanmean(SWE_map_base[snow_classes == class_no]),
                        C,
                        dailySW,
                        daydiff_filt[0],
                        B,
                        offset,
                    )
                    # compute mean squared error between simulated and observed class fSCA on image dates
                    MSE.append(np.mean((test_output[daydiff_filt] - fsca_class) ** 2))

        # select the best-fitting model parameters for this snow class
        best_idx = np.where(MSE == np.nanmin(MSE))[0][0]
        B_class.append(B_saver[best_idx])
        C_class.append(C_saver[best_idx])
        offset_class.append(offset_saver[best_idx])

    return B_class, C_class, offset_class
