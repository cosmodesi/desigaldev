"""
ir_agn_diagnostics.py
Author: Benjamin Floyd

Library containing all infrared AGN/galaxy diagnostic functions.

Original version of code written by:
Becky Canning (University of Portsmouth)
Stephanie Juneau (NOIRlab)
Mar Mezcula (Institut de Ciencies de l'Espai)
"""
from enum import Flag
from typing import Any, Literal

import numpy as np
from astropy.table import MaskedColumn, Table
from numpy.typing import NDArray


def wise_stern12(input_table: Table, snr: float | int = 3, mask: MaskedColumn = None) -> (
        tuple[NDArray[bool], NDArray[bool], NDArray[bool]]):
    r"""WISE color AGN selection originally by [Stern12]_

    Region defined in WISE infrared color space, indicating AGN.

    AGN selection defined by single color :math:`W1 - W2 > 0.8` (Vega).

    Notes:
        If using these diagnostic functions, please ref Mar_&_Steph_2025 and add appropriate references given below.

        If using DESI, please reference Summary_ref_2025 and the appropriate photometry catalog
        (e.g., Tractor or Photometry VAC).

        .. warning:: Note of caution: The points outside the AGN region may still include a significant fraction of
            AGN and are best considered as "uncertain" rather than "star forming" or "non-AGN".

    Args:
        input_table: Table including WISE fluxes and inverse variance. At minimum, catalog must contain columns for W1
            and W2 photometry.
        snr: Signal to noise cut applied to WISE photometry. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Tuple of arrays of same dimension as rows in ``input_table`` which include flags for
        ``avail_ir``, ``agn_ir``, ``sf_ir``.

    .. [Stern12] 2012ApJ...753...30S
    """

    # Mask for zero fluxes
    zero_flux_w1w2 = (input_table['FLUX_W1'] == 0) | (input_table['FLUX_W2'] == 0)

    # If user mask is provided (e.g., missing or masked photometry) apply this to our zero-flux mask
    if mask is not None:
        zero_flux_w1w2 |= mask

    # If ivar=0, set it to NaN to avoid infinities when computing the error
    input_table['FLUX_IVAR_W1'] = np.where(input_table['FLUX_IVAR_W1'] == 0, np.nan, input_table['FLUX_IVAR_W1'])
    input_table['FLUX_IVAR_W2'] = np.where(input_table['FLUX_IVAR_W2'] == 0, np.nan, input_table['FLUX_IVAR_W2'])

    # Mask for SNR
    snr_w1 = input_table['FLUX_W1'] * np.sqrt(input_table['FLUX_IVAR_W1'])
    snr_w2 = input_table['FLUX_W2'] * np.sqrt(input_table['FLUX_IVAR_W2'])

    # IR diagnostic availability based on W1W2 if flux is not zero and above SNR threshold
    w1w2_avail: NDArray[bool] = (~zero_flux_w1w2) & (snr_w1 > snr) & (snr_w2 > snr)

    # Convert fluxes to AB magnitudes
    w1 = 22.5 - 2.5 * np.log10(input_table['FLUX_W1'])
    w2 = 22.5 - 2.5 * np.log10(input_table['FLUX_W2'])

    # Convert the magnitudes from AB to Vega system
    w1_vega, w2_vega = wise_ab_vega(w1, w2)

    # Define W1 - W2 color
    w1w2_vega = w1_vega - w2_vega

    # Stern et al. (2012) cut is just along W1 - W2 color
    agn_stern12: NDArray[bool] = (w1w2_vega > 0.8) & w1w2_avail
    sf_stern: NDArray[bool] = (~agn_stern12)

    return w1w2_avail, agn_stern12, sf_stern


def wise_jarrett11(input_table: Table, snr: int | float = 3, mask: MaskedColumn = None) -> (
        tuple[NDArray[bool], NDArray[bool], NDArray[bool]]):
    r"""WISE color-color AGN selection originally by [Jarrett11]_

    Wedge region in W1-W2-W3 color-color space that largely isolates QSOs as measured in the ecliptic poles.

    AGN selection is defined as a wedged box region in the W1-W2-W3 color-color space given by Equation 1 in
    [Jarrett11]_
    ``agn_jarrett11``:
    :math:`\begin{cases}
    W2 - W3 > 2.2 \quad\mathrm{and}\quad W2 - W3 > 4.2, \\
    W1 - W2 > 0.1 \times (W2 - W3) + 0.38 \quad\mathrm{and}\quad W1 - W2 < 1.7
    \end{cases}`

    Notes:
        If using these diagnostic functions, please ref Mar_&_Steph_2025 and add appropriate references given below.

        If using DESI, please reference Summary_ref_2025 and the appropriate photometry catalog
        (e.g., Tractor or Photometry VAC).

        .. warning:: Note of caution: The points outside the AGN region may still include a significant fraction of
            AGN and are best considered as "uncertain" rather than "star forming" or "non-AGN".

    Args:
        input_table: Table including WISE fluxes and inverse variances. Table must include W1, W2, and W3 photometry.
        snr: Signal to noise ratio cut applied to WISE photometry. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Tuple of arrays of same dimension as rows in ``input_table`` which include flags for
        ``w1w2w3_avail``, ``agn_jarrett11``, and ``non_agn_jarrett11.

    .. [Jarrett11] 2011ApJ...735..112J
    """

    # Mask for zero fluxes
    zero_flux_wise = (input_table['FLUX_W1'] == 0) | (input_table['FLUX_W2'] == 0)
    zero_flux_w3 = input_table['FLUX_W3'] == 0
    if mask is not None:
        # Mask for flux availability - included if input_table photometry is missing/masked
        zero_flux_wise |= mask
        zero_flux_w3 |= mask

    # If ivar=0 set it to NaN to avoid infinities when computing the error:
    input_table['FLUX_IVAR_W1'] = np.where(input_table['FLUX_IVAR_W1'] == 0, np.nan, input_table['FLUX_IVAR_W1'])
    input_table['FLUX_IVAR_W2'] = np.where(input_table['FLUX_IVAR_W2'] == 0, np.nan, input_table['FLUX_IVAR_W2'])
    input_table['FLUX_IVAR_W3'] = np.where(input_table['FLUX_IVAR_W3'] == 0, np.nan, input_table['FLUX_IVAR_W3'])

    # Mask for SNR.
    snr_w1 = input_table['FLUX_W1'] * np.sqrt(input_table['FLUX_IVAR_W1'])
    snr_w2 = input_table['FLUX_W2'] * np.sqrt(input_table['FLUX_IVAR_W2'])
    snr_w3 = input_table['FLUX_W3'] * np.sqrt(input_table['FLUX_IVAR_W3'])

    ## IR diagnostic based on W1-W2-W3 is available if flux is not zero
    w1w2_avail = (~zero_flux_wise) & (snr_w1 > snr) & (snr_w2 > snr)
    w2w3_avail = (~zero_flux_wise) & (~zero_flux_w3) & (snr_w2 > snr) & (snr_w3 > snr)

    # Set availability of all three bands
    w1w2w3_avail = w1w2_avail & w2w3_avail

    # Convert fluxes to AB magnitudes
    w1 = 22.5 - 2.5 * np.log10(input_table['FLUX_W1'])
    w2 = 22.5 - 2.5 * np.log10(input_table['FLUX_W2'])
    w3 = 22.5 - 2.5 * np.log10(input_table['FLUX_W3'])

    # Convert the magnitudes from AB to Vega system
    w1_vega, w2_vega, w3_vega = wise_ab_vega(w1, w2, w3)

    # Define W1 - W2 and W2 - W3 colors
    w1w2_vega = w1_vega - w2_vega
    w2w3_vega = w2_vega - w3_vega

    # Jarrett et al. (2011) AGN box (Equation 1 in paper)
    agn_jarrett11 = ((w2w3_vega > 2.2) & (w2w3_vega < 4.2) & (w1w2_vega > 0.1 * w2w3_vega + 0.38) & (w1w2_vega < 1.7) &
                     w1w2w3_avail)

    # Define the non-AGN as the inverse selection
    non_agn_jarrett11 = w1w2w3_avail & (~agn_jarrett11)

    return w1w2w3_avail, agn_jarrett11, non_agn_jarrett11


def wise_assef18_r(input_table: Table, snr: int | float = 3, reliability: Literal[75, 90] = 90,
                 mask: MaskedColumn = None) -> (tuple[NDArray[bool], NDArray[bool], NDArray[bool]]):
    r"""Reliability-Optimized WISE color-color selection originally by [Assef18]_.

    Specifically, this selection implements the reliability-optimized criteria found in [Assef18]_ Equation 4.
    The completeness-optimized selection criteria (Equation 5 of [Assef18]_) is implemented by
    :func:`wise_assef18_c`.
    This selection improves upon the earlier selection in [Assef13]_.

    The selection criteria is given by Equation 4 in [Assef18]_.
    :math:`W1 - W2 >
    \begin{cases}
    \alpha_{R} \exp\{\beta_{R} (W2 - \gamma_{R})^{2}\}, & W2 > \gamma_{R}, \\
    \alpha_{R}, & W2 \leq \gamma_{R}
    \end{cases}`
    where :math:`\alpha_R, \beta_R,` and :math:`\gamma_R` depend on the reliability fraction targeted. For a reliability
    of 90% this is given by :math:`(\alpha_{R90}, \beta_{R90}, \gamma_{R90}) = (0.650, 0.153, 13.86)` and for a
    reliability of 75% is given by :math:`(\alpha_{R75}, \beta_{R75}, \gamma_{R75}) = (0.486, 0.092, 13.07)`.

    Notes:
        If using these diagnostic functions, please ref Mar_&_Steph_2025 and add appropriate references given below.

        If using DESI, please reference Summary_ref_2025 and the appropriate photometry catalog
        (e.g., Tractor or Photometry VAC).

        .. warning:: Note of caution: The points outside the AGN region may still include a significant fraction of
            AGN and are best considered as "uncertain" rather than "star forming" or "non-AGN".

    See Also:
        :func:`wise_assef18_c` for the completeness-optimized selection criteria version.

    Args:
        input_table: Table including WISE fluxes and inverse variance. At minimum, catalog must contain columns for W1
            and W2 photometry.
        snr: Signal to noise cut applied to WISE photometry. Default is ``3``.
        reliability: The reliability percent threshold to use for the selection. Must be either ``75`` or ``90``, the
            default is ``90``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Tuple of arrays of same dimension as rows in ``input_table`` which include flags for
        ``w1w2_avail``, ``agn_assef18``, and ``non_agn_assef18``.

    Raises:
        ValueError: If the reliability parameter is not ``75`` or ``90``.

    .. [Assef13] 2013ApJ...772...26A
    .. [Assef18] 2018ApJS..234...23A
    """

    # Mask for zero fluxes
    zero_flux_wise = (input_table['FLUX_W1'] == 0) | (input_table['FLUX_W2'] == 0)
    if mask is not None:
        # Mask for flux availability - included if input_table photometry is missing/masked
        zero_flux_wise |= mask

    # If ivar=0 set it to NaN to avoid infinities when computing the error:
    input_table['FLUX_IVAR_W1'] = np.where(input_table['FLUX_IVAR_W1'] == 0, np.nan, input_table['FLUX_IVAR_W1'])
    input_table['FLUX_IVAR_W2'] = np.where(input_table['FLUX_IVAR_W2'] == 0, np.nan, input_table['FLUX_IVAR_W2'])

    # Mask for SNR.
    snr_w1 = input_table['FLUX_W1'] * np.sqrt(input_table['FLUX_IVAR_W1'])
    snr_w2 = input_table['FLUX_W2'] * np.sqrt(input_table['FLUX_IVAR_W2'])

    ## IR diagnostic based on W1-W2-W3 is available if flux is not zero
    w1w2_avail = (~zero_flux_wise) & (snr_w1 > snr) & (snr_w2 > snr)

    # Convert fluxes to AB magnitudes
    w1 = 22.5 - 2.5 * np.log10(input_table['FLUX_W1'])
    w2 = 22.5 - 2.5 * np.log10(input_table['FLUX_W2'])

    # Convert the magnitudes from AB to Vega system
    w1_vega, w2_vega = wise_ab_vega(w1, w2)

    # Define W1 - W2 color
    w1w2_vega = w1_vega - w2_vega

    # Set the selection parameters for the reliability-optimized selection criteria (Equation 4 of Assef+18)
    if reliability in (75, 90):
        alpha = 0.650 if reliability == 90 else 0.486
        beta = 0.153 if reliability == 90 else 0.092
        gamma = 13.86 if reliability == 90 else 13.07
    else:
        raise ValueError('Reliability parameter must be either 75 or 90.')

    # Apply reliability-optimized selection criteria (Equation 4 of Assef+18)
    agn_assef18 = ((((w1w2_vega > alpha * np.exp(beta * (w2_vega - gamma) ** 2)) & (w2_vega > gamma)) |
                   ((w1w2_vega > alpha) & w2_vega <= gamma)) &
                   w1w2_avail)

    # Define non-AGN as the inverse selection
    non_agn_assef18 = w1w2_avail & (~agn_assef18)

    return w1w2_avail, agn_assef18, non_agn_assef18


def wise_assef18_c(input_table: Table, snr: int | float = 3, completeness: Literal[75, 90] = 75,
                   mask: MaskedColumn = None) -> (tuple[NDArray[bool], NDArray[bool], NDArray[bool]]):
    """Reliability-Optimized WISE color-color selection originally by [Assef18]_.

    Specifically, this selection implements the reliability-optimized criteria found in [Assef18]_ Equation 4.
    The completeness-optimized selection criteria (Equation 5 of [Assef18]_) is implemented by
    :func:`wise_assef18_c`.
    This selection improves upon the earlier selection in [Assef13]_.

    The selection criterion is given by Equation 5 of [Assef18]_.
    :math:`W1 - W2 > \delta_C`
    where :math:`\delta_C` is dependent on the completeness fraction given. For a completeness of 90%
    :math:`\delta_{C90} = 0.50` while for a completeness of 75%, :math:`\delta_{C75} = 0.71`.

    Notes:
        If using these diagnostic functions, please ref Mar_&_Steph_2025 and add appropriate references given below.

        If using DESI, please reference Summary_ref_2025 and the appropriate photometry catalog
        (e.g., Tractor or Photometry VAC).

        .. warning:: Note of caution: The points outside the AGN region may still include a significant fraction of
            AGN and are best considered as "uncertain" rather than "star forming" or "non-AGN".

    See Also:
        :func:`wise_assef18_r` for the reliability-optimized selection criteria version.

    Args:
        input_table: Table including WISE fluxes and inverse variance. At minimum, catalog must contain columns for W1
            and W2 photometry.
        snr: Signal to noise cut applied to WISE photometry. Default is ``3``.
        completeness: The completeness percent threshold to use for the selection. Must be either ``75`` or ``90``, the
            default is ``75``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Tuple of arrays of same dimension as rows in ``input_table`` which include flags for
        ``w1w2_avail``, ``agn_assef18``, and ``non_agn_assef18``.

    Raises:
        ValueError: If the completeness parameter is not ``75`` or ``90``.

    .. [Assef13] 2013ApJ...772...26A
    .. [Assef18] 2018ApJS..234...23A
    """

    # Mask for zero fluxes
    zero_flux_wise = (input_table['FLUX_W1'] == 0) | (input_table['FLUX_W2'] == 0)
    if mask is not None:
        # Mask for flux availability - included if input_table photometry is missing/masked
        zero_flux_wise |= mask

    # If ivar=0 set it to NaN to avoid infinities when computing the error:
    input_table['FLUX_IVAR_W1'] = np.where(input_table['FLUX_IVAR_W1'] == 0, np.nan, input_table['FLUX_IVAR_W1'])
    input_table['FLUX_IVAR_W2'] = np.where(input_table['FLUX_IVAR_W2'] == 0, np.nan, input_table['FLUX_IVAR_W2'])

    # Mask for SNR.
    snr_w1 = input_table['FLUX_W1'] * np.sqrt(input_table['FLUX_IVAR_W1'])
    snr_w2 = input_table['FLUX_W2'] * np.sqrt(input_table['FLUX_IVAR_W2'])

    ## IR diagnostic based on W1-W2-W3 is available if flux is not zero
    w1w2_avail = (~zero_flux_wise) & (snr_w1 > snr) & (snr_w2 > snr)

    # Convert fluxes to AB magnitudes
    w1 = 22.5 - 2.5 * np.log10(input_table['FLUX_W1'])
    w2 = 22.5 - 2.5 * np.log10(input_table['FLUX_W2'])

    # Convert the magnitudes from AB to Vega system
    w1_vega, w2_vega = wise_ab_vega(w1, w2)

    # Define W1 - W2 color
    w1w2_vega = w1_vega - w2_vega

    # Set the selection parameters for the completeness-optimized selection criteria (Equation 5 of Assef+18)
    if completeness in (75, 90):
        delta = 0.71 if completeness == 75 else 0.5
    else:
        raise ValueError('Completeness parameter must be either 75 or 90.')

    agn_assef18 = (w1w2_vega > delta) & w1w2_avail

    # Define non-AGN as the inverse selection
    non_agn_assef18 = w1w2_avail & (~agn_assef18)

    return w1w2_avail, agn_assef18, non_agn_assef18


def wise_mateos12(input_table: Table, snr: int | float = 3, mask: MaskedColumn = None) -> (
        tuple[NDArray[bool], NDArray[bool], NDArray[bool]]):
    r"""WISE power-law locus AGN selection originally by [Mateos12]_.

    This implements the three-band WISE power-law locus method described by Equations 1 and 2 of [Mateos12]_.
    :math:`y = 0.315 x`
    where :math:`x = \log_{10}\left(\frac{f_{W3}}{f_{W2}}\right)` and
    :math:`y = \log_{10}\left(\frac{f_{W2}}{f_{W1}}\right)`.
    The upper and lower bounds on the wedge defined by the power-law locus are obtained by adding `y`-axis intercepts of
    :math:`+0.297` and :math:`-0.110` respectively. The power-law with :math:`\alpha = -0.3` corresponding to the
    lower-left boundary is given by
    :math:`y = -3.172 x + 0.436`.

    Notes:

        .. note:: [Mateos12]_ also describes a four-band WISE power-law locus selection, but we do not implement this
        due to the low detection rates in the W4 band.

        If using these diagnostic functions, please ref Mar_&_Steph_2025 and add appropriate references given below.

        If using DESI, please reference Summary_ref_2025 and the appropriate photometry catalog
        (e.g., Tractor or Photometry VAC).

        .. warning:: Note of caution: The points outside the AGN region may still include a significant fraction of
            AGN and are best considered as "uncertain" rather than "star forming" or "non-AGN".

    Args:
        input_table: Table including WISE fluxes and inverse variances. Table must include W1, W2, and W3 photometry.
        snr: Signal to noise ratio cut applied to WISE photometry. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Tuple of arrays of same dimension as rows in ``input_table`` which include flags for
        ``w1w2w3_avail``, ``agn_mateos12``, and ``non_agn_mateos12``.

    .. [Mateos12] 2012MNRAS.426.3271M
    """

    # Mask for zero fluxes
    zero_flux_wise = (input_table['FLUX_W1'] == 0) | (input_table['FLUX_W2'] == 0)
    zero_flux_w3 = input_table['FLUX_W3'] == 0
    if mask is not None:
        # Mask for flux availability - included if input_table photometry is missing/masked
        zero_flux_wise |= mask
        zero_flux_w3 |= mask

    # If ivar=0 set it to NaN to avoid infinities when computing the error:
    input_table['FLUX_IVAR_W1'] = np.where(input_table['FLUX_IVAR_W1'] == 0, np.nan, input_table['FLUX_IVAR_W1'])
    input_table['FLUX_IVAR_W2'] = np.where(input_table['FLUX_IVAR_W2'] == 0, np.nan, input_table['FLUX_IVAR_W2'])
    input_table['FLUX_IVAR_W3'] = np.where(input_table['FLUX_IVAR_W3'] == 0, np.nan, input_table['FLUX_IVAR_W3'])

    # Mask for SNR.
    snr_w1 = input_table['FLUX_W1'] * np.sqrt(input_table['FLUX_IVAR_W1'])
    snr_w2 = input_table['FLUX_W2'] * np.sqrt(input_table['FLUX_IVAR_W2'])
    snr_w3 = input_table['FLUX_W3'] * np.sqrt(input_table['FLUX_IVAR_W3'])

    ## IR diagnostic based on W1-W2-W3 is available if flux is not zero
    w1w2_avail = (~zero_flux_wise) & (snr_w1 > snr) & (snr_w2 > snr)
    w2w3_avail = (~zero_flux_wise) & (~zero_flux_w3) & (snr_w2 > snr) & (snr_w3 > snr)

    # Set availability of all three bands
    w1w2w3_avail = w1w2_avail & w2w3_avail

    # Define the flux ratios
    w2w1_flux = np.log10(input_table['FLUX_W2'] / input_table['FLUX_W1'])
    w3w3_flux = np.log10(input_table['FLUX_W3'] / input_table['FLUX_W2'])

    # Define the bounding box of the power-law locus region.
    upper_bound = w2w1_flux < 0.315 * w3w3_flux + 0.297  # Eq. 1 + offset
    lower_bound = w2w1_flux > 0.315 * w3w3_flux - 0.110  # Eq. 1 + offset
    power_law_bound = w2w1_flux > -3.172 * w3w3_flux + 0.436  # Eq. 2

    agn_mateos12 = upper_bound & lower_bound & power_law_bound & w1w2w3_avail

    # Define non-AGN as the inverse selection
    non_agn_mateos12 = w1w2w3_avail & (~agn_mateos12)

    return w1w2w3_avail, agn_mateos12, non_agn_mateos12


def wise_ab_vega(w1: NDArray[float], w2: NDArray[float], w3: NDArray[float] = None) -> (
        tuple[NDArray[float], NDArray[float]] | tuple[NDArray[float], NDArray[float], NDArray[float]]):
    """Utility function to convert AB magnitudes to Vega.

    Args:
        w1: Magnitude of W1 band in AB units.
        w2: Magnitude of W2 band in AB units.
        w3: Magnitude of W3 band in AB units. Optional.

    Returns:
        Tuple of arrays of the WISE magnitudes in Vega units. ``W1`` and ``W2`` are always returned, if ``W3`` is
        included as input, the converted version is returned as well.
    """
    # Conversions found from Explanatory Supplement to the WISE All-Sky Data Release Products
    # Cutri et al. (https://irsa.ipac.caltech.edu/data/WISE/docs/release/All-Sky/expsup/sec4_4h.html)
    w1_ab_vega_offset = 2.699
    w2_ab_vega_offset = 3.339
    w3_ab_vega_offset = 5.174

    w1_vega = w1 - w1_ab_vega_offset
    w2_vega = w2 - w2_ab_vega_offset

    if w3 is not None:
        w3_vega = w3 - w3_ab_vega_offset

        return w1_vega, w2_vega, w3_vega

    return w1_vega, w2_vega
