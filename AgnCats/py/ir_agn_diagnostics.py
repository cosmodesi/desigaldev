"""
ir_agn_diagnostics.py
Author: Benjamin Floyd

Library containing all infrared AGN/galaxy diagnostic functions.

Original version of code written by:
Becky Canning (University of Portsmouth)
Stephanie Juneau (NOIRlab)
Mar Mezcula (Institut de Ciencies de l'Espai)
"""

import  numpy as np
from astropy.table import MaskedColumn, Table
from numpy.typing import NDArray


def wise_stern12(input_table: Table, snr: float | int = 3, mask: MaskedColumn = None) -> tuple[NDArray[bool],
NDArray[bool], NDArray[bool]]:
    r"""WISE Color AGN selection originally by [Stern12]_

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
        snr: SNR cut applied to WISE photometry. Default is ``3``.
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
    # Conversions found from Explanatory Supplement to the WISE All-Sky Data Release Products
    # Cutri et al. (https://irsa.ipac.caltech.edu/data/WISE/docs/release/All-Sky/expsup/sec4_4h.html)
    w1_ab_vega_offset = 2.699
    w2_ab_vega_offset = 3.339

    w1_vega = w1 - w1_ab_vega_offset
    w2_vega = w2 - w2_ab_vega_offset
    w1w2_vega = w1_vega - w2_vega

    # Stern et al. (2012) cut is just along W1 - W2 color
    agn_stern12: NDArray[bool] = (w1w2_vega > 0.8) & w1w2_avail
    sf_stern: NDArray[bool] = (~agn_stern12)

    return w1w2_avail, agn_stern12, sf_stern


