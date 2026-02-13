"""
util.py
Author: Benjamin Floyd

This library collects miscellaneous utility functions that are useful for DESI AGN/Galaxy Classification VAC.
"""
import numpy as np
from numpy.typing import NDArray


def plot_nii_bpt_lines(x_axes: NDArray[float]) -> tuple[NDArray[float], NDArray[float], NDArray[float]]:
    r"""This function draws the lines for the BPT regions in the [NII] BPT plot

    BPT regions are defined as:
        [Kew01]_ Kewley et al. (2001): Starburst vs AGN classification.
            ``kew01_nii``:
            :math:`\log_10(flux_{[OIII]_\lambda5006} / flux_{H\beta}) =
            \frac{0.61}{\log_10(flux_{[NII]_\lambda6583} / flux_{H\alpha}) - 0.47} + 1.19`

        [Ka03]_ Kauffmann et al. (2003): Starburst vs composite classification.
            ``ka03``:
            :math:`\log_10(flux_{[OIII]_\lambda5006} / flux_{H\beta}) =
            \frac{0.61}{\log_10(flux_{[NII]_\lambda6583} / flux_{H\alpha}) - 0.05} + 1.3`

        [Scha07]_ Schawinski et al. (2007): Seyferts vs LINERs
            ``scha07``:
            :math:`\log_10(flux_{[OIII]_\lambda5006} / flux_{H\beta}) =
            1.05 * \log_10(flux_{[NII]_\lambda6583} / flux_{H\alpha}) + 0.45`

    Other BPT regions not implemented here:
        [Law21]_ Law et al. 2021: Proposed revised lines based on MaNGA observation (not implemented because similar
        to [Ka03]_):
            :math:`\log_10(flux_{[OIII]_\lambda5006} / flux_{H\beta}) =
            \frac{0.438}{\log_10(flux_{[NII]_\lambda6583} / flux_{H\alpha}) + 0.023} + 1.222`

        Law et al. define an extra "intermediate" region (not yet implemented)

    Args:
        x_axes: Array over range of values to plot. Should correspond to the :math:`\log_10([NII] / H\alpha) line flux
            ratio.

    Returns:
        Tuple of three arrays ``kew01_nii``, ``ka03``, ``scha07`` corresponding to the Kewley+01, Kauffmann+03, and
        Schawinski+07 selection lines respectively.

    .. [Law21] 2021ApJ...915...35L
    .. [Ka03]  2003MNRAS.346.1055K
    .. [Kew01] 2001ApJ...556..121K
    .. [Scha07] 2007MNRAS.382.1415S
    """

    kew01_nii = 0.61 / (x_axes - 0.47) + 1.19
    kew01_nii[x_axes >= 0.47] = np.nan

    ka03 = 0.61 / (x_axes - 0.05) + 1.3
    ka03[x_axes >= 0.05] = np.nan

    scha07 = 1.05 * x_axes + 0.45
    scha07[scha07 < kew01_nii] = np.nan

    return kew01_nii, ka03, scha07


def decode_bitmask(maskbit_num: int) -> NDArray[int]:
    """Simple utility function that when given a maskbit number, will decode

    Args:
        maskbit_num:

    Returns:

    """
    powers = []
    i = 1
    while i <= maskbit_num:
        if i & maskbit_num:  # Checks to see if ``i`` is "in" num
            powers.append(i)
        i <<= 1  # Left bitwise shift. Essentially iterate the binary number up by one bit. 1 -> 2 -> 4 -> ...
    return np.array(powers)
