"""
AGNdiagnosticsFunctionsDESI.py

Library containing all AGN/Galaxy diagnostic functions used in the DESI AGN/Galaxy Classification VAC.

Original Authors:
Becky Canning (University of Portsmouth)
Stephanie Juneau (NOIRlab)
Mar Mezcula (Institut de Ciencies de l'Espai)

Revised by:
Benjamin Floyd (University of Portsmouth)
"""

import numpy as np
from astropy.table import MaskedColumn, Table
from numpy.typing import NDArray


# notes for us:
# Find/replace: Mar_&_Steph_2025 with correct reference
# Find/replace: Summary_ref_2025 with correct reference
# Find/replace: FastSpecFit_ref with correct reference

def broad_line(input_table: Table, snr: int | float = 3, mask: MaskedColumn = None, vel_thresh: float = 1200.) -> (
        NDArray[bool]):
    r"""Assigns ``BROAD_LINE`` bitmask to object.

    This function will assign the ``BROAD_LINE`` bitmask to any object that has a FWHM of at least the value defined by
    ``vel_thresh`` in km/s for *any* of the following lines: :math:`H\alpha`, :math:`H\beta`, Mg II], C IV.

    Notes:
        If using these diagnostic functions please ref the appropriate references given below.

        If using DESI please reference Summary_ref_2023 and the appropriate emission line catalog
        (e.g. FastSpecFit ref FastSpecFit_ref)

    Args:
        input_table: Table including H⍺, Hβ, MgII], and CIV emission lines (fluxes and widths).
        snr: The Signal-to-noise cut applied to all axes. Default is ``3``.
        mask: Optional mask (e.g., from the masked column array). Default is ``None``.
        vel_thresh: Velocity threshold definition in km/s for broad line of FWHM in H⍺, Hβ,
            Mg II] and/or C IV line. Default is ``1200.`` km/s.

    Returns:
        Vectors of same dimension as rows in ``input_table`` which include flags for ``broad_line``
    """

    # Mask for zero fluxes when NONE of the lines are available
    zero_flux = ((input_table['HALPHA_BROAD_FLUX'] == 0) &
                 (input_table['HBETA_BROAD_FLUX'] == 0) &
                 (input_table['MGII_2796_FLUX'] == 0) &
                 (input_table['MGII_2803_FLUX'] == 0) &
                 (input_table['CIV_1549_FLUX'] == 0))
    if mask is not None:
        # Mask for flux availability - included as fastspecfit columns are MaskedColumn data
        zero_flux |= mask

    # If ivar = 0 set it to NaN to avoid infinities when computing the error:
    input_table['HALPHA_BROAD_FLUX_IVAR'] = np.where(input_table['HALPHA_BROAD_FLUX_IVAR'] == 0,
                                                     np.nan, input_table['HALPHA_BROAD_FLUX_IVAR'])
    input_table['HBETA_BROAD_FLUX_IVAR'] = np.where(input_table['HBETA_BROAD_FLUX_IVAR'] == 0,
                                                    np.nan, input_table['HBETA_BROAD_FLUX_IVAR'])
    input_table['MGII_2796_FLUX_IVAR'] = np.where(input_table['MGII_2796_FLUX_IVAR'] == 0,
                                                  np.nan, input_table['MGII_2796_FLUX_IVAR'])
    input_table['MGII_2803_FLUX_IVAR'] = np.where(input_table['MGII_2803_FLUX_IVAR'] == 0,
                                                  np.nan, input_table['MGII_2803_FLUX_IVAR'])
    input_table['CIV_1549_FLUX_IVAR'] = np.where(input_table['CIV_1549_FLUX_IVAR'] == 0,
                                                 np.nan, input_table['CIV_1549_FLUX_IVAR'])

    # Broad components for Balmer lines
    snr_ha = input_table['HALPHA_BROAD_FLUX'] * np.sqrt(input_table['HALPHA_BROAD_FLUX_IVAR'])
    snr_hb = input_table['HBETA_BROAD_FLUX'] * np.sqrt(input_table['HBETA_BROAD_FLUX_IVAR'])

    # For MgII, sum the doublet
    mgii_flux = input_table['MGII_2796_FLUX'] + input_table['MGII_2803_FLUX']
    mgii_flux_ivar = 1. / (1. / input_table['MGII_2796_FLUX_IVAR'] + 1. / input_table['MGII_2803_FLUX_IVAR'])
    snr_mgii = mgii_flux * np.sqrt(mgii_flux_ivar)

    # CIV
    snr_civ = input_table['CIV_1549_FLUX'] * np.sqrt(input_table['CIV_1549_FLUX_IVAR'])

    # Factor to convert from Gaussian sigma to FWHM
    sig2fwhm = 2. * np.sqrt(2. * np.log(2.))

    # Define breadth in FWHM in km/s
    broad_fwhm_ha = input_table['HALPHA_BROAD_SIGMA'] * sig2fwhm
    broad_fwhm_hb = input_table['HBETA_BROAD_SIGMA'] * sig2fwhm
    broad_fwhm_mgii_2796 = input_table['MGII_2796_SIGMA'] * sig2fwhm
    broad_fwhm_mgii_2803 = input_table['MGII_2803_SIGMA'] * sig2fwhm  # TODO: BenFloyd - Not used.
    broad_fwhm_civ = input_table['CIV_1549_SIGMA'] * sig2fwhm

    # Check for each line separately first
    is_broad_ha = (snr_ha >= snr) & (broad_fwhm_ha >= vel_thresh) & (~zero_flux)
    is_broad_hb = (snr_hb >= snr) & (broad_fwhm_hb >= vel_thresh) & (~zero_flux)
    is_broad_mgii = (snr_mgii >= snr) & (broad_fwhm_mgii_2796 >= vel_thresh) & (~zero_flux)
    is_broad_civ = (snr_civ >= snr) & (broad_fwhm_civ >= vel_thresh) & (~zero_flux)

    # Decision: flag a BL if any of the 4 lines meet the criteria
    is_broad = is_broad_ha | is_broad_hb | is_broad_mgii | is_broad_civ

    return is_broad


def nii_bpt(input_table: Table, snr: int | float = 3, mask: MaskedColumn = None) -> (
        tuple[NDArray[bool], NDArray[bool], NDArray[bool], NDArray[bool], NDArray[bool]]):
    r"""[NII] BPT diagnostic originally from [BPT81]_

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
            ``scha07_nii``:
            :math:`\log_10(flux_{[OIII]_\lambda5006} / flux_{H\beta}) =
            1.05 * \log_10(flux_{[NII]_\lambda6583} / flux_{H\alpha}) + 0.45`

    Other BPT regions not implemented here:
        [Law21]_ Law et al. 2021: Proposed revised lines based on MaNGA observation (not implemented because similar to [Ka03]_):
            :math:`\log_10(flux_{[OIII]_\lambda5006} / flux_{H\beta}) =
            \frac{0.438}{\log_10(flux_{[NII]_\lambda6583} / flux_{H\alpha}) + 0.023} + 1.222`

        Law et al. define an extra "intermediate" region (not yet implemented)

    Notes:
        If using these diagnostic functions please ref the appropriate references given below.

        If using DESI please reference Summary_ref_2023 and the appropriate emission line catalog
        (e.g. FastSpecFit ref FastSpecFit_ref)

        On Output Classifications:
            ``nii_bpt_avail``: Flag for SNR in all lines higher than ``snr`` and no zero fluxes.

            ``sf_nii``: Flag for SNR > ``snr`` and not in any of ``agn_nii`` or ``liner_nii`` or ``composite_nii``.

            ``agn_nii``: Flag for a Kew01 AGN and Scha07 Seyfert.
                Flag SNR > ``snr`` & [ log(OIII / Hb) >= ``kew01_nii`` & log(OIII / Hb) > ``scha07`` | log(NII / Ha) >= 0.47 ]

            ``liner_nii``: Flag for a Kew01 AGN and Scha07 LINER.
                Flag SNR > ``snr`` & [ log(OIII / Hb) >= ``kew01_nii`` & log(OIII / Hb) < ``scha07`` | log(NII / Ha) >= 0.47 ]

            ``composite_nii``: Flag for a ka03 composite but not an ``agn_nii``.
                Flag SNR > ``snr`` & not ``agn_nii`` & [ log(OIII / Hb) >= ``ka03`` | log(NII / Ha) >= 0.05 ]

    Args:
        input_table: Table including [NII], H⍺, [OIII], Hβ fluxes and associated inverse variances.
        snr: Signal-to-noise cut applied to all axes. Default is 3.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Tuple of arrays of same dimension as rows in ``input_table`` which include flags for
         ``nii_bpt_avail``, ``sf_nii``, ``agn_nii``, ``liner_nii``, ``composite_nii``.
         See note for more information on the definitions of these flags.

    .. [BPT81] 1981PASP...93....5B
    .. [Law21] 2021ApJ...915...35L
    .. [Ka03]  2003MNRAS.346.1055K
    .. [Kew01] 2001ApJ...556..121K
    .. [Scha07] 2007MNRAS.382.1415S
    """

    # Mask for zero fluxes
    zero_flux_nii = ((input_table['HALPHA_FLUX'] == 0)
                     | (input_table['HBETA_FLUX'] == 0)
                     | (input_table['OIII_5007_FLUX'] == 0)
                     | (input_table['NII_6584_FLUX'] == 0))
    if mask is not None:
        # Mask for flux availability - included as fastspecfit columns are MaskedColumn data
        zero_flux_nii |= mask

    # If ivar=0 set it to NaN to avoid infinites when computing the error:
    input_table['HALPHA_FLUX_IVAR'] = np.where(input_table['HALPHA_FLUX_IVAR'] == 0,
                                               np.nan, input_table['HALPHA_FLUX_IVAR'])
    input_table['HBETA_FLUX_IVAR'] = np.where(input_table['HBETA_FLUX_IVAR'] == 0,
                                              np.nan, input_table['HBETA_FLUX_IVAR'])
    input_table['OIII_5007_FLUX_IVAR'] = np.where(input_table['OIII_5007_FLUX_IVAR'] == 0,
                                                  np.nan, input_table['OIII_5007_FLUX_IVAR'])
    input_table['NII_6584_FLUX_IVAR'] = np.where(input_table['NII_6584_FLUX_IVAR'] == 0,
                                                 np.nan, input_table['NII_6584_FLUX_IVAR'])

    # Mask for SNR. Default is NII-BPT is available if all SNR >= 3
    SNR_Ha = input_table['HALPHA_FLUX'] * np.sqrt(input_table['HALPHA_FLUX_IVAR'])
    SNR_Hb = input_table['HBETA_FLUX'] * np.sqrt(input_table['HBETA_FLUX_IVAR'])
    SNR_OIII = input_table['OIII_5007_FLUX'] * np.sqrt(input_table['OIII_5007_FLUX_IVAR'])
    SNR_NII = input_table['NII_6584_FLUX'] * np.sqrt(input_table['NII_6584_FLUX_IVAR'])

    # Define regions
    log_nii_ha = np.log10(input_table['NII_6584_FLUX'] / input_table['HALPHA_FLUX'])
    log_oiii_hb = np.log10(input_table['OIII_5007_FLUX'] / input_table['HBETA_FLUX'])
    kew01_nii = 0.61 / (log_nii_ha - 0.47) + 1.19
    scha07 = 1.05 * log_nii_ha + 0.45
    ka03 = 0.61 / (log_nii_ha - 0.05) + 1.3

    ## NII-BPT is available (All lines SNR >= 3)
    nii_bpt_avail = (SNR_Ha >= snr) & (SNR_Hb >= snr) & (SNR_OIII >= snr) & (SNR_NII >= snr) & (~zero_flux_nii)

    ## NII-AGN, LINER, COMP, SF
    agnliner_nii = (nii_bpt_avail) & ((log_oiii_hb >= kew01_nii) | (log_nii_ha >= 0.47))
    agn_nii = (agnliner_nii) & (log_oiii_hb >= scha07)
    liner_nii = (agnliner_nii) & (log_oiii_hb < scha07)
    composite_nii = (nii_bpt_avail) & ((log_oiii_hb >= ka03) | (log_nii_ha >= 0.05)) & (~agnliner_nii)
    sf_nii = (nii_bpt_avail) & (~agnliner_nii) & (~composite_nii)

    return nii_bpt_avail, sf_nii, agn_nii, liner_nii, composite_nii

# TODO: BenFloyd - This should be moved to a util library to keep this library focused on just diagnostics
def NII_BPT_lines(x_axes):
    '''
    This function draws the lines for the BPT regions int he NII_BPT plot
    
    Kewley et al. 2001: starburst vs AGN classification.
    Kew01_nii: log10(flux_oiii_5006/flux_hbeta)=0.61/(log10(flux_nii_6583/flux_halpha)-0.47)+1.19

    Kauffmann et al. 2003: starburst vs composites.
    Ka03: log10(flux_oiii_5006/flux_hbeta)=0.61/(log10(flux_nii_6583/flux_halpha)-0.05)+1.3
    
    Schawinsky et al. 2007: Seyferts vs LINERS
    Scha07: log10(flux_oiii_5006/flux_hbeta)=1.05*log10(flux_nii_6583/flux_halpha)+0.45
    
    Other BPT regions not implemented here yet:
    
    Law et al. 2021 proposed revised lines based on MaNGA observation (not implemented b/c similar to Ka03):
    log10(flux_oiii_5006/flux_hbeta)=0.438/(log10(flux_nii_6583/flux_halpha)+0.023)+1.222
    
    Law et al. define an extra "intermediate" region (not yet implemented)
    '''
    Kew01_nii = 0.61 / (x_axes - 0.47) + 1.19
    n = np.where(x_axes >= 0.47)
    Kew01_nii[n] = np.nan

    Ka03 = 0.61 / (x_axes - 0.05) + 1.3
    n = np.where(x_axes >= 0.05)
    Ka03[n] = np.nan

    Scha07 = 1.05 * x_axes + 0.45
    n = np.where(Scha07 < Kew01_nii)
    Scha07[n] = np.nan
    return Kew01_nii, Ka03, Scha07


def sii_bpt(input_table: Table, snr: int | float = 3, kewley01: bool = False, mask: MaskedColumn = None) -> (
        tuple[NDArray[bool], NDArray[bool], NDArray[bool], NDArray[bool]]):
    r"""[SII] BPT diagnostic originally from [VO87]_.

    By default, here we use the [Law21]_ line for SF/AGN separation and the [Kew06]_ line for LINER/Seyfert separation
    on the AGN side. Optionally, can set :code:`kewley01=True` to use the [Kew01]_ line instead of [Law21]_.

    BPT regions defined as:
        [Law21]_ Law et al. 2021: Proposed revised lines based on MaNGA observation.
            ``law21_sii``:
            :math:`\log_10(flux_{[OIII]_\lambda5006} / flux_{H\beta}) =
            \frac{0.648}{\log_10(flux_{[SII]_\lambda\lambda6716,6731} / flux_{H\alpha}) - 0.324} + 1.349`

        [Kew06]_ Kewley et al. 2006: Seyferts vs LINERs
            ``kew06_sii``:
            :math:`\log_10(flux_{[OIII]_\lambda5006} / flux_{H\beta}) =
            1.89 \log_10(flux_{[SII]_\lambda\lambda6716,6731} / flux_{H\alpha}) + 0.76`

    Optional BPT region definition:
        [Kew01]_ Kewley et al. 2001: Starburst vs AGN classification. Solid lines in BPT
            ``kew01_sii``:
            :math:`\log_10(flux_{[OIII]_\lambda5006} / flux_{H\beta}) =
            \frac{0.72}{\log_10(flux_{[SII]_\lambda\lambda6716,6731}/flux_{H\alpha}) - 0.32} + 1.30`

    Other BPT regions not implemented here:
        Law et al. define an extra "intermediate" region (not yet implemented)

    Notes:
        If using these diagnostic functions please ref the appropriate references given below.

        If using DESI please reference Summary_ref_2023 and the appropriate emission line catalog
        (e.g. FastSpecFit ref FastSpecFit_ref)

    Args:
        input_table: Table including [SII], H⍺, [OIII], Hβ fluxes and inverse variances.
        snr: The SNR cut applied to all axes. Default is ``3``.
        kewley01: Optional flag to use Kewley+01 lines for SF/AGN classification instead of Law+21 lines.
            Default is ``False``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Tuple of arrays of same dimension as rows in ``input_table`` which include flags for
        ``sii_bpt_avail``, ``sf_sii``, ``agn_sii``, ``liner_sii``.

    .. [Law21] 2021ApJ...915...35L
    .. [Kew01] 2001ApJ...556..121K
    .. [Kew06] 2006MNRAS.372..961K
    .. [VO87] 1987ApJS...63..295V
    """

    # Mask for zero fluxes
    zero_flux_sii = ((input_table['HALPHA_FLUX'] == 0)
                     | (input_table['HBETA_FLUX'] == 0)
                     | (input_table['OIII_5007_FLUX'] == 0)
                     | (input_table['SII_6716_FLUX'] + input_table['SII_6731_FLUX'] == 0))
    if mask is not None:
        # Mask for flux availability - included as fastspecfit columns are MaskedColumn data
        zero_flux_sii |= mask

        # If ivar=0 set it to NaN to avoid infinites when computing the error:
    input_table['HALPHA_FLUX_IVAR'] = np.where(input_table['HALPHA_FLUX_IVAR'] == 0,
                                               np.nan, input_table['HALPHA_FLUX_IVAR'])
    input_table['HBETA_FLUX_IVAR'] = np.where(input_table['HBETA_FLUX_IVAR'] == 0,
                                              np.nan, input_table['HBETA_FLUX_IVAR'])
    input_table['OIII_5007_FLUX_IVAR'] = np.where(input_table['OIII_5007_FLUX_IVAR'] == 0,
                                                  np.nan, input_table['OIII_5007_FLUX_IVAR'])
    input_table['SII_6716_FLUX_IVAR'] = np.where(input_table['SII_6716_FLUX_IVAR'] == 0,
                                                 np.nan, input_table['SII_6716_FLUX_IVAR'])
    input_table['SII_6731_FLUX_IVAR'] = np.where(input_table['SII_6731_FLUX_IVAR'] == 0,
                                                 np.nan, input_table['SII_6731_FLUX_IVAR'])
    sii_flux = input_table['SII_6716_FLUX'] + input_table['SII_6731_FLUX']
    sii_flux_ivar = 1 / (1 / input_table['SII_6716_FLUX_IVAR'] + 1 / input_table['SII_6731_FLUX_IVAR'])

    # Mask for SNR. Default is SII-BPT is available if all SNR >= 3
    snr_ha = input_table['HALPHA_FLUX'] * np.sqrt(input_table['HALPHA_FLUX_IVAR'])
    snr_hb = input_table['HBETA_FLUX'] * np.sqrt(input_table['HBETA_FLUX_IVAR'])
    snr_oiii = input_table['OIII_5007_FLUX'] * np.sqrt(input_table['OIII_5007_FLUX_IVAR'])
    snr_sii = sii_flux * np.sqrt(sii_flux_ivar)

    # Define regions    
    log_sii_ha = np.log10((input_table['SII_6716_FLUX'] + input_table['SII_6731_FLUX']) / input_table['HALPHA_FLUX'])
    log_oiii_hb = np.log10(input_table['OIII_5007_FLUX'] / input_table['HBETA_FLUX'])
    kew01_sii = 0.72 / (log_sii_ha - 0.32) + 1.30
    kew06_sii = 1.89 * log_sii_ha + 0.76
    law21_sii = 0.648 / (log_sii_ha - 0.324) + 1.43  # modified (+1.349 was original)
    if kewley01:
        line_sii = kew01_sii
    else:
        line_sii = law21_sii

    ## SII-BPT is available (All lines SNR >= 3)
    sii_bpt_avail = (snr_ha >= snr) & (snr_hb >= snr) & (snr_oiii >= snr) & (snr_sii >= snr) & (~zero_flux_sii)

    ## SII-AGN, LINER, SF
    agnliner_sii = sii_bpt_avail & ((log_oiii_hb >= line_sii) | (log_sii_ha >= 0.32))
    agn_sii = agnliner_sii & (log_oiii_hb >= kew06_sii)
    liner_sii = agnliner_sii & (log_oiii_hb < kew06_sii)
    sf_sii = sii_bpt_avail & (~agnliner_sii)

    return sii_bpt_avail, sf_sii, agn_sii, liner_sii


def oi_bpt(input_table: Table, snr: int | float = 3, snr_oi: int | float = 1, kewley01: bool = False,
           mask: MaskedColumn = None) -> tuple[NDArray[bool], NDArray[bool], NDArray[bool], NDArray[bool]]:
    r"""[OI] diagnostic originally from [VO87]_.

    By default, here we use the [Law21]_ line for SF/AGN separation and the [Kew06]_ line for LINER/Seyfert separation
    on the AGN side. Optionally, can set :code:`kewley01=True` to use the [Kew01]_ line instead of [Law21]_.

    BPT regions defined as:
        [Law21]_ Law et al. 2021: By default, use the Law+21 line for SF/AGN separation
            ``law21_oi``:
            :math:`\log_10(flux_{[OIII]_\lambda5006} / flux_{H\beta}) =
            \frac{0.884}{\log_10(flux_{[OI]_\lambda6300} / flux_{H\alpha}) + 0.124} + 1.291`

        [Kew06]_ Kewley et al. 2006: By default Kewley+06 line for LINER/Seyfert separation on the AGN side
            ``kew06_oi``:
            :math:`\log_10(flux_{[OIII]_5006} / flux_{H\beta}) =
            1.18 \log_10(flux_{[OI]_\lambda6300} / flux_{H\alpha}) + 1.30`

    Optional BPT region definition:
        [Kew01]_ Kewley et al. 2001: starburst vs AGN classification. Solid lines in BPT
            ``kew01_oi``:
            :math:`\log_10(flux_{[OIII]_\lambda5006} / flux_{H\beta}) =
            \frac{0.73}{\log_10(flux_{[OI]_\lambda6300} / flux_{H\alpha}) + 0.59} + 1.33`

    Other BPT regions not implemented here:
        Law et al. define an extra "intermediate" region (not yet implemented)

    Notes:
        If using these diagnostic functions please ref the appropriate references given below.

        If using DESI please reference Summary_ref_2023 and the appropriate emission line catalog
        (e.g. FastSpecFit ref FastSpecFit_ref)

    Args:
        input_table: Table including [O I], H⍺, [OIII], Hβ fluxes and inverse variances.
        snr: SNR cut applied to H⍺, Hβ, and [OIII]. Default is ``3``.
        snr_oi: SNR cut applied to the [OI]λ6300 emission line. Default is ``1``.
        kewley01: Optional flag to use Kewley+01 lines for SF/AGN classification instead of Law+21 lines.
            Default is ``False``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Tuple of arrays of same dimension as rows in ``input_table`` which include flags for
        ``oi_bpt_avail``, ``sf_oi``, ``agn_oi``, ``liner_oi``.

    .. [Law21] 2021ApJ...915...35L
    .. [Kew01] 2001ApJ...556..121K
    .. [Kew06] 2006MNRAS.372..961K
    .. [VO87] 1987ApJS...63..295V
    """

    # Mask for zero fluxes
    zero_flux_oi = ((input_table['HALPHA_FLUX'] == 0)
                    | (input_table['HBETA_FLUX'] == 0)
                    | (input_table['OIII_5007_FLUX'] == 0)
                    | (input_table['OI_6300_FLUX'] == 0))
    if mask is not None:
        # Mask for flux availability - included as fastspecfit columns are MaskedColumn data
        zero_flux_oi |= mask

    # If ivar=0 set it to NaN to avoid infinities when computing the error:
    input_table['HALPHA_FLUX_IVAR'] = np.where(input_table['HALPHA_FLUX_IVAR'] == 0,
                                               np.nan, input_table['HALPHA_FLUX_IVAR'])
    input_table['HBETA_FLUX_IVAR'] = np.where(input_table['HBETA_FLUX_IVAR'] == 0,
                                              np.nan, input_table['HBETA_FLUX_IVAR'])
    input_table['OIII_5007_FLUX_IVAR'] = np.where(input_table['OIII_5007_FLUX_IVAR'] == 0,
                                                  np.nan, input_table['OIII_5007_FLUX_IVAR'])
    input_table['OI_6300_FLUX_IVAR'] = np.where(input_table['OI_6300_FLUX_IVAR'] == 0,
                                                np.nan, input_table['OI_6300_FLUX_IVAR'])

    # Mask for SNR. Default is OI-BPT is available if Ha, Hb, OIII SNR >= 3 and OI SNR >= 1.
    snr_ha = input_table['HALPHA_FLUX'] * np.sqrt(input_table['HALPHA_FLUX_IVAR'])
    snr_hb = input_table['HBETA_FLUX'] * np.sqrt(input_table['HBETA_FLUX_IVAR'])
    snr_oiii = input_table['OIII_5007_FLUX'] * np.sqrt(input_table['OIII_5007_FLUX_IVAR'])
    _snr_oi = input_table['OI_6300_FLUX'] * np.sqrt(input_table['OI_6300_FLUX_IVAR'])

    # Define regions
    log_oi_ha = np.log10(input_table['OI_6300_FLUX'] / input_table['HALPHA_FLUX'])
    log_oiii_hb = np.log10(input_table['OIII_5007_FLUX'] / input_table['HBETA_FLUX'])
    kew01_oi = 0.73 / (log_oi_ha + 0.59) + 1.33
    kew06_oi = 1.18 * log_oi_ha + 1.30
    law21_oi = 0.884 / (log_oi_ha + 0.124) + 1.4  # modified (original was +1.291)
    if kewley01:
        line_oi = kew01_oi
    else:
        line_oi = law21_oi

    ## OI-BPT is available (SNR for the 3 lines other than OI >= 3)
    oi_bpt_avail = (snr_ha >= snr) & (snr_hb >= snr) & (snr_oiii >= snr) & (_snr_oi >= snr_oi) & (~zero_flux_oi)

    ## OI-AGN, LINER, SF
    agnliner_oi = oi_bpt_avail & ((log_oiii_hb >= line_oi) | (log_oi_ha >= -0.59))
    agn_oi = agnliner_oi & (log_oiii_hb >= kew06_oi)
    liner_oi = agnliner_oi & (log_oiii_hb < kew06_oi)
    sf_oi = oi_bpt_avail & (~agnliner_oi)

    return oi_bpt_avail, sf_oi, agn_oi, liner_oi


def whan(input_table: Table, snr: int | float = 3, snr_ew: int | float = 1, mask: MaskedColumn = None) -> (
        tuple[NDArray[bool], NDArray[bool], NDArray[bool], NDArray[bool], NDArray[bool], NDArray[bool]]):
    r"""WHAN diagnostic originally from [CidFer11]_

    WHAN regions defined as:
        [CidFer11]_ Cid Fernandes et al. 2011

        Pure star-forming galaxies:
            ``whan_sf``:
            :math:`\log_10(flux_{[NII]_\lambda6583} / flux_{H\alpha}) < -0.4 \wedge EW_{H\alpha_\lambda6562} > 3\AA`

        Strong AGN (e.g. Seyferts):
            ``whan_sagn``:
            :math:`\log_10(flux_{[NII]_\lambda6583} / flux_{H\alpha}) > -0.4 \wedge EW_{H\alpha_\lambda6562} > 6\AA`

        Weak AGN:
            ``whan_wagn``:
            :math:`\log_10(flux_{[NII]_\lambda6583} / flux_{H\alpha}) > -0.4 \wedge 3\AA < EW_{H\alpha_\lambda6562} < 6\AA`

        Retired galaxies (fake AGN, i.e. galaxies that have stopped forming stars and are ionized by their hot low-mass evolved stars):
            ``whan_retired``:
            :math:`0.5\AA < EW_{H\alpha_\lambda6562} < 3\AA`

        Passive:
            ``whan_passive``:
            :math:`EW_{H\alpha_\lambda6562} < 0.5\AA`

    Notes:
        If using these diagnostic functions please ref Mar_&_Steph_2025
        and the appropriate references given below.

        If using DESI please reference Summary_ref_2025 and the appropriate emission line catalog
        (e.g. FastSpecFit ref FastSpecFit_ref)

    Args:
        input_table: Table including H⍺, [NII] fluxes, H⍺ equivalent width and inverse variances.
        snr: SNR cut applied to all axes. Default is ``3``.
        snr_ew: SNR cut applied to the H⍺ equivalent width. Default is ``1``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Tuple of arrays of same dimension as rows in ``input_table`` which include flags for
        ``whan_avail``, ``whan_sf``, ``whan_sagn``, ``whan_wagn``, ``whan_retired``, ``whan_passive``.

    .. [CidFer11] 2011MNRAS.413.1687C
    """

    # Mask for zero fluxes
    zero_flux_whan = (input_table['HALPHA_FLUX'] == 0) | (input_table['NII_6584_FLUX'] == 0)
    if mask is not None:
        # Mask for flux availability - included as fastspecfit columns are MaskedColumn data
        zero_flux_whan |= mask

    # If ivar=0 set it to NaN to avoid infinities when computing the error:
    input_table['HALPHA_FLUX_IVAR'] = np.where(input_table['HALPHA_FLUX_IVAR'] == 0,
                                               np.nan, input_table['HALPHA_FLUX_IVAR'])
    input_table['NII_6584_FLUX_IVAR'] = np.where(input_table['NII_6584_FLUX_IVAR'] == 0,
                                                 np.nan, input_table['NII_6584_FLUX_IVAR'])

    # Mask for SNR. Default is WHAN is available if Ha, NII SNR >= 3.
    snr_ha = input_table['HALPHA_FLUX'] * np.sqrt(input_table['HALPHA_FLUX_IVAR'])
    snr_nii = input_table['NII_6584_FLUX'] * np.sqrt(input_table['NII_6584_FLUX_IVAR'])
    snr_ha_ew = input_table['HALPHA_EW'] * np.sqrt(input_table['HALPHA_EW_IVAR'])

    # Define regions
    ew_ha_6562 = input_table['HALPHA_EW']
    log_nii_ha = np.log10(input_table['NII_6584_FLUX'] / input_table['HALPHA_FLUX'])

    ## WHAN is available: 
    # - NII and Halpha line flux SNR >= snr (=3 by default) when using the [NII]/Ha ratio
    # - Halpha EW measured at > snr_ew (=1 by default) sigma significance when cutting just on EW
    whan_ew_cut = (snr_ha_ew >= snr_ew) & (~zero_flux_whan)
    whan_flux_cut = (snr_ha >= snr) & (snr_nii >= snr) & (~zero_flux_whan)
    whan_avail = whan_ew_cut | whan_flux_cut

    ## WHAN-SF, strong AGN, weak AGN, retired, passive
    whan_sf = whan_flux_cut & (log_nii_ha < -0.4) & (ew_ha_6562 >= 3)
    whan_sagn = whan_flux_cut & (log_nii_ha >= -0.4) & (ew_ha_6562 >= 6)
    whan_wagn = whan_flux_cut & (log_nii_ha >= -0.4) & (ew_ha_6562 < 6) & (ew_ha_6562 >= 3)
    whan_retired = whan_ew_cut & (ew_ha_6562 < 3) & (ew_ha_6562 >= 0.5)
    whan_passive = whan_ew_cut & ew_ha_6562 < 0.5

    return whan_avail, whan_sf, whan_sagn, whan_wagn, whan_retired, whan_passive


def blue(input_table: Table, snr: int | float = 3, snr_oii: int | float = 3, mask: MaskedColumn = None) -> (
        tuple[NDArray[bool], NDArray[bool], NDArray[bool], NDArray[bool], NDArray[bool], NDArray[bool]]):
    r"""BLUE diagnostic originally from [Lam04]_ and [Lam10]_.

    Blue diagram regions defined as:
        Main division between SF/AGN (Eq. 1 of [Lam10]_):
            :math:`\log_10(flux_{[OIII]_\lambda5006} / flux_{H\beta}) =
            \frac{0.11}{\log_10(EW_{[OII]_\lambda3727} / EW_{H\beta_\lambda4861}) - 0.92} + 0.85`

        Division between SF and "mixed" SF/Sy2 (Eq. 2 of [Lam10]_):
            :math:`\log_10(flux_{[OIII]_\lambda5006} / flux_{H\beta}) > 0.3`

        Divisions for the SF-LIN/Comp overlap region (Eq. 3 of [Lam10]_):
            ``blue1``:
            :math:`\log_10(flux_{[OIII]_\lambda5006} / flux_{H\beta}) =
            - (\log_10(EW_{[OII]_\lambda3727} / EW_{H\beta_\lambda4861}) - 1.0)^2
            - 0.1 \log_10(EW_{[OII]_\lambda3727} / EW_{H\beta_\lambda4861}) + 0.25`

            ``blue2``:
            :math:`\log_10(flux_{[OIII]_\lambda5006} / flux_{H\beta}) =
            \frac{\log_10(EW_{[OII]_\lambda3727}}{EW_{H\beta_\lambda4861}} - 0.2)^2 - 0.6`

            where :math:`y = \log_10(flux_{[OIII]_\lambda5006} / flux_{H\beta})` and :math:`x = \log_10(EW_{[OII]_\lambda3727} / EW_{H\beta_\lambda4861})`

        Division between Sy2/LINER (Eq. 4 of [Lam10]_):
            :math:`\log_10(flux_{[OIII]_\lambda5006} / flux_{H\beta}) =
            0.95 * \log_10(EW_{[OII]_\lambda3727} / EW_{H\beta_\lambda4861}) - 0.4`

    Notes:
        If using these diagnostic functions please ref the appropriate references given below.

        If using DESI please reference Summary_ref_2023 and the appropriate emission line catalog
        (e.g. FastSpecFit ref FastSpecFit_ref)

    Args:
        input_table: Table including [OII], [OIII], and Hβ fluxes, associated inverse variances,
            and the Hβ equivalent width.
        snr: SNR cut applied to the Hβ and [O III] fluxes. Default is ``3``.
        snr_oii: SNR cut applied to the [O II]λ3727 flux. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Tuple of arrays of same dimension as rows in ``input_table`` which include flags for
        ``blue_avail``, ``agn_blue``, ``sflin_blue``, ``liner_blue``, ``sf_blue``, ``sfagn_blue``.

    .. [Lam04] 2004MNRAS.350..396L
    .. [Lam10] 2010A&A...509A..53L
    """

    # Mask for zero fluxes
    zero_flux_blue = ((input_table['HBETA_FLUX'] == 0) |
                      (input_table['OIII_5007_FLUX'] == 0) |
                      (input_table['OII_3726_FLUX'] == 0))
    if mask is not None:
        # Mask for flux availability - included as fastspecfit columns are MaskedColumn data
        mask = mask
        zero_flux_blue = ((input_table['HBETA_FLUX'] == 0) |
                          (input_table['OIII_5007_FLUX'] == 0) |
                          (input_table['OII_3726_FLUX'] == 0) | mask)

    # TODO: BenFloyd - Following up on this.
    ## SJ: DO WE NEED THIS?? We only take sqrt(IVAR) so a zero is fine (it gives SNR=0)
    # If ivar=0 set it to NaN to avoid infinities when computing the error:
    #    input_table['HBETA_FLUX_IVAR']=np.where(input_table['HBETA_FLUX_IVAR']==0,np.nan,input_table['HBETA_FLUX_IVAR'])
    #    input_table['OIII_5007_FLUX_IVAR']=np.where(input_table['OIII_5007_FLUX_IVAR']==0,np.nan,input_table['OIII_5007_FLUX_IVAR'])

    # Mask for SNR. Default is BLUE is available if Hb, OIII SNR >= 3 and OII SNR >= 1.
    snr_hb = input_table['HBETA_FLUX'] * np.sqrt(input_table['HBETA_FLUX_IVAR'])
    snr_oiii = input_table['OIII_5007_FLUX'] * np.sqrt(input_table['OIII_5007_FLUX_IVAR'])
    snr_hb_ew = input_table['HBETA_EW'] * np.sqrt(input_table['HBETA_EW_IVAR'])

    # [OII]3727 is the sum of the doublet [OII]3726,3729
    oii_ew = input_table['OII_3726_EW'] + input_table['OII_3729_EW']
    oii_ew_ivar = 1. / (1. / input_table['OII_3726_EW_IVAR'] + 1. / input_table['OII_3729_EW_IVAR'])
    snr_oii_ew = oii_ew * np.sqrt(oii_ew_ivar)

    # Parameters for horizontal and vertical axes
    log_ewoii_ewhb = np.log10(oii_ew / input_table['HBETA_EW'])
    log_oiii_hb = np.log10(input_table['OIII_5007_FLUX'] / input_table['HBETA_FLUX'])

    # Define regions
    main_blue = 0.11 / (log_ewoii_ewhb - 0.92) + 0.85

    # Mixed region
    eq3_blue1 = -(log_ewoii_ewhb - 1.0) ** 2 - 0.1 * log_ewoii_ewhb + 0.25
    eq3_blue2 = (log_ewoii_ewhb - 0.2) ** 2 - 0.6

    # Seyfert/LINER
    eq4_blue = 0.95 * log_ewoii_ewhb - 0.4

    ## BLUE is available (SNR for the 3 lines other than OII >= 3)
    blue_avail = (snr_hb >= snr) & (snr_oiii >= snr) & (snr_hb_ew >= snr) & (snr_oii_ew >= snr_oii) & (~zero_flux_blue)

    ## BLUE-AGN, SF/LINER/Composite, LINER, SF, SF/AGN
    # Region that overlaps with other classes (set an extra bit for info)
    sflin_blue = blue_avail & ((log_oiii_hb <= eq3_blue1) & (log_oiii_hb >= eq3_blue2))

    # AGN will be subdivided between Seyfert2 & LINER
    agnlin_blue = blue_avail & ((log_oiii_hb >= main_blue) | (log_ewoii_ewhb >= 0.92))
    agn_blue = agnlin_blue & (log_oiii_hb >= eq4_blue)
    liner_blue = agnlin_blue & (log_oiii_hb < eq4_blue)

    # SF 
    sf_blue = blue_avail & (~agnlin_blue) & (log_oiii_hb < 0.3)
    sfagn_blue = blue_avail & (~agnlin_blue) & (log_oiii_hb >= 0.3)

    return blue_avail, agn_blue, sflin_blue, liner_blue, sf_blue, sfagn_blue


def mex(input_table: Table, snr: int | float = 3, mask: MaskedColumn = None) -> (
        tuple[NDArray[bool], NDArray[bool], NDArray[bool], NDArray[bool]]):
    r"""MEx diagnostic originally by [Jun14]_

    MeX diagram regions defined as:
        Top division between SF/AGN (Eq. 1 of [Jun14]_):
            ``mex_upper``:
            :math:` = y
            \begin{cases}
            0.375 / (x - 10.5) + 1.14 & x \leq 10 \\
            a_0 + a_1 x + a_2 x^2 + a_3 x^3 & \text{otherwise}
            \end{cases}`
            with :math:`(a_0, a_1, a_2, a_3) = (410.24, -109.333, 9.71731, -0.288244)`.

        Lower division between SF and "intermediate" (Eq. 2 of [Jun14]_):
            ``mex_lower``:
            :math:`y =
            \begin{cases}
            0.375 / (x - 10.5) + 1.14 & x \leq 9.6 \\
            a_0 + a_1 x + a_2 x^2 + a_3 x^3 & \text{otherwise}
            \end{cases}`
            with :math:`(a_0, a_1, a_2, a_3) = (352.066, -93.8249, 8.32651, -0.246416)`

        Where in both divisions, :math:`y = \log_10(flux_{[OIII]_\lambda5006} / flux_{H\beta})`
        and :math:`x = \log_10(M_*)`

    Notes:
        If using these diagnostic functions please ref the appropriate references given below.

        If using DESI please reference Summary_ref_2023 and the appropriate emission line catalog
        (e.g. FastSpecFit ref FastSpecFit_ref)

    Args:
        input_table: Table including [OIII] and Hβ fluxes, their associated inverse variances and stellar mass
            (derived using either a Chabrier or Kroupa IMF)
        snr: SNR cut applied to the Hβ and [O III] fluxes. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Tuple of arrays of same dimension as rows in ``input_table`` which include flags for
        ``mex_avail``, ``mex_agn``, ``mex_sf``, ``mex_interm``.

    .. [Jun14] 2014ApJ...788...88J
    """

    # Mask for zero fluxes
    zero_flux_mex = (input_table['HBETA_FLUX'] == 0) | (input_table['OIII_5007_FLUX'] == 0)
    if mask is not None:
        # Mask for flux availability - included as fastspecfit columns are MaskedColumn data
        zero_flux_mex = (input_table['HBETA_FLUX'] == 0) | (input_table['OIII_5007_FLUX'] == 0) | mask

    # If ivar=0 set it to NaN to avoid infinites when computing the error:
    input_table['HBETA_FLUX_IVAR'] = np.where(input_table['HBETA_FLUX_IVAR'] == 0,
                                              np.nan, input_table['HBETA_FLUX_IVAR'])
    input_table['OIII_5007_FLUX_IVAR'] = np.where(input_table['OIII_5007_FLUX_IVAR'] == 0,
                                                  np.nan, input_table['OIII_5007_FLUX_IVAR'])

    # Mask for SNR. Default is MEx is available if all SNR >= 3
    snr_hb = input_table['HBETA_FLUX'] * np.sqrt(input_table['HBETA_FLUX_IVAR'])
    snr_oiii = input_table['OIII_5007_FLUX'] * np.sqrt(input_table['OIII_5007_FLUX_IVAR'])

    ## MEx is available (line fluxes SNR >= 3 and valid mass)
    mex_avail = (snr_hb >= snr) & (snr_oiii >= snr) & (input_table['LOGMSTAR'] > 4.) & (~zero_flux_mex)

    # Define variables for equations 1 & 2
    x = input_table['LOGMSTAR']
    y = np.log10(input_table['OIII_5007_FLUX'] / input_table['HBETA_FLUX'])

    # upper MEx
    a0, a1, a2, a3 = 410.24, -109.333, 9.71731, -0.288244
    mex_agn = (((y > 0.375 / (x - 10.5) + 1.14) & (x <= 10)) |
               ((y > a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3) & (x > 10)))

    # lower MEx
    a0, a1, a2, a3 = 352.066, -93.8249, 8.32651, -0.246416
    mex_sf = (((y < 0.375 / (x - 10.5) + 1.14) & (x <= 9.6)) |
              ((y < a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3) & (x > 9.6)))

    # MEX intermediate
    mex_interm = (x > 9.6) & (y >= a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3) & (~mex_agn)

    # Define final output flags by combining the divisions above with the available flag.
    mex_agn &= mex_avail
    mex_sf &= mex_avail
    mex_interm &= mex_avail

    # Return whether it's available and then the 3 classes when also available
    return mex_avail, mex_agn, mex_sf, mex_interm


def kex(input_table: Table, snr: int | float = 3, mask: MaskedColumn = None) -> (
        tuple[NDArray[bool], NDArray[bool], NDArray[bool], NDArray[bool]]):
    r"""KEx diagnostic originally by [Zha18]_.

    KEx diagnostic regions defined as:

        Main division between SF/AGN (Eq. 1 of [Zha18]_):
            ``kex_agn``:
            :math:`log_10(flux_{[OIII]_\lambda5006}/flux_{H\beta}) = -2*\sigma_{[OIII]} + 4.2`

        Division between SF and "intermediate" (Eq. 2 of [Zha18]_):
            ``kex_interm``:
            :math:`log_10(flux_{[OIII]_\lambda5006}/flux_{H\beta}) = 0.3`

    Notes:
        If using these diagnostic functions please ref the appropriate references given below.

        If using DESI please reference Summary_ref_2023 and the appropriate emission line catalog
        (e.g. FastSpecFit ref FastSpecFit_ref)

    Args:
        input_table: Table including [OIII] and Hβ fluxes, their associated inverse variances, and the [OIII]
            equivalent width.
        snr: SNR cut applied to Hβ and [OIII]. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Tuple of arrays of same dimension as rows in ``input_table`` which include flags for
        ``kex_avail``, ``kex_agn``, ``kex_sf``, ``kex_interm``.

    .. [Zha18] 2018ApJ...856..171Z
    """

    # Masks for zero fluxes
    zero_flux_kex = (input_table['HBETA_FLUX'] <= 0.) | (input_table['OIII_5007_FLUX'] <= 0.)
    if mask is not None:
        zero_flux_kex |= mask

    # If ivar=0 set it to NaN to avoid infinites when computing the error:
    input_table['HBETA_FLUX_IVAR'] = np.where(input_table['HBETA_FLUX_IVAR'] == 0,
                                              np.nan, input_table['HBETA_FLUX_IVAR'])
    input_table['OIII_5007_FLUX_IVAR'] = np.where(input_table['OIII_5007_FLUX_IVAR'] == 0,
                                                  np.nan, input_table['OIII_5007_FLUX_IVAR'])

    # Mask for SNR. Default is KEx is available if all SNR >= 3
    snr_hb = input_table['HBETA_FLUX'] * np.sqrt(input_table['HBETA_FLUX_IVAR'])
    snr_oiii = input_table['OIII_5007_FLUX'] * np.sqrt(input_table['OIII_5007_FLUX_IVAR'])

    ## KEx is available (line fluxes SNR >= 3 and valid OIII width)
    kex_avail = (snr_hb >= snr) & (snr_oiii >= snr) & (input_table['OIII_5007_SIGMA'] > 0) & (~zero_flux_kex)

    # Upper KEX
    kex_agn = ((np.log10(input_table['OIII_5007_FLUX'] / input_table['HBETA_FLUX']) >=
                -2. * np.log10(input_table['OIII_5007_SIGMA']) + 4.2)
               & (np.log10(input_table['OIII_5007_FLUX'] / input_table['HBETA_FLUX']) >= 0.3))

    # Lower KEX
    kex_sf = (np.log10(input_table['OIII_5007_FLUX'] / input_table['HBETA_FLUX']) <
              -2. * np.log10(input_table['OIII_5007_SIGMA']) + 4.2)

    # KEX intermediate
    kex_interm = ((np.log10(input_table['OIII_5007_FLUX'] / input_table['HBETA_FLUX']) >=
                   -2. * np.log10(input_table['OIII_5007_SIGMA']) + 4.2)
                  & (np.log10(input_table['OIII_5007_FLUX'] / input_table['HBETA_FLUX']) < 0.3)
                  & (~kex_agn))

    # Define final output flags by combining the divisions above with the available flag.
    kex_agn &= kex_avail
    kex_sf &= kex_avail
    kex_interm &= kex_avail

    # Return whether it's available and then the 3 classes when also available
    return kex_avail, kex_agn, kex_sf, kex_interm


def heii_bpt(input_table: Table, snr: int | float = 3, mask: MaskedColumn = None) -> (
        tuple[NDArray[bool], NDArray[bool], NDArray[bool]]):
    r"""He II BPT diagnostic originally by [Shi12]_

    He II diagnostic regions defined by:
        AGN/SF division:
            :math:`log_10(flux_{HeII_\lambda4685}/flux_{H\beta}) = -1.22 +
            \frac{1}{8.92\log_10(flux_{[NII]_\lambda6583}/flux_{H\alpha}) + 1.32}`

    Notes:
        If using these diagnostic functions please ref Mar_&_Steph_2025 and the appropriate references given below.

        If using DESI please reference Summary_ref_2025 and the appropriate emission line catalog
        (e.g. FastSpecFit ref FastSpecFit_ref)

    Args:
        input_table: Table including H⍺, Hβ, HeII, [NII] fluxes and inverse variances.
        snr: SNR cut applied to all axes. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Tuple of arrays of same dimension as rows in ``input_table`` which include flags for
        ``heii_bpt_avail``, ``agn_heii``, ``sf_heii``

    .. [Shi12] 2012MNRAS.421.1043S
    """

    # Mask for zero fluxes
    zero_flux_heii = ((input_table['HALPHA_FLUX'] == 0)
                      | (input_table['HBETA_FLUX'] == 0)
                      | (input_table['HEII_4686_FLUX'] == 0)
                      | (input_table['NII_6584_FLUX'] == 0))
    if mask is not None:
        # Mask for flux availability - included as fastspecfit columns are MaskedColumn data
        zero_flux_heii |= mask

    # If ivar=0 set it to NaN to avoid infinities when computing the error:
    input_table['HALPHA_FLUX_IVAR'] = np.where(input_table['HALPHA_FLUX_IVAR'] == 0,
                                               np.nan, input_table['HALPHA_FLUX_IVAR'])
    input_table['HBETA_FLUX_IVAR'] = np.where(input_table['HBETA_FLUX_IVAR'] == 0,
                                              np.nan, input_table['HBETA_FLUX_IVAR'])
    input_table['HEII_4686_FLUX_IVAR'] = np.where(input_table['HEII_4686_FLUX_IVAR'] == 0,
                                                  np.nan, input_table['HEII_4686_FLUX_IVAR'])
    input_table['NII_6584_FLUX_IVAR'] = np.where(input_table['NII_6584_FLUX_IVAR'] == 0,
                                                 np.nan, input_table['NII_6584_FLUX_IVAR'])

    # Mask for SNR. Default is HeII-BPT is available if Ha, Hb, NII, HeII SNR >= 3
    snr_ha = input_table['HALPHA_FLUX'] * np.sqrt(input_table['HALPHA_FLUX_IVAR'])
    snr_hb = input_table['HBETA_FLUX'] * np.sqrt(input_table['HBETA_FLUX_IVAR'])
    snr_heii = input_table['HEII_4686_FLUX'] * np.sqrt(input_table['HEII_4686_FLUX_IVAR'])
    snr_nii = input_table['NII_6584_FLUX'] * np.sqrt(input_table['NII_6584_FLUX_IVAR'])

    # Define regions
    log_nii_ha = np.log10(input_table['NII_6584_FLUX'] / input_table['HALPHA_FLUX'])
    log_heii_hb = np.log10(input_table['HEII_4686_FLUX'] / input_table['HBETA_FLUX'])
    shir12 = -1.22 + 1 / (8.92 * log_nii_ha + 1.32)

    # Value where denominator goes to zero (non-finite)
    log_nii_ha_0 = -1.32 / 8.92

    ## HeII-BPT is available (All lines SNR >= 3)
    heii_bpt_avail = (snr_ha >= snr) & (snr_hb >= snr) & (snr_heii >= snr) & (snr_nii >= snr) & (~zero_flux_heii)

    ## HeII-AGN, SF
    agn_heii = heii_bpt_avail & ((log_heii_hb >= shir12) | (log_nii_ha >= log_nii_ha_0))
    sf_heii = heii_bpt_avail & (~agn_heii)

    return heii_bpt_avail, agn_heii, sf_heii


def nev(input_table: Table, snr: int | float = 2.5, mask: MaskedColumn = None) -> (
        tuple[NDArray[bool], NDArray[bool], NDArray[bool]]):
    r"""[NeV] diagnostic originally by [ref]_

     If using these diagnostic functions please ref Mar_&_Steph_2025 and the appropriate references given below.

    If using DESI please reference Summary_ref_2025 and the appropriate emission line catalog
    (e.g. FastSpecFit ref FastSpecFit_ref)

    Notes:
        [Ne V] diagnostic is defined by the detection of the [NeV] :math:`\lambda3426` emission line which implies
        hard radiation from photon energies :math:`kT > 96.6` eV, indicating AGN activity.

    Args:
        input_table: Table including [NeV] flux and inverse variance.
        snr: SNR cut applied to [NeV]. Default is ``2.5``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Tuple of arrays of same dimension as rows in ``input_table`` which include flags for
        ``nev_avail``, ``agn_nev``, ``sf_nev``.

    .. [ref] TODO Need actual reference. Cleri+23? Berg+21? Schmidt+98?
    """

    # Mask for zero fluxes
    zero_flux_nev: NDArray[bool] = input_table['NEV_3426_FLUX'] == 0
    if mask is not None:
        # Mask for flux availability - included as fastspecfit columns are MaskedColumn data
        zero_flux_nev |= mask

    # If ivar=0 set it to NaN to avoid infinities when computing the error:
    input_table['NEV_3426_FLUX_IVAR'] = np.where(input_table['NEV_3426_FLUX_IVAR'] == 0,
                                                 np.nan, input_table['NEV_3426_FLUX_IVAR'])

    # Mask for SNR.
    snr_nev = input_table['NEV_3426_FLUX'] * np.sqrt(input_table['NEV_3426_FLUX_IVAR'])

    ## NeV diagnostic is available if flux is not zero
    nev_avail = ~zero_flux_nev

    ## NeV-AGN, SF
    agn_nev = nev_avail & (snr_nev >= snr)
    sf_nev = ~agn_nev

    # TODO: BenFloyd Figure out why inspection thinks that the types should be tuple[int, int|Any, int].
    return nev_avail, agn_nev, sf_nev


##########################################################################################################
##########################################################################################################

# TODO: BenFloyd - This function needs to be split up like the optical diagnostics are.
def WISE_colors(input_table: Table, snr: float | int = 3, mask: MaskedColumn = None, diag: str = 'All',
                weak_agn: bool = False) -> (tuple[NDArray[bool], NDArray[bool], NDArray[bool]] |
                                            tuple[NDArray[bool], NDArray[bool], NDArray[bool], NDArray[bool]]):
    """WISE Color diagnostics

    Regions defined as:
    Region defined in WISE infrared color space, indicating AGN. Note of caution: The points outside the AGN region may
    still include a significant fraction of AGN and are best considered as "uncertain" rather than "star-forming" or
    "non-AGN"

    Notes:
        If using these diagnostic functions please ref Mar_&_Steph_2025 and the appropriate references given below.

        If using DESI please reference Summary_ref_2025 and the appropriate photometry catalog (e.g., Tractor or
        Photometry VAC)

    Args:
        input_table: Table including WISE fluxes and inverse variances for W1, W2, and W3 bands.
        snr: SNR cut applied to WISE fluxes. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.
        diag: String indicating which WISE AGN selection to use. Must be one of ``'Stern12'``, ``'Assef18'``,
            ``'Jarrett11'``, ``Mateos12``, ``'Yao20'``, ``'Hviding22'``, or ``'All'``. Defaults to ``'All'``.
            Note: Selecting ``'All'`` produces an AGN selection that is a union combination of all the individual
            selections. If ``'Yao20'`` is used, the function can output an additional selection for `week_agn`.
        weak_agn: Boolean flag to optionally output ``weak_agn`` as a selection. Only valid for ``'Yao20'`` selection.

    Returns:
        Tuple of arrays of same dimension as rows in ``input_table`` which include flags for
        ``avail_ir``, ``agn_ir``, ``sf_ir``, and optionally ``weak_agn`` if ``'Yao20'`` selection used.
    """

    # Mask for zero fluxes
    zero_flux_wise = (input_table['FLUX_W1'] == 0) | (input_table['FLUX_W2'] == 0)
    zero_flux_w3 = (input_table['FLUX_W3'] == 0)
    if mask != None:
        # Mask for flux availability - included if input_table photometry is missing/masked
        mask = mask  # TODO: BenFloyd - Redundant statement?
        zero_flux_wise = (input_table['FLUX_W1'] == 0) | (input_table['FLUX_W2'] == 0) | mask
        zero_flux_w3 = (input_table['FLUX_W3'] == 0) | mask

    # If ivar=0 set it to NaN to avoid infinites when computing the error:
    input_table['FLUX_IVAR_W1'] = np.where(input_table['FLUX_IVAR_W1'] == 0, np.nan, input_table['FLUX_IVAR_W1'])
    input_table['FLUX_IVAR_W2'] = np.where(input_table['FLUX_IVAR_W2'] == 0, np.nan, input_table['FLUX_IVAR_W2'])
    input_table['FLUX_IVAR_W3'] = np.where(input_table['FLUX_IVAR_W3'] == 0, np.nan, input_table['FLUX_IVAR_W3'])

    # Mask for SNR.
    snr = snr  # TODO: BenFloyd - Redundant statement?
    SNR_W1 = input_table['FLUX_W1'] * np.sqrt(input_table['FLUX_IVAR_W1'])
    SNR_W2 = input_table['FLUX_W2'] * np.sqrt(input_table['FLUX_IVAR_W2'])
    SNR_W3 = input_table['FLUX_W3'] * np.sqrt(input_table['FLUX_IVAR_W3'])

    ## IR diagnostic based on W1W2 is available if flux is not zero
    ## Note: Yao+2020 used S/N>5 for W1, W2 and S/N>2 for W3 due to being much less sensitive
    W1W2_avail = (~zero_flux_wise) & (SNR_W1 > snr) & (SNR_W2 > snr)
    W2W3_avail = (~zero_flux_wise) & (~zero_flux_w3) & (SNR_W2 > snr) & (SNR_W3 > snr)

    # Convert to AB magnitudes (most diagnostics use Vega mags so need to apply offsets)
    W1 = 22.5 - 2.5 * np.log10(input_table['FLUX_W1'])
    W2 = 22.5 - 2.5 * np.log10(input_table['FLUX_W2'])
    W3 = 22.5 - 2.5 * np.log10(input_table['FLUX_W3'])
    W1W2 = W1 - W2
    W2W3 = W2 - W3

    # Offsets from Vega to AB magnitudes (Jarrett+2011) 
    W1_vega2ab = 2.699
    W2_vega2ab = 3.339
    W3_vega2ab = 5.174

    # Offsets from Vega to AB WISE colors
    W1W2_vega2ab = W1_vega2ab - W2_vega2ab
    W2W3_vega2ab = W2_vega2ab - W3_vega2ab

    # Subtract offsets to go from AB to Vega (add to go from Vega to AB)
    W1_Vega = W1 - W1_vega2ab
    W2_Vega = W2 - W2_vega2ab
    W3_Vega = W3 - W3_vega2ab
    W1W2_Vega = W1_Vega - W2_Vega  # W1W2 - W1W2_vega2ab
    W2W3_Vega = W2_Vega - W3_Vega  # W2W3 - W2W3_vega2ab

    ## Jarrett et al. 2011 box in W1-W2 vs. W2-W3 space in Vega mags
    y_top = 1.7
    y_bot = 0.1 * W2W3_Vega + 0.38
    x_left = 2.2
    x_right = 4.2

    agn_jarrett11 = W1W2_avail & W2W3_avail & (W2W3_Vega > x_left) & (W2W3_Vega < x_right) & (W1W2_Vega > y_bot) & (
            W1W2_Vega < y_top)
    # TODO: BenFloyd - The following selections are not used
    sf_jarrett11 = W1W2_avail & W2W3_avail & (~agn_jarrett11)
    unavail_jarrett11 = (~W1W2_avail) | (~W2W3_avail)  # unavailable

    ## Stern et al. 2012 cut along just W1-W2 color
    agn_stern12 = W1W2_avail & (W1W2_Vega > 0.8)
    # TODO: BenFloyd - The following selections are not used
    sf_stern12 = W1W2_avail & (~agn_stern12)
    unavail_stern12 = ~W1W2_avail  # unavailable

    ## Mateos et al. 2012 box in W1-W2 vs. W2-W3 space
    x_M12 = W2W3 / (2.5)  # from eqn 1 using AB mags
    y_M12 = W1W2 / (2.5)

    # top/bottom around the power-law
    y_top = 0.315 * x_M12 + 0.297  # eqn 1 + offset
    y_bot = 0.315 * x_M12 - 0.110  # eqn 1 - offset
    y_pl = -3.172 * x_M12 + 0.436  # eqn 2 for the power-law

    agn_mateos12 = W1W2_avail & W2W3_avail & (y_M12 > y_bot) & (y_M12 > y_pl) & (y_M12 < y_top)
    # TODO: BenFloyd - The following selections are not used
    sf_mateos12 = W1W2_avail & W2W3_avail & (~agn_mateos12)
    unavail_mateos12 = (~W1W2_avail) | (~W2W3_avail)  # unavailable

    ## Assef et al. 2018: https://ui.adsabs.harvard.edu/abs/2018ApJS..234...23A/abstract
    # equation 2 (simplistic from Stern+12): (W1W2_Vega >= 0.8)&((W2 - W2_vega2ab)<15.05)
    # equation 3: W1W2_Vega > alpha* exp(beta*(W2_Vega-gamma)**2)

    ## 90% reliability
    alpha_90 = 0.65
    beta_90 = 0.153
    gamma_90 = 13.86

    # TODO: BenFloyd - The following parameterization is not implemented.
    ## 75% reliability
    alpha_75 = 0.486
    beta_75 = 0.092
    gamma_75 = 13.07

    ## Choose here:
    alpha = alpha_90
    beta = beta_90
    gamma = gamma_90

    bright_a18 = W2_Vega <= gamma

    agn_assef18 = W1W2_avail & ((W1W2_Vega > alpha * np.exp(beta * (W2_Vega - gamma) ** 2)) |
                                ((W1W2_Vega > alpha) & bright_a18))
    sf_assef18 = W1W2_avail & (~agn_assef18)
    unavail_assef18 = ~W1W2_avail  # unavailable

    ## Yao et al. 2020 cuts
    # Vega mags: w1w2 = (0.015 * exp(w2w3/1.38)) - 0.08 + offset
    # where offset of 0.3 is reported in paper as the 2*sigma cut to create a demarcation line
    line_yao20 = (0.015 * np.exp(W2W3_Vega / 1.38)) - 0.08 + 0.3
    strong_agn_yao20 = W1W2_avail & W2W3_avail & (agn_jarrett11 | agn_stern12)
    # Line for low-power AGN
    weak_agn_yao20 = W1W2_avail & W2W3_avail & (W1W2_Vega > line_yao20) & ~strong_agn_yao20
    unavail_yao20 = (~W1W2_avail) | (~W2W3_avail)  # unavailable

    ## Hviding et al. 2022 cuts in (y=)W1-W2 vs. (x=)W2-W3 space in Vega mags (eq. 3)
    x_left = 1.734
    x_right = 3.916
    y_bot1 = 0.0771 * W2W3_Vega + 0.319
    y_bot2 = 0.261 * W2W3_Vega - 0.260
    agn_hviding22 = W1W2_avail & W2W3_avail & (W2W3_Vega > x_left) & (W2W3_Vega < x_right) & (W1W2_Vega > y_bot1) & (
            W1W2_Vega > y_bot2)
    # TODO: BenFloyd - The following selection is not used
    unavail_hviding22 = (~W1W2_avail) | (~W2W3_avail)  # unavailable

    ## Set the choice here for individual diagnostics or our default combination # agn_hviding22 not yet implemented
    if diag == 'Stern12':
        agn_ir = agn_stern12
        avail_ir = W1W2_avail
    if diag == 'Assef18':
        agn_ir = agn_assef18
        avail_ir = W1W2_avail
    if diag == 'Jarrett11':
        agn_ir = agn_jarrett11
        avail_ir = W1W2_avail & W2W3_avail
    if diag == 'Mateos12':
        agn_ir = agn_mateos12
        avail_ir = W1W2_avail & W2W3_avail
    if diag == 'Yao20':
        agn_ir = strong_agn_yao20
        wagn_ir = weak_agn_yao20  # not used for now (would need code changes)
        avail_ir = W1W2_avail & W2W3_avail
    if diag == 'Hviding22':
        agn_ir = agn_hviding22
        avail_ir = W1W2_avail & W2W3_avail
    ## By default, combine the diagnostics based on W1W2W3 when all 3 bands available;
    #  otherwise use the Stern cut on W1-W2 only
    if diag == 'All':
        agn_ir = agn_mateos12 | agn_jarrett11 | (agn_stern12 & ~W2W3_avail) | agn_assef18 | agn_hviding22
        avail_ir = W1W2_avail
        # By default, not considering weak (low-power) AGN; only return if specified
        wagn_ir = weak_agn_yao20 & ~agn_ir

    # SF defined based on the above
    sf_ir = avail_ir & (~agn_ir)

    # By default, not considering weak (low-power) AGN from Yao+20; only return if specified
    if not weak_agn:
        return avail_ir, agn_ir, sf_ir
    else:
        return avail_ir, agn_ir, sf_ir & ~wagn_ir, wagn_ir


##########################################################################################################
##########################################################################################################

# TODO: - BenFloyd - Not updating this function as it is as yet not fully implemented
def Xray(input, H0=67.4, Om0=0.315, snr=3):
    ## X-ray diagnostic ##
    # 2-10 keV X-ray luminosity equal or above 1e42 erg/s indicates AGN
    thres = 1e42

    # Fiducial Cosmology used in DESI from Planck 2018 results: https://ui.adsabs.harvard.edu/abs/2020A%26A...641A...6P/abstract
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    DL = cosmo.luminosity_distance(input['z'].values)  # in Mpc
    DL_cm = 3.08567758e24 * DL.value

    # Convert the CSC 2-7 keV flux to 2-10 keV and compute the LX2-10 keV
    # Conversion factor =  1.334E-15 erg cm^-2 s^-1 for an unabsorbed flux of 1E-15 erg cm^-2 s^-1 using PIMMS (https://cxc.harvard.edu/toolkit/pimms.jsp) and assuming gamma=1.8
    factor = 1.334E-15  # for gamma=1.8
    flux_2_10 = (input['FLUX_2_7'] / 1E-15) * factor
    LX210 = 4 * np.pi * DL_cm ** 2 * flux_2_10  # in erg/s

    # Apply K-correction:
    gamma = 1.8
    k = (1 + input['z']) ** (gamma - 2)
    LX210_Kcorr = k * LX210

    # Mask for zero flux
    zero_flux_xray = input['FLUX_2_7'] == 0

    # Mask for SNR
    snr = snr
    SNR_Xray = input['FLUX_2_7'] / input['FLUX_2_7_err']

    ## Xray diagnostic is available if flux is not zero and SNR_Xray >= 3
    xray = (SNR_Xray >= snr) & (~zero_flux_xray)

    ## Xray-AGN, SF, footprint
    agn_xray = (xray) & (LX210_Kcorr >= thres)
    sf_xray = (xray) & ~agn_xray
    fp_xray = (zero_flux_xray) & (input['FLUX_2_7_err'] > 0)

    return (agn_xray, sf_xray, fp_xray)
