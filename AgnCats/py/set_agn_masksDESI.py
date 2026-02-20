"""
set_agn_masksDESI.py

Applies the BitMask values for variety of AGN/Galaxy classifications according to the diagnostics defined in
``uv_opt_agn_diagnostics.py`` and ``ir_agn_diagnostics.py``.

Original Author:
Raga Pucha, 2021

Modified by:
Becky Canning (University of Portsmouth), 2023
Stephanie Juneau (NOIRlab), Nov 2024, Feb 2025

Revised by:
Benjamin Floyd (University of Portsmouth)
"""
from pathlib import Path
from typing import Literal

import yaml
from astropy.table import Table, MaskedColumn
from desiutil.bitmask import BitMask

import uv_opt_agn_diagnostics as uv_opt_agn
import ir_agn_diagnostics as ir_agn


## Original notes:
## Raga Pucha: First draft of the function (2021)
## Need to add detailed comments to the script
## Returns a lot of warnings because of division by zero - have to use filterwarnings after checking everything
## Version: 2025 February 25
## Edited: B. Canning 2023; S. Juneau November 2024, Feb 2025

###
# Retrieve the bitmasks definitions from the yaml file
# Note: QSO_MASKBITS are applied to the first 9 bits of agn_maskbits
#       OPT_UV_TYPE and IR_TYPE include definitions for detailed classification

def get_agn_maskbits(file: Path | str) -> tuple[BitMask, BitMask, BitMask]:
    """Parses the AGN bitmask definition YAML file into DESI BitMask objects.

    Args:
        file: AGN bitmask definition YAML file.

    Returns:
        Bitmasks for ``agn_maskbits``, ``opt_uv_type``, and ``ir_type``.
    """
    with open(file, 'r') as file_yaml:
        yaml_defs = yaml.safe_load(file_yaml)

    agn_maskbits = BitMask('AGN_MASKBITS', yaml_defs)
    opt_uv_type = BitMask('OPT_UV_TYPE', yaml_defs)
    ir_type = BitMask('IR_TYPE', yaml_defs)

    return agn_maskbits, opt_uv_type, ir_type


def update_agn_maskbits(input_table: Table, agn_maskbits: BitMask, snr: int | float = 3, snr_oi: int | float = 1,
                        snr_oii: int | float = 1, snr_wise: int | float = 3, kewley01: bool = False,
                        mask: MaskedColumn = None) -> Table:
    """Sets the ``AGN_MASKBITS`` values in the input catalog.

    ``AGN_MASKBITS`` are initialized from the ``QSO_MASKBITS`` column from QSO MAKER. They are then further modified by
    applying the various UV/Optical diagnostics (e.g., BPT, WHAN, Blue, MEx, KEx) and by the WISE IR selections to
    provide additional bitmasks to indicate if a galaxy is classified as an AGN by any of these diagnostics.

    Args:
        input_table: Table consisting of columns joined from the QSO Maker, Redshift Summary (zcat) VAC, and FastSpecFit
            catalogs.
        agn_maskbits: DESI BitMask object containing the definitions of the ``AGN_MASKBITS`` values.
        snr: Signal-to-noise cut applied to all axes passed to diagnostics. Default is ``3``.
        snr_oi: Signal-to-noise cut applied to the [OI]λ6300 emission line. Used for the [OI] BPT diagnostic.
            Default is ``1``.
        snr_wise: Signal-to-noise cut applied to the WISE fluxes. Default is ``3``.
        snr_oii: Signal-to-noise cut applied to the [OII]λ3727 flux. Used for the Blue diagnostic. Default is ``3``.
        kewley01: Optional flag to use Kewley+01 lines for SF/AGN classification instead of Law+21 lines. Used for the
            [SII] and [OI] BPT diagnostics. Default is ``False``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Input table with new or updated column with ``AGN_MASKBITS`` information for all rows.

    """

    # Create masks for objects in the input catalog with QSO_MASKBITS matching our AGN_MASKBITS
    qsom_rr = input_table['QSO_MASKBITS'] & agn_maskbits.RR == agn_maskbits.RR
    qsom_mgii = input_table['QSO_MASKBITS'] & agn_maskbits.MGII == agn_maskbits.MGII
    qsom_qn = input_table['QSO_MASKBITS'] & agn_maskbits.QN == agn_maskbits.QN
    qsom_qn_rr = input_table['QSO_MASKBITS'] & agn_maskbits.QN_NEW_RR == agn_maskbits.QN_NEW_RR
    qsom_qn_bgs = input_table['QSO_MASKBITS'] & agn_maskbits.QN_BGS == agn_maskbits.QN_BGS
    qsom_qn_elg = input_table['QSO_MASKBITS'] & agn_maskbits.QN_ELG == agn_maskbits.QN_ELG
    qsom_qn_var_wise = input_table['QSO_MASKBITS'] & agn_maskbits.QN_VAR_WISE == agn_maskbits.QN_VAR_WISE

    # Initialize the AGN_MASKBITS with the existing QSO_MASKBITS
    agn_bits = qsom_rr * agn_maskbits.RR
    agn_bits |= qsom_mgii * agn_maskbits.MGII
    agn_bits |= qsom_qn * agn_maskbits.QN
    agn_bits |= qsom_qn_rr * agn_maskbits.QN_NEW_RR
    agn_bits |= qsom_qn_bgs * agn_maskbits.QN_BGS
    agn_bits |= qsom_qn_elg * agn_maskbits.QN_ELG
    agn_bits |= qsom_qn_var_wise * agn_maskbits.QN_VAR_WISE

    # BPT classifications from individual diagnostics
    *_, agn_nii, liner_nii, composite_nii = uv_opt_agn.nii_bpt(input_table, snr=snr, mask=mask)
    *_, agn_sii, liner_sii = uv_opt_agn.sii_bpt(input_table, snr=snr, kewley01=kewley01, mask=mask)
    *_, agn_oi, liner_oi = uv_opt_agn.oi_bpt(input_table, snr=snr, snr_oi=snr_oi, kewley01=kewley01, mask=mask)

    # Combined BPT classification
    bpt_any_sy = agn_nii | agn_sii | agn_oi
    bpt_any_agn = agn_nii | agn_sii | agn_oi | liner_nii | composite_nii | liner_sii | liner_oi
    agn_bits |= bpt_any_sy * agn_maskbits.BPT_ANY_SY
    agn_bits |= bpt_any_agn * agn_maskbits.BPT_ANY_AGN

    # Whether there is a broad line (FWHM>= 1200 km/s)
    bl = uv_opt_agn.broad_line(input_table, snr=snr, mask=mask, vel_thresh=1200.)
    agn_bits |= bl * agn_maskbits.BROAD_LINE

    # Other (non-BPT) optical diagnostics: WHAN, MEx, KEx, Blue
    _, _, whan_sagn, *_ = uv_opt_agn.whan(input_table, snr=snr, mask=mask)
    _, mex_agn, *_ = uv_opt_agn.mex(input_table, snr=snr, mask=mask)
    _, agn_blue, *_ = uv_opt_agn.blue(input_table, snr=snr, snr_oii=snr_oii,
                                      mask=mask)
    kex, kex_agn, kex_sf, kex_interm = uv_opt_agn.kex(input_table, snr=snr, mask=mask)

    # Combine them for the OPT_OTHER_AGN (keeping mostly more confident ones and 
    # excluding possible weak AGN / blended classes)
    opt_other_agn = whan_sagn | mex_agn | agn_blue | kex_agn
    agn_bits |= opt_other_agn * agn_maskbits.OPT_OTHER_AGN

    # Overall WISE classification (combining all diagnostics)
    _, jarret11_agn, _ = ir_agn.wise_jarrett11(input_table, snr=snr_wise, mask=mask)
    _, stern12_agn, _ = ir_agn.wise_stern12(input_table, snr=snr_wise, mask=mask)
    _, mateos12_agn, _ = ir_agn.wise_mateos12(input_table, snr=snr_wise, mask=mask)
    _, assef18_r_agn, _ = ir_agn.wise_assef18_r(input_table, snr=snr_wise, reliability=90, mask=mask)
    _, yao20_strong_agn, _ = ir_agn.wise_yao20(input_table, snr=snr_wise, weak_agn=False, mask=mask)
    _, hviding22_agn, _ = ir_agn.wise_hviding22(input_table, snr=snr_wise, mask=mask)
    agn_wise = jarret11_agn | stern12_agn | mateos12_agn | assef18_r_agn | yao20_strong_agn | hviding22_agn
    agn_bits |= agn_wise * agn_maskbits.WISE_ANY_AGN

    # uv, xray, radio =
    #agn_bits |= uv * agn_mask.UV
    #agn_bits |= xray * agn_mask.XRAY
    #agn_bits |= radio * agn_mask.RADIO

    # Finally, add the AGN_ANY definition based on confident AGNs
    agn_any = (qsom_rr | qsom_mgii | qsom_qn | qsom_qn_bgs | qsom_qn_elg | qsom_qn_var_wise
               | bpt_any_sy | agn_wise | opt_other_agn)
    agn_bits |= agn_any * agn_maskbits.AGN_ANY

    try:
        input_table['AGN_MASKBITS'] |= agn_bits
    except KeyError:
        input_table['AGN_MASKBITS'] = agn_bits

    return input_table


def update_agntype_nii_bpt(input_table: Table, opt_uv_type: BitMask, snr: int | float = 3,
                           mask: MaskedColumn = None) -> Table:
    """Applies the [NII] BPT masks and sets the bitmasks for ``OPT_UV_TYPE``.

    Args:
        input_table: Table joined with FastSpecFit columns.
        opt_uv_type: DESI BitMask object containing the definitions of the ``OPT_UV_TYPE`` values.
        snr: Signal-to-noise cut applied to all flux axes. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Input table with new or updated column with ``OPT_UV_TYPE`` bit masks for [N II] BPT selections for all rows.
    """

    nii_bpt, sf_nii, agn_nii, liner_nii, composite_nii = uv_opt_agn.nii_bpt(input_table, snr=snr, mask=mask)

    # If any of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)
    bpt_mask = nii_bpt * opt_uv_type.NII_BPT  ## All the emission lines have S/N >= 3
    bpt_mask |= sf_nii * opt_uv_type.NII_SF  ## [NII] - Star Forming
    bpt_mask |= agn_nii * opt_uv_type.NII_SY  ## [NII] - Seyfert
    bpt_mask |= liner_nii * opt_uv_type.NII_LINER  ## [NII] - LINER
    bpt_mask |= composite_nii * opt_uv_type.NII_COMP  ## [NII] - Composite

    try:
        input_table['OPT_UV_TYPE'] |= bpt_mask
    except KeyError:
        input_table['OPT_UV_TYPE'] = bpt_mask

    return input_table


def update_agntype_sii_bpt(input_table: Table, opt_uv_type: BitMask, snr: int | float = 3, kewley01: bool = False,
                           mask: MaskedColumn = None) -> Table:
    """Applies the [SII] BPT masks and sets the bitmasks for ``OPT_UV_TYPE``.

    Args:
        input_table: Table joined with FastSpecFit columns.
        opt_uv_type: DESI BitMask object containing the definitions of the ``OPT_UV_TYPE`` values.
        snr: Signal-to-noise cut applied to all flux axes. Default is ``3``.
        kewley01: Optional flag to use Kewley+01 lines for SF/AGN classification instead of Law+21 lines.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Input table with new or updated column with ``OPT_UV_TYPE`` bit masks for [S II] BPT selections for all rows.
    """

    sii_bpt, sf_sii, agn_sii, liner_sii = uv_opt_agn.sii_bpt(input_table, snr=snr, kewley01=kewley01, mask=mask)

    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    bpt_mask = sii_bpt * opt_uv_type.SII_BPT  ## All the emission lines have S/N >= 3
    bpt_mask |= sf_sii * opt_uv_type.SII_SF  ## [SII] - Star Forming
    bpt_mask |= agn_sii * opt_uv_type.SII_SY  ## [SII] - Seyfert
    bpt_mask |= liner_sii * opt_uv_type.SII_LINER  ## [SII] - LINER

    try:
        input_table['OPT_UV_TYPE'] |= bpt_mask
    except KeyError:
        input_table['OPT_UV_TYPE'] = bpt_mask

    return input_table


def update_agntype_oi_bpt(input_table: Table, opt_uv_type: BitMask, snr: int | float = 3, snr_oi: int | float = 1,
                          kewley01: bool = False, mask: MaskedColumn = None) -> Table:
    """Applies the [OI] BPT masks and sets the bitmasks for ``OPT_UV_TYPE``.

    Args:
        input_table: Table joined with FastSpecFit columns.
        opt_uv_type: DESI BitMask object containing the definitions of the ``OPT_UV_TYPE`` values.
        snr: Signal-to-noise cut applied to all flux axes. Default is ``3``.
        snr_oi: Signal-to-noise cut applied to the [OI]λ6300 emission line. Default is ``1``.
        kewley01: Optional flag to use Kewley+01 lines for SF/AGN classification instead of Law+21 lines.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Input table with new or updated column with ``OPT_UV_TYPE`` bit masks for [OI] BPT selections for all rows.
    """

    oi_bpt, sf_oi, agn_oi, liner_oi = uv_opt_agn.oi_bpt(input_table, snr=snr, snr_oi=snr_oi, kewley01=kewley01, mask=mask)

    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    bpt_mask = oi_bpt * opt_uv_type.OI_BPT  ## Except [OI] - other em lines have S/N >= 3
    bpt_mask |= sf_oi * opt_uv_type.OI_SF  ## [OI] - Star Forming
    bpt_mask |= agn_oi * opt_uv_type.OI_SY  ## [OI] - Seyfert
    bpt_mask |= liner_oi * opt_uv_type.OI_LINER  ## [OI] - LINER

    try:
        input_table['OPT_UV_TYPE'] |= bpt_mask
    except KeyError:
        input_table['OPT_UV_TYPE'] = bpt_mask

    return input_table


def update_agntype_whan(input_table: Table, opt_uv_type: BitMask, snr: int | float = 3,
                        mask: MaskedColumn = None) -> Table:
    """Applies the WHAN masks and sets the bitmasks for ``OPT_UV_TYPE``.

    Args:
        input_table: Table joined with FastSpecFit columns.
        opt_uv_type: DESI BitMask object containing the definitions of the ``OPT_UV_TYPE`` values.
        snr: Signal-to-noise cut applied to all flux axes. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
       Input table with new or updated column with ``OPT_UV_TYPE`` bit masks for WHAN selections for all rows.
    """

    whan, whan_sf, whan_sagn, whan_wagn, whan_retired, whan_passive = uv_opt_agn.whan(input_table, snr=snr, mask=mask)

    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    agn_mask = whan * opt_uv_type.WHAN  ## WHAN is available (Halpha and [NII])
    agn_mask |= whan_sf * opt_uv_type.WHAN_SF  ## WHAN Star-forming
    agn_mask |= whan_sagn * opt_uv_type.WHAN_SAGN  ## WHAN Strong AGN
    agn_mask |= whan_wagn * opt_uv_type.WHAN_WAGN  ## WHAN Weak AGN
    agn_mask |= whan_retired * opt_uv_type.WHAN_RET  ## WHAN Retired
    agn_mask |= whan_passive * opt_uv_type.WHAN_PASS  ## WHAN Passive

    try:
        input_table['OPT_UV_TYPE'] |= agn_mask
    except KeyError:
        input_table['OPT_UV_TYPE'] = agn_mask

    return input_table


def update_agntype_blue(input_table: Table, opt_uv_type: BitMask, snr: int | float = 3, snr_oii: int | float = 1,
                        mask: MaskedColumn = None) -> Table:
    """Applies the Blue masks and sets the bitmasks for ``OPT_UV_TYPE``.

    Args:
        input_table: Table joined with FastSpecFit columns.
        opt_uv_type: DESI BitMask object containing the definitions of the ``OPT_UV_TYPE`` values.
        snr: Signal-to-noise cut applied to all flux axes. Default is ``3``.
        snr_oii: Signal-to-noise cut applied to the [OII]λ3727 flux. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
       Input table with new or updated column with ``OPT_UV_TYPE`` bit masks for Blue selections for all rows.
    """

    blue, agn_blue, sflin_blue, liner_blue, sf_blue, sfagn_blue = uv_opt_agn.blue(input_table, snr=snr, snr_oii=snr_oii, mask=mask)

    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    agn_mask = blue * opt_uv_type.BLUE
    agn_mask |= agn_blue * opt_uv_type.BLUE_AGN
    agn_mask |= sflin_blue * opt_uv_type.BLUE_SLC
    agn_mask |= liner_blue * opt_uv_type.BLUE_LINER
    agn_mask |= sf_blue * opt_uv_type.BLUE_SF
    agn_mask |= sfagn_blue * opt_uv_type.BLUE_SFAGN

    try:
        input_table['OPT_UV_TYPE'] |= agn_mask
    except KeyError:
        input_table['OPT_UV_TYPE'] = agn_mask

    return input_table


def update_agntype_mex(input_table: Table, opt_uv_type: BitMask, snr: int | float = 3,
                       mask: MaskedColumn = None) -> Table:
    """Applies the MEx masks and sets the bitmasks for ``OPT_UV_TYPE``.

    Args:
        input_table: Table joined with FastSpecFit columns.
        opt_uv_type: DESI BitMask object containing the definitions of the ``OPT_UV_TYPE`` values.
        snr: Signal-to-noise cut applied to all flux axes. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Input table with new or updated column with ``OPT_UV_TYPE`` bit masks for MEx selections for all rows.
    """

    mex, mex_agn, mex_sf, mex_interm = uv_opt_agn.mex(input_table, snr=snr, mask=mask)

    agn_mask = mex * opt_uv_type.MEX
    agn_mask |= mex_agn * opt_uv_type.MEX_AGN
    agn_mask |= mex_sf * opt_uv_type.MEX_SF
    agn_mask |= mex_interm * opt_uv_type.MEX_INTERM

    try:
        input_table['OPT_UV_TYPE'] |= agn_mask
    except KeyError:
        input_table['OPT_UV_TYPE'] = agn_mask

    return input_table


def update_agntype_kex(input_table: Table, opt_uv_type: BitMask, snr: int | float = 3,
                       mask: MaskedColumn = None) -> Table:
    """Applies the KEx masks and sets the bitmasks for ``OPT_UV_TYPE``.

    Args:
        input_table: Table joined with FastSpecFit columns.
        opt_uv_type: DESI BitMask object containing the definitions of the ``OPT_UV_TYPE`` values.
        snr: Signal-to-noise cut applied to all flux axes. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Input table with new or updated column with ``OPT_UV_TYPE`` bit masks for KEx selections for all rows.
    """

    kex, kex_agn, kex_sf, kex_interm = uv_opt_agn.kex(input_table, snr=snr, mask=mask)

    agn_mask = kex * opt_uv_type.KEX
    agn_mask |= kex_agn * opt_uv_type.KEX_AGN
    agn_mask |= kex_sf * opt_uv_type.KEX_SF
    agn_mask |= kex_interm * opt_uv_type.KEX_INTERM

    try:
        input_table['OPT_UV_TYPE'] |= agn_mask
    except KeyError:
        input_table['OPT_UV_TYPE'] = agn_mask

    return input_table


def update_agntype_heii(input_table: Table, opt_uv_type: BitMask, snr: int | float = 3,
                        mask: MaskedColumn = None) -> Table:
    """Applies the HeII masks and sets the bitmasks for ``OPT_UV_TYPE``.

    Args:
        input_table: Table joined with FastSpecFit columns.
        opt_uv_type: DESI BitMask object containing the definitions of the ``OPT_UV_TYPE`` values.
        snr: Signal-to-noise cut applied to all flux axes. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Input table with new or updated column with ``OPT_UV_TYPE`` bit masks for HeII selections for all rows.
    """

    heii_bpt, agn_heii, sf_heii = uv_opt_agn.heii_bpt(input_table, snr=snr, mask=mask)

    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    agn_mask = heii_bpt * opt_uv_type.HEII_BPT
    agn_mask |= agn_heii * opt_uv_type.HEII_AGN
    agn_mask |= sf_heii * opt_uv_type.HEII_SF

    try:
        input_table['OPT_UV_TYPE'] |= agn_mask
    except KeyError:
        input_table['OPT_UV_TYPE'] = agn_mask

    return input_table


def update_agntype_nev(input_table: Table, opt_uv_type: BitMask, snr: int | float = 2.5,
                       mask: MaskedColumn = None) -> Table:
    """Applies the [NeV] masks and sets the bitmasks for ``OPT_UV_TYPE``.

    Args:
        input_table: Table joined with FastSpecFit columns.
        opt_uv_type: DESI BitMask object containing the definitions of the ``OPT_UV_TYPE`` values.
        snr: Signal-to-noise cut applied to all flux axes. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Input table with new or updated column with ``OPT_UV_TYPE`` bit masks for [NeV] selections for all rows.
    """

    nev, agn_nev, sf_nev = uv_opt_agn.nev(input_table, snr=snr, mask=mask)

    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    agn_mask = nev * opt_uv_type.NEV
    agn_mask |= agn_nev * opt_uv_type.NEV_AGN
    agn_mask |= sf_nev * opt_uv_type.NEV_SF

    try:
        input_table['OPT_UV_TYPE'] |= agn_mask
    except KeyError:
        input_table['OPT_UV_TYPE'] = agn_mask

    return input_table


def update_agntype_wise_jarrett11(input_table: Table, ir_type: BitMask, snr: int | float = 3,
                                mask: MaskedColumn = None) -> Table:
    """Applies the Jarrett+11 WISE AGN selection masks and sets the bitmasks for ``IR_TYPE``.

    Args:
        input_table: Table joined with FastSpecFit columns.
        ir_type: DESI BitMask object containing the definitions of the ``IR_TYPE`` values.
        snr: Signal-to-noise cut applied to all flux axes. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Input table with new or updated column with ``IR_TYPE`` bit masks for Jarrett+11 WISE color-color selection for
        all rows.
    """

    wise_w123, agn_ir, non_agn_ir = ir_agn.wise_jarrett11(input_table, snr=snr, mask=mask)

    # For the WISE selections, all galaxies will have flags for at least whether there was photometry available.
    # If W1, W2, W3 fluxes are above threshold snr (required for Jarrett+, Mateos+, Yao+, Hviding+)
    agn_mask = wise_w123 * ir_type.WISE_W123

    # Further, apply masks for if the galaxy is classified as an AGN or non-AGN (called "SF" in bitmask definitions.)
    agn_mask |= agn_ir * ir_type.WISE_AGN_J11
    agn_mask |= non_agn_ir * ir_type.WISE_SF_J11

    try:
        input_table['IR_TYPE'] |= agn_mask
    except KeyError:
        input_table['IR_TYPE'] = agn_mask

    return input_table

def update_agntype_wise_stern12(input_table: Table, ir_type: BitMask, snr: int | float = 3,
                                mask: MaskedColumn = None) -> Table:
    """Applies the Stern+12 WISE AGN selection masks and sets the bitmasks for ``IR_TYPE``.

    Args:
        input_table: Table joined with FastSpecFit columns.
        ir_type: DESI BitMask object containing the definitions of the ``IR_TYPE`` values.
        snr: Signal-to-noise cut applied to all flux axes. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Input table with new or updated column with ``IR_TYPE`` bit masks for Stern+12 WISE color selection for all
        rows.
    """

    wise_w12, agn_ir, non_agn_ir = ir_agn.wise_stern12(input_table, snr=snr, mask=mask)

    # For the WISE selections, all galaxies will have flags for at least whether there was photometry available.
    # If W1, W2 fluxes are above threshold snr (required for Stern+ and Assef+)
    agn_mask = wise_w12 * ir_type.WISE_W12

    # Further, apply masks for if the galaxy is classified as an AGN or non-AGN (called "SF" in bitmask definitions.)
    agn_mask |= agn_ir * ir_type.WISE_AGN_S12
    agn_mask |= non_agn_ir * ir_type.WISE_SF_S12

    try:
        input_table['IR_TYPE'] |= agn_mask
    except KeyError:
        input_table['IR_TYPE'] = agn_mask

    return input_table

def update_agntype_wise_mateos12(input_table: Table, ir_type: BitMask, snr: int | float = 3,
                                 mask: MaskedColumn = None) -> Table:
    """Applies the Mateos+12 WISE AGN selection masks and sets the bitmasks for ``IR_TYPE``.

    Args:
        input_table: Table joined with FastSpecFit columns.
        ir_type: DESI BitMask object containing the definitions of the ``IR_TYPE`` values.
        snr: Signal-to-noise cut applied to all flux axes. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Input table with new or updated column with ``IR_TYPE`` bit masks for Mateos+12 WISE color-color selection for
        all rows.
    """

    wise_w123, agn_ir, non_agn_ir = ir_agn.wise_mateos12(input_table, snr=snr, mask=mask)

    # For the WISE selections, all galaxies will have flags for at least whether there was photometry available.
    # If W1, W2, W3 fluxes are above threshold snr (required for Jarrett+, Mateos+, Yao+, Hviding+)
    agn_mask = wise_w123 * ir_type.WISE_W123

    # Further, apply masks for if the galaxy is classified as an AGN or non-AGN (called "SF" in bitmask definitions.)
    agn_mask |= agn_ir * ir_type.WISE_AGN_M12
    agn_mask |= non_agn_ir * ir_type.WISE_SF_M12

    try:
        input_table['IR_TYPE'] |= agn_mask
    except KeyError:
        input_table['IR_TYPE'] = agn_mask

    return input_table

def update_agntype_wise_assef18_r(input_table: Table, ir_type: BitMask, snr: int | float = 3,
                                 reliability: Literal[75, 90] = 90, mask: MaskedColumn = None) -> Table:
    """Applies the reliability-optimized Assef+18 WISE AGN selection masks and sets the bitmasks for ``IR_TYPE``.

    Args:
        input_table: Table joined with FastSpecFit columns.
        ir_type: DESI BitMask object containing the definitions of the ``IR_TYPE`` values.
        snr: Signal-to-noise cut applied to all flux axes. Default is ``3``.
        reliability: The reliability percent threshold to use for the selection. Must be either ``75`` or ``90``, the
            default is ``90``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Input table with new or updated column with ``IR_TYPE`` bit masks for the reliability-optimized Assef+18 WISE
        color-color selection for all rows.
    """

    wise_w12, agn_ir, non_agn_ir = ir_agn.wise_assef18_r(input_table, snr=snr, reliability=reliability, mask=mask)

    # For the WISE selections, all galaxies will have flags for at least whether there was photometry available.
    # If W1, W2 fluxes are above threshold snr (required for Stern+ and Assef+)
    agn_mask = wise_w12 * ir_type.WISE_W12

    # Further, apply masks for if the galaxy is classified as an AGN or non-AGN (called "SF" in bitmask definitions.)
    agn_mask |= agn_ir * ir_type.WISE_AGN_A18
    agn_mask |= non_agn_ir * ir_type.WISE_SF_A18

    try:
        input_table['IR_TYPE'] |= agn_mask
    except KeyError:
        input_table['IR_TYPE'] = agn_mask

    return input_table

def update_agntype_wise_yao20(input_table: Table, ir_type: BitMask, snr: int | float = 3,
                              mask: MaskedColumn = None) -> Table:
    """Applies the Yao+20 WISE AGN selection masks and sets the bitmasks for ``IR_TYPE``.

    Args:
        input_table: Table joined with FastSpecFit columns.
        ir_type: DESI BitMask object containing the definitions of the ``IR_TYPE`` values.
        snr: Signal-to-noise cut applied to all flux axes. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Input table with new or updated column with ``IR_TYPE`` bit masks for the reliability-optimized Yao+20 WISE
        color-color selection for all rows.
    """

    # We will not implement the "weak AGN" selection here to preserve compatability with existing bitmasks
    wise_w123, agn_ir, non_agn_ir = ir_agn.wise_yao20(input_table, snr=snr, weak_agn=False, mask=mask)

    # For WISE selections, all galaxies will have flags for at least whether there was photometry available.
    # If W1, W2, W3 fluxes are above threshold snr (required for Jarrett+, Mateos+, Yao+, Hviding+)
    agn_mask = wise_w123 * ir_type.WISE_W123

    # Further, apply masks for if the galaxy is classified as an AGN or non-AGN (called "SF" in bitmask definitions.)
    agn_mask |= agn_ir * ir_type.WISE_AGN_Y20
    agn_mask |= non_agn_ir * ir_type.WISE_SF_Y20

    try:
        input_table['IR_TYPE'] |= agn_mask
    except KeyError:
        input_table['IR_TYPE'] = agn_mask

    return input_table


def update_agntype_wise_hviding22(input_table: Table, ir_type: BitMask, snr: int | float = 3,
                                  mask: MaskedColumn = None) -> Table:
    """Applies the Hviding+22 WISE AGN selection masks and sets the bitmasks for ``IR_TYPE``.

    Args:
        input_table: Table joined with FastSpecFit columns.
        ir_type: DESI BitMask object containing the definitions of the ``IR_TYPE`` values.
        snr: Signal-to-noise cut applied to all flux axes. Default is ``3``.
        mask: Optional mask (e.g., from masked column array). Default is ``None``.

    Returns:
        Input table with new or updated column with ``IR_TYPE`` bit masks for Hviding+22 WISE color-color selection for
        all rows.
    """

    wise_w123, agn_ir, sf_ir = ir_agn.wise_hviding22(input_table, snr=snr, mask=mask)

    # For WISE selections, all galaxies will have flags for at least whether there was photometry available.
    # If W1, W2, W3 fluxes are above threshold snr (required for Jarrett+, Mateos+, Yao+, Hviding+)
    agn_mask = wise_w123 * ir_type.WISE_W123

    # Further, apply masks for if the galaxy is classified as an AGN or non-AGN (called "SF" in bitmask definitions.)
    agn_mask |= agn_ir * ir_type.WISE_AGN_H22
    agn_mask |= sf_ir * ir_type.WISE_SF_H22

    try:
        input_table['IR_TYPE'] |= agn_mask
    except KeyError:
        input_table['IR_TYPE'] = agn_mask

    return input_table
