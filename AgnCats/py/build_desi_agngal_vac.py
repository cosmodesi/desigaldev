"""
build_desi_agngal_vac.py
Author: Benjamin Floyd

This top-level script builds the DESI AGN/Galaxy Classification VAC. This supersedes the DR1 00_AGNQSO_summary_cat.ipynb
notebook and provides parallelized computation abilities in constructing the final catalog.
"""

from pathlib import Path

import fitsio
from astropy.table import Table, hstack, join

from AgnCats.py import set_agn_masksDESI as agn_masks

# First we want to build a dispatch pattern to handle the various file selections between data releases
desi_specprod = {
    # EDR
    'fuji': {
        # QSO-Maker catalog from Edmonds catalog keeping all columns
        'qso_maker': Path('/global/cfs/cdirs/desi/users/edmondc/QSO_catalog/fuji/'
                          'QSO_cat_fuji_healpix_all_targets_v2.fits'),

        # FastSpecFit catalog
        'fast_spec': Path('/global/cfs/cdirs/desi/spectro/fastspecfit/fuji/v3.2/catalogs/fastspec-fuji.fits'),

        # Redshift catalog
        'zcat': Path('/global/cfs/cdirs/desi/public/edr/vac/edr/zcat/fuji/v1.0/zall-pix-edr-vac.fits'),
        'zcat_cols': ['TARGETID','SURVEY','PROGRAM','HEALPIX','TSNR2_LRG','SV_NSPEC','SV_PRIMARY',
                      'ZCAT_NSPEC','ZCAT_PRIMARY','MIN_MJD','MEAN_MJD','MAX_MJD', 'OBJTYPE']
    },
    # DR1
    'iron': {
        # QSO-Maker catalog from `merge_QSOmaker.ipynb`. DR1 version from after Edmond ran on all targets/all surveys
        'qso_maker': Path('/global/cfs/cdirs/desi/science/gqp/agncatalog/qsomaker/iron/'
                          'QSO_cat_iron_healpix_all_targets_v1.fits'),

        # FastSpecFit catalog
        'fast_spec': Path('/global/cfs/cdirs/desi/spectro/fastspecfit/iron/v2.1/catalogs/fastspec-iron.fits'),

        # Redshift catalog
        'zcat': Path('/global/cfs/cdirs/desi/spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits'),
        'zcat_cols': ['TARGETID','SURVEY','PROGRAM','HEALPIX','TSNR2_LRG','ZCAT_NSPEC','ZCAT_PRIMARY',
                      'SV_NSPEC','SV_PRIMARY','MAIN_PRIMARY','MAIN_NSPEC','MIN_MJD','MEAN_MJD','MAX_MJD','OBJTYPE']
    },
    # DR2
    'loa': {
        # QSO-Maker
        # FastSpecFit Catalog
        # Redshift Catalog
    }
}

qso_maker_cols = ['TARGETID', 'Z', 'ZERR', 'ZWARN', 'SPECTYPE', 'COADD_FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC',
                  'MORPHTYPE', 'MASKBITS', 'COADD_NUMEXP', 'COADD_EXPTIME', 'TSNR2_LYA', 'TSNR2_QSO',
                  'Z_RR', 'Z_QN', 'C_LYA', 'C_CIV', 'C_CIII', 'C_MgII', 'C_Hbeta', 'C_Halpha',
                  'QSO_MASKBITS', 'SURVEY', 'PROGRAM']



zcat_cols = ['DESI_TARGET','BGS_TARGET','SCND_TARGET','CMX_TARGET','SV1_DESI_TARGET','SV1_BGS_TARGET','SV1_SCND_TARGET',
             'SV2_DESI_TARGET','SV2_BGS_TARGET','SV2_SCND_TARGET','SV3_DESI_TARGET','SV3_BGS_TARGET','SV3_SCND_TARGET']



fsf_data_cols=['TARGETID','SURVEY','PROGRAM','LOGMSTAR',
               'CIV_1549_FLUX','CIV_1549_FLUX_IVAR', 'CIV_1549_SIGMA',
               'MGII_2796_FLUX','MGII_2796_FLUX_IVAR', 'MGII_2796_SIGMA',
               'MGII_2803_FLUX','MGII_2803_FLUX_IVAR', 'MGII_2803_SIGMA',
               'OII_3726_FLUX','OII_3726_FLUX_IVAR','OII_3726_EW','OII_3726_EW_IVAR',
               'OII_3729_FLUX','OII_3729_FLUX_IVAR','OII_3729_EW','OII_3729_EW_IVAR',
               'NEV_3426_FLUX','NEV_3426_FLUX_IVAR',
               'HEII_4686_FLUX','HEII_4686_FLUX_IVAR',
               'HBETA_EW','HBETA_EW_IVAR','HBETA_FLUX','HBETA_FLUX_IVAR',
               'HBETA_BROAD_FLUX', 'HBETA_BROAD_FLUX_IVAR', 'HBETA_BROAD_SIGMA','HBETA_BROAD_CHI2',
               'OIII_5007_FLUX','OIII_5007_FLUX_IVAR','OIII_5007_SIGMA',
               'OI_6300_FLUX','OI_6300_FLUX_IVAR',
               'HALPHA_EW','HALPHA_EW_IVAR', 'HALPHA_FLUX','HALPHA_FLUX_IVAR',
               'HALPHA_BROAD_FLUX','HALPHA_BROAD_FLUX_IVAR','HALPHA_BROAD_VSHIFT','HALPHA_BROAD_SIGMA',
               'NII_6584_FLUX','NII_6584_FLUX_IVAR',
               'SII_6716_FLUX','SII_6716_FLUX_IVAR',
               'SII_6731_FLUX','SII_6731_FLUX_IVAR']

fsf_meta_cols=['TARGETID','SURVEY','PROGRAM','PHOTSYS','LS_ID',
               'FIBERFLUX_G','FIBERFLUX_R','FIBERFLUX_Z','FIBERTOTFLUX_G','FIBERTOTFLUX_R','FIBERTOTFLUX_Z',
               'FLUX_G','FLUX_R','FLUX_Z','FLUX_W1','FLUX_W2','FLUX_W3','FLUX_W4',
               'FLUX_IVAR_G','FLUX_IVAR_R','FLUX_IVAR_Z','FLUX_IVAR_W1','FLUX_IVAR_W2','FLUX_IVAR_W3','FLUX_IVAR_W4',
               'EBV','MW_TRANSMISSION_G','MW_TRANSMISSION_R','MW_TRANSMISSION_Z',
               'MW_TRANSMISSION_W1','MW_TRANSMISSION_W2','MW_TRANSMISSION_W3','MW_TRANSMISSION_W4']


def read_fastspecfit(fastspec_path: Path, fastspec_data_colnames: list[str],
                     fastspec_meta_colnames: list[str]) -> Table:
    """Reads and merges the FastSpecFit catalog extensions into a single table.

    Args:
        fastspec_path:
            Path to the FastSpecFit catalog.
        fastspec_data_colnames:
            Data Extension columns to be read in.
        fastspec_meta_colnames:
            Meta Extension columns to be read in.

    Returns:
        Merged table of the two extensions.
    """

    # Read in the two extensions and cast as a table.
    fastspec_data_catalog = Table(fitsio.read(str(fastspec_path), columns=fastspec_data_colnames, ext=1))
    fastspec_meta_catalog = Table(fitsio.read(str(fastspec_path), columns=fastspec_meta_colnames, ext=2))

    # Remove any common columns between the extensions.
    fastspec_meta_catalog.remove_columns(set(fastspec_data_colnames).intersection(fastspec_meta_colnames))

    # As the two extensions are already row-aligned we can do a fast hstack operation rather than a full join.
    fastspec_catalog = hstack([fastspec_data_catalog, fastspec_meta_catalog])

    return fastspec_catalog


def read_input_catalogs(specprod_info: dict[str, Path | list[str]], fastspec_data_colnames: list[str],
                        fastspec_meta_colnames: list[str], qsom_colnames: list[str],
                        redshift_colnames: list[str]) -> Table:
    """Reads in the input catalogs and merges them into a single table to be used for AGN/Galaxy classification.

    Args:
        specprod_info:
            Dictionary with information about the targeted data release. Must include path names to relevant catalogs
            and associated data-release specific column names.
        fastspec_data_colnames:
            List of column names to be read in for the FastSpecFit catalog (data extension).
        fastspec_meta_colnames:
            List of column names to be read in for the FastSpecFit catalog (meta extension).
        qsom_colnames:
            List of column names to be read in for the QSO-Maker catalog.
        redshift_colnames:
            List of column names to be read in for the redshift catalog.

    Returns:
        Joined table of the three input catalogs.
    """

    # Read in and merge the FastSpecFit catalog extensions into a combined table
    fastspec_catalog = read_fastspecfit(specprod_info['fast_spec'], fastspec_data_colnames, fastspec_meta_colnames)

    # Read in the QSO-Maker catalog
    qso_maker_catalog = Table(fitsio.read(str(specprod_info['qso_maker']), ext=1, columns=qsom_colnames))

    # Read in the Redshift catalog (columns used will be the data-release specific columns and global columns)
    redshift_catalog = Table(fitsio.read(str(specprod_info['zcat']), ext=1,
                                         columns=redshift_colnames + specprod_info['zcat_cols']))

    # Main identifiers for Joins
    keys_for_join = ['TARGETID', 'SURVEY', 'PROGRAM']

    # Join FastSpecFit with QSO-Maker
    desi_catalog = join(fastspec_catalog, qso_maker_catalog, keys=keys_for_join, join_type='left')

    # Test for consistency
    try:
        assert all(desi_catalog['Z'] < 0.001)
    except AssertionError:
        raise ValueError('Joined FastSpecFit + QSO-Maker catalog contains objects z < 0.001') from AssertionError

    try:
        assert all(desi_catalog['COADD_EXPTIME'] > 0.0)
    except AssertionError:
        raise ValueError('Joined FastSpecFit + QSO-Maker catalog contains objects '
                         'with zero coadd exposure time') from AssertionError

    # Join the FastSpecFit+QSO-Maker catalog with the redshift catalog
    desi_catalog = join(desi_catalog, redshift_catalog, keys=keys_for_join, join_type='left')

    # Test for consistency
    try:
        assert all(desi_catalog['OBJTYPE'] == 'TGT')
    except AssertionError:
        raise ValueError('Joined FastSpecFit + QSO-Maker + Redshift catalog contains '
                         'non "TGT" object types') from AssertionError

    return desi_catalog

# Begin AGN/Galaxy classifications
def apply_agngal_class(input_table: Table, agnmask_defs: Path) -> Table:
    """Applies the AGN/Galaxy classification definitions and adds bitmasks to the input table.

    Args:
        input_table:
            Table containing spectroscopic and photometric fluxes and inverse variances.
        agnmask_defs:
            Path to YAML file containing AGN/Galaxy classification definitions.

    Returns:
        Input table with AGN/Galaxy classification bitmask columns added.
    """

    # Read in the bit mask definitions
    agn_maskbits, uv_opt_type, ir_type = agn_masks.get_agn_maskbits(agnmask_defs)

    # Apply the AGN_MASKBITS to the catalog
    desi_catalog = agn_masks.update_agn_maskbits(input_table, agn_maskbits, snr=3, snr_oi=1, snr_wise=3, kewley01=False)

    # Apply the BPT UV_OPT_TYPE maskbits
    desi_catalog = agn_masks.update_agntype_nii_bpt(desi_catalog, uv_opt_type, snr=3)
    desi_catalog = agn_masks.update_agntype_sii_bpt(desi_catalog, uv_opt_type, snr=3, kewley01=False)
    desi_catalog = agn_masks.update_agntype_oi_bpt(desi_catalog, uv_opt_type, snr=3, snr_oi=1, kewley01=False)

    # Apply the non-BPT optical maskbits
    desi_catalog = agn_masks.update_agntype_whan(desi_catalog, uv_opt_type, snr=3)
    desi_catalog = agn_masks.update_agntype_blue(desi_catalog, uv_opt_type, snr=3, snr_oii=1)
    desi_catalog = agn_masks.update_agntype_mex(desi_catalog, uv_opt_type, snr=3)
    desi_catalog = agn_masks.update_agntype_kex(desi_catalog, uv_opt_type, snr=3)
    desi_catalog = agn_masks.update_agntype_heii(desi_catalog, uv_opt_type, snr=3)
    desi_catalog = agn_masks.update_agntype_nev(desi_catalog, uv_opt_type, snr=3)

    # Apply the WISE IR-selection maskbits
    desi_catalog = agn_masks.update_agntype_wise_stern12(desi_catalog, ir_type, snr=3)
    desi_catalog = agn_masks.update_agntype_wise_mateos12(desi_catalog, ir_type, snr=3)
    desi_catalog = agn_masks.update_agntype_wise_assef18_r(desi_catalog, ir_type, snr=3, reliability=90)
    desi_catalog = agn_masks.update_agntype_wise_yao20(desi_catalog, ir_type, snr=3)
    desi_catalog = agn_masks.update_agntype_wise_hviding22(desi_catalog, ir_type, snr=3)

    return desi_catalog


