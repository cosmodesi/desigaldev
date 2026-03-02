"""
build_desi_agngal_vac.py
Author: Benjamin Floyd

This top-level script builds the DESI AGN/Galaxy Classification VAC. This supersedes the DR1 00_AGNQSO_summary_cat.ipynb
notebook and provides parallelized computation abilities in constructing the final catalog.
"""

import sys

sys.path.append('/global/homes/b/bfloyd/agngal_dr2')

import re
from argparse import ArgumentParser
from itertools import groupby
from multiprocessing import Pool
from pathlib import Path

import fitsio
from astropy.io import fits
from astropy.table import Table, hstack, join
from desiutil.annotate import annotate_fits, load_yml_units

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
        'fast_spec_data_cols': ['TARGETID', 'SURVEY', 'PROGRAM', 'LOGMSTAR',
                                'CIV_1549_FLUX', 'CIV_1549_FLUX_IVAR', 'CIV_1549_SIGMA',
                                'MGII_2796_FLUX', 'MGII_2796_FLUX_IVAR', 'MGII_2796_SIGMA',
                                'MGII_2803_FLUX', 'MGII_2803_FLUX_IVAR', 'MGII_2803_SIGMA',
                                'OII_3726_FLUX', 'OII_3726_FLUX_IVAR', 'OII_3726_EW', 'OII_3726_EW_IVAR',
                                'OII_3729_FLUX', 'OII_3729_FLUX_IVAR', 'OII_3729_EW', 'OII_3729_EW_IVAR',
                                'NEV_3426_FLUX', 'NEV_3426_FLUX_IVAR',
                                'HEII_4686_FLUX', 'HEII_4686_FLUX_IVAR',
                                'HBETA_EW', 'HBETA_EW_IVAR', 'HBETA_FLUX', 'HBETA_FLUX_IVAR',
                                'HBETA_BROAD_FLUX', 'HBETA_BROAD_FLUX_IVAR', 'HBETA_BROAD_SIGMA', 'HBETA_BROAD_CHI2',
                                'OIII_5007_FLUX', 'OIII_5007_FLUX_IVAR', 'OIII_5007_SIGMA',
                                'OI_6300_FLUX', 'OI_6300_FLUX_IVAR',
                                'HALPHA_EW', 'HALPHA_EW_IVAR', 'HALPHA_FLUX', 'HALPHA_FLUX_IVAR',
                                'HALPHA_BROAD_FLUX', 'HALPHA_BROAD_FLUX_IVAR', 'HALPHA_BROAD_VSHIFT',
                                'HALPHA_BROAD_SIGMA',
                                'NII_6584_FLUX', 'NII_6584_FLUX_IVAR',
                                'SII_6716_FLUX', 'SII_6716_FLUX_IVAR',
                                'SII_6731_FLUX', 'SII_6731_FLUX_IVAR'],

        'fast_spec_meta_cols': ['TARGETID', 'SURVEY', 'PROGRAM', 'PHOTSYS', 'LS_ID',
                                'FIBERFLUX_G', 'FIBERFLUX_R', 'FIBERFLUX_Z', 'FIBERTOTFLUX_G', 'FIBERTOTFLUX_R',
                                'FIBERTOTFLUX_Z',
                                'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4',
                                'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_IVAR_W1', 'FLUX_IVAR_W2',
                                'FLUX_IVAR_W3',
                                'FLUX_IVAR_W4',
                                'EBV', 'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z',
                                'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2', 'MW_TRANSMISSION_W3', 'MW_TRANSMISSION_W4'],

        # Redshift catalog
        'zcat': Path('/global/cfs/cdirs/desi/public/edr/vac/edr/zcat/fuji/v1.0/zall-pix-edr-vac.fits'),
        'zcat_cols': ['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX', 'TSNR2_LRG', 'SV_NSPEC', 'SV_PRIMARY',
                      'ZCAT_NSPEC', 'ZCAT_PRIMARY', 'MIN_MJD', 'MEAN_MJD', 'MAX_MJD', 'OBJTYPE'],

        # Output catalog extension 1 column names
        'output_cols_ext1': ['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX',
                             'Z', 'ZERR', 'ZWARN', 'SPECTYPE',
                             'AGN_MASKBITS', 'OPT_UV_TYPE', 'IR_TYPE',
                             'COADD_FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC', 'LS_ID',
                             'MIN_MJD', 'MEAN_MJD', 'MAX_MJD', 'COADD_NUMEXP', 'COADD_EXPTIME',
                             'SV_PRIMARY', 'ZCAT_PRIMARY',
                             'DESI_TARGET', 'SCND_TARGET', 'BGS_TARGET', 'CMX_TARGET',
                             'SV1_DESI_TARGET', 'SV2_DESI_TARGET', 'SV3_DESI_TARGET',
                             'SV1_BGS_TARGET', 'SV2_BGS_TARGET', 'SV3_BGS_TARGET',
                             'SV1_SCND_TARGET', 'SV2_SCND_TARGET', 'SV3_SCND_TARGET']
    },
    # DR1
    'iron': {
        # QSO-Maker catalog from `merge_QSOmaker.ipynb`. DR1 version from after Edmond ran on all targets/all surveys
        'qso_maker': Path('/global/cfs/cdirs/desi/science/gqp/agncatalog/qsomaker/iron/'
                          'QSO_cat_iron_healpix_all_targets_v1.fits'),

        # FastSpecFit catalog
        'fast_spec': Path('/global/cfs/cdirs/desi/spectro/fastspecfit/iron/v2.1/catalogs/fastspec-iron.fits'),
        'fast_spec_data_cols': ['TARGETID', 'SURVEY', 'PROGRAM', 'LOGMSTAR',
                                'CIV_1549_FLUX', 'CIV_1549_FLUX_IVAR', 'CIV_1549_SIGMA',
                                'MGII_2796_FLUX', 'MGII_2796_FLUX_IVAR', 'MGII_2796_SIGMA',
                                'MGII_2803_FLUX', 'MGII_2803_FLUX_IVAR', 'MGII_2803_SIGMA',
                                'NEV_3426_FLUX', 'NEV_3426_FLUX_IVAR',
                                'OII_3726_FLUX', 'OII_3726_FLUX_IVAR', 'OII_3726_EW', 'OII_3726_EW_IVAR',
                                'OII_3729_FLUX', 'OII_3729_FLUX_IVAR', 'OII_3729_EW', 'OII_3729_EW_IVAR',
                                'HEII_4686_FLUX', 'HEII_4686_FLUX_IVAR',
                                'HBETA_EW', 'HBETA_EW_IVAR', 'HBETA_FLUX', 'HBETA_FLUX_IVAR',
                                'HBETA_BROAD_FLUX', 'HBETA_BROAD_FLUX_IVAR', 'HBETA_BROAD_SIGMA', 'HBETA_BROAD_CHI2',
                                'OIII_5007_FLUX', 'OIII_5007_FLUX_IVAR', 'OIII_5007_SIGMA',
                                'OI_6300_FLUX', 'OI_6300_FLUX_IVAR',
                                'HALPHA_EW', 'HALPHA_EW_IVAR', 'HALPHA_FLUX', 'HALPHA_FLUX_IVAR',
                                'HALPHA_BROAD_FLUX', 'HALPHA_BROAD_FLUX_IVAR', 'HALPHA_BROAD_VSHIFT',
                                'HALPHA_BROAD_SIGMA',
                                'NII_6584_FLUX', 'NII_6584_FLUX_IVAR',
                                'SII_6716_FLUX', 'SII_6716_FLUX_IVAR',
                                'SII_6731_FLUX', 'SII_6731_FLUX_IVAR'],

        'fast_spec_meta_cols': ['TARGETID', 'LS_ID', 'SURVEY', 'PROGRAM', 'PHOTSYS',
                                'FIBERFLUX_G', 'FIBERFLUX_R', 'FIBERFLUX_Z', 'FIBERTOTFLUX_G', 'FIBERTOTFLUX_R',
                                'FIBERTOTFLUX_Z',
                                'FLUX_G', 'FLUX_IVAR_G',
                                'FLUX_R', 'FLUX_IVAR_R',
                                'FLUX_Z', 'FLUX_IVAR_Z',
                                'FLUX_W1', 'FLUX_IVAR_W1',
                                'FLUX_W2', 'FLUX_IVAR_W2',
                                'FLUX_W3', 'FLUX_IVAR_W3',
                                'FLUX_W4', 'FLUX_IVAR_W4',
                                'EBV',
                                'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z',
                                'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2', 'MW_TRANSMISSION_W3', 'MW_TRANSMISSION_W4'],

        # Redshift catalog
        'zcat': Path('/global/cfs/cdirs/desi/spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits'),
        'zcat_cols': ['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX', 'TSNR2_LRG', 'ZCAT_NSPEC', 'ZCAT_PRIMARY',
                      'SV_NSPEC', 'SV_PRIMARY', 'MAIN_PRIMARY', 'MAIN_NSPEC', 'MIN_MJD', 'MEAN_MJD', 'MAX_MJD',
                      'OBJTYPE'],

        # Output catalog extension 1 column names
        'output_cols_ext1': ['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX',
                             'Z', 'ZERR', 'ZWARN', 'SPECTYPE',
                             'AGN_MASKBITS', 'OPT_UV_TYPE', 'IR_TYPE',
                             'COADD_FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC', 'LS_ID',
                             'MIN_MJD', 'MEAN_MJD', 'MAX_MJD', 'COADD_NUMEXP', 'COADD_EXPTIME',
                             'SV_PRIMARY', 'MAIN_PRIMARY', 'ZCAT_PRIMARY',
                             'DESI_TARGET', 'SCND_TARGET', 'BGS_TARGET', 'CMX_TARGET',
                             'SV1_DESI_TARGET', 'SV2_DESI_TARGET', 'SV3_DESI_TARGET',
                             'SV1_BGS_TARGET', 'SV2_BGS_TARGET', 'SV3_BGS_TARGET',
                             'SV1_SCND_TARGET', 'SV2_SCND_TARGET', 'SV3_SCND_TARGET']
    },
    # DR2
    'loa': {
        # QSO-Maker
        'qso_maker_dir': Path('/global/cfs/cdirs/desi/science/gqp/agncatalog/qsomaker/loa'),

        # FastSpecFit Catalog
        'fast_spec_dir': Path('/global/cfs/cdirs/desi/vac/dr2/fastspecfit/loa/v1.0/catalogs'),
        'fast_spec_data_cols': ['TARGETID', 'PROGRAM', 'SURVEY',
                                'CIV_1549_FLUX', 'CIV_1549_FLUX_IVAR', 'CIV_1549_SIGMA',
                                'MGII_2796_FLUX', 'MGII_2796_FLUX_IVAR', 'MGII_2796_SIGMA',
                                'MGII_2803_FLUX', 'MGII_2803_FLUX_IVAR', 'MGII_2803_SIGMA',
                                'NEV_3426_FLUX', 'NEV_3426_FLUX_IVAR',
                                'OII_3726_EW', 'OII_3726_EW_IVAR', 'OII_3726_FLUX', 'OII_3726_FLUX_IVAR',
                                'OII_3729_EW', 'OII_3729_EW_IVAR', 'OII_3729_FLUX', 'OII_3729_FLUX_IVAR',
                                'HEII_4686_FLUX', 'HEII_4686_FLUX_IVAR',
                                'HBETA_FLUX', 'HBETA_FLUX_IVAR', 'HBETA_EW', 'HBETA_EW_IVAR',
                                'HBETA_BROAD_CHI2', 'HBETA_BROAD_FLUX', 'HBETA_BROAD_FLUX_IVAR', 'HBETA_BROAD_SIGMA',
                                'OIII_5007_FLUX', 'OIII_5007_FLUX_IVAR', 'OIII_5007_SIGMA',
                                'HALPHA_FLUX', 'HALPHA_FLUX_IVAR', 'HALPHA_EW', 'HALPHA_EW_IVAR',
                                'HALPHA_BROAD_FLUX', 'HALPHA_BROAD_FLUX_IVAR', 'HALPHA_BROAD_SIGMA',
                                'HALPHA_BROAD_VSHIFT',
                                'NII_6584_FLUX', 'NII_6584_FLUX_IVAR', 'OI_6300_FLUX', 'OI_6300_FLUX_IVAR',
                                'SII_6716_FLUX', 'SII_6716_FLUX_IVAR',
                                'SII_6731_FLUX', 'SII_6731_FLUX_IVAR'],

        'fast_spec_meta_cols': ['TARGETID', 'LS_ID', 'PROGRAM', 'SURVEY', 'PHOTSYS',
                                'FIBERFLUX_G', 'FIBERFLUX_R', 'FIBERFLUX_Z',
                                'FIBERTOTFLUX_G', 'FIBERTOTFLUX_R', 'FIBERTOTFLUX_Z',
                                'FLUX_G', 'FLUX_IVAR_G',
                                'FLUX_R', 'FLUX_IVAR_R',
                                'FLUX_Z', 'FLUX_IVAR_Z',
                                'FLUX_W1', 'FLUX_IVAR_W1',
                                'FLUX_W2', 'FLUX_IVAR_W2',
                                'FLUX_W3', 'FLUX_IVAR_W3',
                                'FLUX_W4', 'FLUX_IVAR_W4',
                                'EBV',
                                'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z',
                                'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2', 'MW_TRANSMISSION_W3', 'MW_TRANSMISSION_W4'],

        'fast_spec_specphot_cols': ['TARGETID', 'PROGRAM', 'SURVEY', 'LOGMSTAR'],

        # Redshift Catalog
        'zcat_dir': Path('/global/cfs/cdirs/desi/science/gqp/agncatalog/zpix_nside1/loa/v1'),
        'zcat_cols': ['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX', 'TSNR2_LRG', 'ZCAT_NSPEC', 'ZCAT_PRIMARY',
                      'SV_NSPEC', 'SV_PRIMARY', 'MAIN_PRIMARY', 'MAIN_NSPEC', 'MIN_MJD', 'MEAN_MJD', 'MAX_MJD',
                      'OBJTYPE'],

        # Output catalog extension 1 column names
        'output_cols_ext1': ['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX',
                             'Z', 'ZERR', 'ZWARN', 'SPECTYPE',
                             'AGN_MASKBITS', 'OPT_UV_TYPE', 'IR_TYPE',
                             'COADD_FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC', 'LS_ID',
                             'MIN_MJD', 'MEAN_MJD', 'MAX_MJD', 'COADD_NUMEXP', 'COADD_EXPTIME',
                             'SV_PRIMARY', 'MAIN_PRIMARY', 'ZCAT_PRIMARY',
                             'DESI_TARGET', 'SCND_TARGET', 'BGS_TARGET', 'CMX_TARGET',
                             'SV1_DESI_TARGET', 'SV2_DESI_TARGET', 'SV3_DESI_TARGET',
                             'SV1_BGS_TARGET', 'SV2_BGS_TARGET', 'SV3_BGS_TARGET',
                             'SV1_SCND_TARGET', 'SV2_SCND_TARGET', 'SV3_SCND_TARGET']
    }
}

# AGN BitMask Definitions file
agn_bitmask_defs = Path('/global/u2/b/bfloyd/agngal_dr2/AgnCats/py/agnmask.yaml')

# Output file unit definitions files
output_ext1_unit_defs = Path('/global/u2/b/bfloyd/agngal_dr2/AgnCats/py/ext1_units.yaml')
output_ext2_unit_defs = Path('/global/u2/b/bfloyd/agngal_dr2/AgnCats/py/ext2_units.yaml')

# Universal input catalog column names
qso_maker_cols = ['TARGETID', 'Z', 'ZERR', 'ZWARN', 'SPECTYPE', 'COADD_FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC',
                  'MORPHTYPE', 'MASKBITS', 'COADD_NUMEXP', 'COADD_EXPTIME', 'TSNR2_LYA', 'TSNR2_QSO',
                  'Z_RR', 'Z_QN', 'C_LYA', 'C_CIV', 'C_CIII', 'C_MgII', 'C_Hbeta', 'C_Halpha',
                  'QSO_MASKBITS', 'SURVEY', 'PROGRAM']

zcat_cols = ['DESI_TARGET', 'BGS_TARGET', 'SCND_TARGET', 'CMX_TARGET',
             'SV1_DESI_TARGET', 'SV1_BGS_TARGET', 'SV1_SCND_TARGET',
             'SV2_DESI_TARGET', 'SV2_BGS_TARGET', 'SV2_SCND_TARGET',
             'SV3_DESI_TARGET', 'SV3_BGS_TARGET', 'SV3_SCND_TARGET']

# Universal output catalog column names
output_cols_ext2 = ['TARGETID', 'SURVEY', 'PROGRAM', 'LOGMSTAR',
                    'FLUX_W1', 'FLUX_W2', 'FLUX_W3',
                    'FLUX_IVAR_W1', 'FLUX_IVAR_W2', 'FLUX_IVAR_W3',
                    'CIV_1549_FLUX', 'CIV_1549_FLUX_IVAR', 'CIV_1549_SIGMA',
                    'MGII_2796_FLUX', 'MGII_2796_FLUX_IVAR', 'MGII_2796_SIGMA',
                    'MGII_2803_FLUX', 'MGII_2803_FLUX_IVAR', 'MGII_2803_SIGMA',
                    'OII_3726_FLUX', 'OII_3726_FLUX_IVAR', 'OII_3726_EW', 'OII_3726_EW_IVAR',
                    'OII_3729_FLUX', 'OII_3729_FLUX_IVAR', 'OII_3729_EW', 'OII_3729_EW_IVAR',
                    'NEV_3426_FLUX', 'NEV_3426_FLUX_IVAR',
                    'HEII_4686_FLUX', 'HEII_4686_FLUX_IVAR',
                    'HBETA_EW', 'HBETA_EW_IVAR', 'HBETA_FLUX', 'HBETA_FLUX_IVAR',
                    'HBETA_BROAD_FLUX', 'HBETA_BROAD_FLUX_IVAR', 'HBETA_BROAD_SIGMA', 'HBETA_BROAD_CHI2',
                    'OIII_5007_FLUX', 'OIII_5007_FLUX_IVAR', 'OIII_5007_SIGMA',
                    'OI_6300_FLUX', 'OI_6300_FLUX_IVAR',
                    'HALPHA_EW', 'HALPHA_EW_IVAR', 'HALPHA_FLUX', 'HALPHA_FLUX_IVAR',
                    'HALPHA_BROAD_FLUX', 'HALPHA_BROAD_FLUX_IVAR', 'HALPHA_BROAD_VSHIFT', 'HALPHA_BROAD_SIGMA',
                    'NII_6584_FLUX', 'NII_6584_FLUX_IVAR',
                    'SII_6716_FLUX', 'SII_6716_FLUX_IVAR',
                    'SII_6731_FLUX', 'SII_6731_FLUX_IVAR']


def generate_loa_dispatchers(specprod_info: dict[str, Path | list[str]]) -> dict[str, dict[str, Path | list[str]]]:
    """Processes the DR2 suite of catalogs into dispatchers matching the scheme for EDR and DR1.

    Args:
        specprod_info:
            Dictionary containing paths to directories to find the relevant catalogs for DR2. Additionally, any
            associated data-release specific column names that should be included in the final dispatcher.

    Returns:
        Dispatch patterns matching required path names and column names for catalog input. Each dispatch pattern is
        identified by the DESI ``<Survey>-<Program>[-nside1-hp<HEALPix>]`` naming scheme e.g.,
        "main-dark-nside1-hp00" or "sv1-bright".
    """

    # Define a RegEx pattern for the file grouping names
    file_pattern = re.compile(r'(cmx|main|special|sv1|sv2|sv3)[-_](backup|bright|dark|other)([-_]nside1[-_]hp[0-9]*)?')

    # Group all the input catalogs by survey-program-healpix into a dictionary of lists of file paths
    all_catalogs = [*list(specprod_info['fast_spec_dir'].glob('*.fits')),
                    *list(specprod_info['qso_maker_dir'].glob('*.fits')),
                    *list(specprod_info['zcat_dir'].glob('*.fits'))]
    all_catalogs = sorted([file_path for file_path in all_catalogs if file_pattern.search(str(file_path))],
                          key=lambda f: file_pattern.search(str(f).replace('_', '-')).group(0))
    all_catalogs_grp = groupby(all_catalogs, key=lambda f: file_pattern.search(str(f).replace('_', '-')).group(0))
    all_catalogs_dict = {catalog_name: list(file_paths) for catalog_name, file_paths in all_catalogs_grp}

    # Remove the full "main-bright" and "main-dark" entries in our grouped dictionary.
    # Due to extra files being present in QSO-Maker directory.
    del all_catalogs_dict['main-bright']
    del all_catalogs_dict['main-dark']

    # Convert the lists of file paths into dictionaries with the same structure as the dispatch patterns for previous
    # data releases
    loa_dispatchers = {survey_program: {**loa_paths(catalog_paths, specprod_info),
                                        'fast_spec_data_cols': specprod_info['fast_spec_data_cols'],
                                        'fast_spec_meta_cols': specprod_info['fast_spec_meta_cols'],
                                        'fast_spec_specphot_cols': specprod_info['fast_spec_specphot_cols'],
                                        'zcat_cols': specprod_info['zcat_cols'],
                                        'output_cols_ext1': specprod_info['output_cols_ext1']}
                       for survey_program, catalog_paths in all_catalogs_dict.items()}

    return loa_dispatchers


def loa_paths(file_paths: list[Path], specprod_info: dict[str, Path]) -> dict[str, Path]:
    """Parse a list of file paths into a dictionary with keys matching the expected dispatch pattern

    Args:
        file_paths:
            A list of file paths describing the path name to the FastSpecFit, QSO-Maker, and Redshift catalogs.
        specprod_info:
            A dictionary containing keys indicating the paths to the directories for the associated catalogs. This will
            be used to filter the paths into the appropriate categories.

    Returns:
        A dictionary with structure matching the expected dispatch patterns for input catalogs.

    Raises:
        ValueError: If a path not matching the correct parent directories is found.
    """

    file_dict = {}
    for file_path in file_paths:
        match file_path.parent:
            case path if path == specprod_info['fast_spec_dir']:
                file_dict['fast_spec'] = file_path
            case path if path == specprod_info['qso_maker_dir']:
                file_dict['qso_maker'] = file_path
            case path if path == specprod_info['zcat_dir']:
                file_dict['zcat'] = file_path
            case _:
                raise ValueError(f'Unknown path: {file_path}')

    return file_dict


def read_fastspecfit(specprod_info: dict[str, Path | list[str]]) -> Table:
    """Reads and merges the FastSpecFit catalog extensions into a single table.

    Args:
        specprod_info:
            Dictionary with information about the targeted data release. Must include path to FastSpecFit catalog and
            lists of column names for each extension we wish to read in.

    Returns:
        Merged table of the two extensions.
    """

    # Read in the two extensions and cast as a table.
    fastspec_data_catalog = Table(fitsio.read(str(specprod_info['fast_spec']),
                                              columns=specprod_info['fast_spec_data_cols'], ext='FASTSPEC'))
    fastspec_meta_catalog = Table(fitsio.read(str(specprod_info['fast_spec']),
                                              columns=specprod_info['fast_spec_meta_cols'], ext='METADATA'))

    try:
        # Only DR2/Loa will have this extension. At present, we only need the LOGMSTAR from it.
        fastspec_specphot_catalog = Table(fitsio.read(str(specprod_info['fast_spec']),
                                                      columns=specprod_info['fast_spec_specphot_cols'], ext='SPECPHOT'))
    except KeyError:
        # For non-DR2 catalogs, we'll just assign this catalog to an empty Table as it will pass through the hstack
        # without issue and minimizes special-case handling.
        fastspec_specphot_catalog = Table(data=None)

    # Remove any common columns between the extensions.
    fastspec_meta_catalog.remove_columns(set(specprod_info['fast_spec_data_cols'])
                                         .intersection(specprod_info['fast_spec_meta_cols']))

    try:
        fastspec_specphot_catalog.remove_columns(set(specprod_info['fast_spec_data_cols'])
                                                 .intersection(specprod_info['fast_spec_specphot_cols']))
    except KeyError:
        # For non-DR2 catalogs, we will just pass this error as it has no effect on the stand-in empty catalog merging.
        pass

    # As all the extensions are already row-aligned we can do a fast hstack operation rather than a full join.
    fastspec_catalog = hstack([fastspec_data_catalog, fastspec_meta_catalog, fastspec_specphot_catalog])

    return fastspec_catalog


def read_input_catalogs(specprod_info: dict[str, Path | list[str]], qsom_colnames: list[str],
                        redshift_colnames: list[str]) -> Table:
    """Reads in the input catalogs and merges them into a single table to be used for AGN/Galaxy classification.

    Args:
        specprod_info:
            Dictionary with information about the targeted data release. Must include path names to relevant catalogs
            and associated data-release specific column names.
        qsom_colnames:
            List of universal column names to be read in for the QSO-Maker catalog.
        redshift_colnames:
            List of universal column names to be read in for the redshift catalog. These will be combined with the
            data-release specific column names.

    Returns:
        Joined table of the three input catalogs.

    Raises:
        ValueError: Under any of the following conditions:

            - If the merged FastSpecFit + QSO-Maker catalog contains objects with redshifts :math:`z < 0.001`.
            - If the merged FastSpecFit + QSO-Maker catalog contains objects with zero coadd exposure time.
            - If the merged FastSpecFit + QSO-Maker + Redshift catalog contains non-"TGT" object types.

        KeyError: When running the standard dispatcher ``specprod_info`` on DR2 (Loa) entries without building the
            DR2-specific dispatcher.
        OSError: On failure to open an input catalog file.

    """

    try:
        # Read in and merge the FastSpecFit catalog extensions into a combined table
        fastspec_catalog = read_fastspecfit(specprod_info)

        # Read in the QSO-Maker catalog
        qso_maker_catalog = Table(fitsio.read(str(specprod_info['qso_maker']), ext=1, columns=qsom_colnames))

        # Read in the Redshift catalog (columns used will be the data-release specific columns and global columns)
        redshift_catalog = Table(fitsio.read(str(specprod_info['zcat']), ext=1,
                                             columns=redshift_colnames + specprod_info['zcat_cols']))
    except KeyError as e:
        raise KeyError('Error when trying to read in an input catalog. '
                       'If you are trying to read DR2 (Loa) catalogs, '
                       'function `generate_loa_dispatchers` must be ran first.') from e
    except OSError as e:
        raise OSError('Error on reading an input catalog.') from e

    # Main identifiers for Joins
    keys_for_join = ['TARGETID', 'SURVEY', 'PROGRAM']

    # Join FastSpecFit with QSO-Maker
    desi_catalog = join(fastspec_catalog, qso_maker_catalog, keys=keys_for_join, join_type='left')

    # Test for consistency
    try:
        assert all(desi_catalog['Z'] > 0.001)
    except AssertionError as e:
        raise ValueError('Joined FastSpecFit + QSO-Maker catalog contains objects z < 0.001') from e

    try:
        assert all(desi_catalog['COADD_EXPTIME'] > 0.0)
    except AssertionError as e:
        raise ValueError('Joined FastSpecFit + QSO-Maker catalog contains objects with zero coadd exposure time') from e

    # Join the FastSpecFit+QSO-Maker catalog with the redshift catalog
    desi_catalog = join(desi_catalog, redshift_catalog, keys=keys_for_join, join_type='left')

    # Test for consistency
    try:
        assert all(desi_catalog['OBJTYPE'] == 'TGT')
    except AssertionError as e:
        raise ValueError('Joined FastSpecFit + QSO-Maker + Redshift catalog contains non-"TGT" object types') from e

    return desi_catalog


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


def output_processing(input_table: Table, output_filename: str | Path,
                      ext1_colnames: list[str], ext2_colnames: list[str],
                      ext1_units: dict[str, str], ext2_units: dict[str, str]) -> None:
    """Processes catalog for final write out.

    Args:
        input_table:
            Table with AGN/Galaxy classification bit masks present.
        output_filename:
            Path to the output FITS file.
        ext1_colnames:
            List of column names in ``input_table`` to place in extension 1 of output FITS file.
        ext2_colnames:
            List of column names in ``input_table`` to place in extension 2 of output FITS file.
        ext1_units:
            Dictionary of units to apply on columns in extension 1 of output FITS file.
        ext2_units:
            Dictionary of units to apply on columns in extension 2 of output FITS file.

    """

    # Create the FITS HDU list structure and write out file
    primary_hdu = fits.PrimaryHDU()
    agn_gal_table_hdu = fits.BinTableHDU(input_table[ext1_colnames], name='AGNGALCAT')
    flux_table_hdu = fits.BinTableHDU(input_table[ext2_colnames], name='AUXDATA')
    hdu_list = fits.HDUList([primary_hdu, agn_gal_table_hdu, flux_table_hdu])
    hdu_list.writeto(output_filename, overwrite=True, checksum=True)

    # We will use the ``annotate_fits`` function to add units to the extensions.
    annotate_fits(output_filename, extension=1, output=output_filename, units=ext1_units, overwrite=True)
    annotate_fits(output_filename, extension=2, output=output_filename, units=ext2_units, overwrite=True)


def build_agngal_catalog(data_release: dict[str, Path | list[str]], output_filename: str | Path) -> None:
    """Builds the DESI AGN/Galaxy Classification VAC.

    Args:
        data_release:
            Dispatcher including all relevant input files and column names associated with the data release VAC is being
            built against.
        output_filename:
            Path to output FITS file.

    """

    # Read in unit definitions from file
    out_ext1_units, _ = load_yml_units(output_ext1_unit_defs)
    out_ext2_units, _ = load_yml_units(output_ext2_unit_defs)

    # Build the initial input catalog
    desi_table = read_input_catalogs(specprod_info=data_release, qsom_colnames=qso_maker_cols,
                                     redshift_colnames=zcat_cols)

    # Apply all AGN/Galaxy classifications and build BitMask columns
    desi_table = apply_agngal_class(desi_table, agnmask_defs=agn_bitmask_defs)

    # Write out file to disk
    output_processing(desi_table, output_filename,
                      ext1_colnames=data_release['output_cols_ext1'], ext2_colnames=output_cols_ext2,
                      ext1_units=out_ext1_units, ext2_units=out_ext2_units)


if __name__ == "__main__":
    # Provide CLI arguments for easy execution via SLURM scripts.
    parser = ArgumentParser()
    parser.add_argument("data_release", choices=['edr', 'dr1', 'dr2', 'fuji', 'iron', 'loa', 'testing'],
                        help='Data release to build catalog from.')
    parser.add_argument("-o", "--output", default="desi_agngal.fits", required=True,
                        help="Path to output FITS file.", type=Path)
    args = parser.parse_args()

    if args.data_release == 'edr' or args.data_release == 'fuji':
        spec_prod = 'fuji'
    elif args.data_release == 'dr1' or args.data_release == 'iron':
        spec_prod = 'iron'
    elif args.data_release == 'dr2' or args.data_release == 'loa':
        spec_prod = 'loa'
    elif args.data_release == 'testing':
        spec_prod = 'testing'
    else:
        raise ValueError(f"Invalid data release: {args.data_release}")

    # Due to size and complexity, DR2/Loa needs to be handled by parallel processing compared to previous DRs.
    if spec_prod == 'loa':
        # Using the directory paths listed above, build dispatchers for all survey-program-(optionally healpix)
        # sub-catalogs.
        dr_dispatcher = generate_loa_dispatchers(desi_specprod['loa'])

        # We need to assign unique output filenames for Loa catalogs based on the input catalog names.
        output_filenames = [args.output / Path(f'desi_agngal_loa_{survey_program}.fits')
                            for survey_program in dr_dispatcher.keys()]

        # Run all catalog operations in parallel simultaneously
        with Pool() as pool:
            pool.starmap_async(build_agngal_catalog, zip(dr_dispatcher.values(), output_filenames))

    elif spec_prod == 'testing':
        dr_dispatcher = generate_loa_dispatchers(desi_specprod['loa'])
        cmx_other_dispatcher = dr_dispatcher['cmx-other']

        build_agngal_catalog(cmx_other_dispatcher, args.output)

    else:
        # For all previous data releases (EDR/Fuji, DR1/Iron) we will run the operations in serial.
        dr_dispatcher = desi_specprod[spec_prod]
        build_agngal_catalog(data_release=dr_dispatcher, output_filename=args.output)
