"""
generate_agngal_config.py
Author: Benjamin Floyd

This script generates the YAML configuration files used to generate the DESI AGN/Galaxy Classification VACs for all
major data releases/specprods.

These files will contain all necessary paths to input files and the required column definitions for each data release.
"""
import os
import re
from itertools import groupby
from pathlib import Path

import yaml

DESI_ROOT_RO = os.getenv('DESI_ROOT_READONLY', None)


def generate_fuji_iron_config(specprod_info: dict[str, str | list[str]], universal_info: dict[str, str | list[str]],
                              config_file_name: Path, specprod_name: str):
    """Processes the EDR and DR1 configuration information into a YAML file.

    As these data releases are simple, we only need to combine the data-release specific information with the universal
    information and output the dictionary to the configuration file.

    Args:
        specprod_info:
            Dictionary containing paths to find the relevant catalogs for EDR or DR1. Additionally, any
            associated data-release specific column names that should be included in the final configuration file.
        universal_info:
            Dictionary containing paths to find the relevant files or input and output catalog column names that are
            universal to all data releases.
        config_file_name:
            Path to output configuration file containing all relevant information to build the targeted data release
            VAC.
        specprod_name:
            Name to assign the top-level label of the dictionary/yaml file. Should be ``'fuji'`` or ``'iron'``.
    """

    # Merge the column lists between the data release-specific and universal lists.
    merged_column_lists = {list_name: [*specprod_info[list_name], *universal_info[list_name]]
                           for list_name in set(specprod_info.keys()).intersection(universal_info.keys())}


    # Simply union the two dictionaries together to create the complete data-release configuration.
    data_release_info = specprod_info | universal_info | merged_column_lists

    # In order to preserve consistency with the Loa data release configuration we need to nest our dictionary.
    data_release_info = {f'{specprod_name}': {f'{specprod_name}_all': data_release_info}}

    # Write the configuration to file
    with open(config_file_name, 'w') as config_file:
        yaml.safe_dump(data_release_info, config_file)


def generate_loa_config(specprod_info: dict[str, Path | list[str]], universal_info: dict[str, str | list[str]],
                        config_file_name: Path, specprod_name: str):
    """Processes the DR2 suite of catalogs into dispatchers matching the scheme for EDR and DR1.

    Args:
        specprod_info:
            Dictionary containing paths to directories to find the relevant catalogs for DR2. Additionally, any
            associated data-release specific column names that should be included in the final dispatcher.
        universal_info:
            Dictionary containing paths to find the relevant files or input and output catalog column names that are
            universal to all data releases.
        config_file_name:
            Path to output configuration file containing all relevant information to build the targeted data release
            VAC.
        specprod_name:
            Name to assign the top-level label of the dictionary/yaml file. Should be ``'loa'``.

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
    loa_info = {survey_program: loa_paths(catalog_paths, specprod_info)
                for survey_program, catalog_paths in all_catalogs_dict.items()}

    # Extract all data for all the column data common to all survey-program(-healpix) subcatalog of Loa
    all_loa_info = {label: info for label, info in specprod_info.items() if 'cols' in label}

    # Merge the column lists between the data release-specific and universal lists.
    merged_column_lists = {list_name: [*all_loa_info[list_name], *universal_info[list_name]]
                           for list_name in set(all_loa_info.keys()).intersection(universal_info.keys())}

    # To each survey-program(-healpix) subcatalog of Loa, union the universal info dictionary.
    loa_info = {f'{specprod_name}': {survey_program: survey_program_info | all_loa_info | universal_info | merged_column_lists
                 for survey_program, survey_program_info in loa_info.items()}}

    # Write the configuration to file
    with open(config_file_name, 'w') as config_file:
        yaml.safe_dump(loa_info, config_file)


def loa_paths(file_paths: list[Path], specprod_info: dict[str, Path | list[str]]) -> dict[str, str]:
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
                file_dict['fast_spec'] = str(file_path)
            case path if path == specprod_info['qso_maker_dir']:
                file_dict['qso_maker'] = str(file_path)
            case path if path == specprod_info['zcat_dir']:
                file_dict['zcat'] = str(file_path)
            case _:
                raise ValueError(f'Unknown path: {file_path}')

    return file_dict


# First we want to build a dispatch pattern to handle the various file selections between data releases
# EDR
fuji_info = {
    # QSO-Maker catalog from Edmonds catalog keeping all columns
    'qso_maker': f'{DESI_ROOT_RO}/users/edmondc/QSO_catalog/fuji/QSO_cat_fuji_healpix_all_targets_v2.fits',

    # FastSpecFit catalog
    'fast_spec': f'{DESI_ROOT_RO}/spectro/fastspecfit/fuji/v3.2/catalogs/fastspec-fuji.fits',
    'fast_spec_data_cols': ['LOGMSTAR'],

    # Redshift catalog
    'zcat': f'{DESI_ROOT_RO}/public/edr/vac/edr/zcat/fuji/v1.0/zall-pix-edr-vac.fits',
    'zcat_cols': ['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX', 'ZERR', 'TSNR2_LRG', 'SV_NSPEC', 'SV_PRIMARY',
                  'ZCAT_NSPEC', 'ZCAT_PRIMARY', 'MIN_MJD', 'MEAN_MJD', 'MAX_MJD', 'OBJTYPE'],

    # Output catalog extension 1 column names
    'output_cols_ext1': ['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX',
                         'Z', 'ZWARN', 'Z_RR', 'ZERR_RR', 'Z_QSOM', 'ZERR_QSOM', 'Z_QN', 'QN_C_LINE_BEST',
                         'SPECTYPE',
                         'AGN_MASKBITS', 'OPT_UV_TYPE', 'IR_TYPE',
                         'COADD_FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC', 'LS_ID',
                         'MIN_MJD', 'MEAN_MJD', 'MAX_MJD', 'COADD_NUMEXP', 'COADD_EXPTIME',
                         'SV_PRIMARY', 'ZCAT_PRIMARY',
                         'DESI_TARGET', 'SCND_TARGET', 'BGS_TARGET', 'CMX_TARGET',
                         'SV1_DESI_TARGET', 'SV2_DESI_TARGET', 'SV3_DESI_TARGET',
                         'SV1_BGS_TARGET', 'SV2_BGS_TARGET', 'SV3_BGS_TARGET',
                         'SV1_SCND_TARGET', 'SV2_SCND_TARGET', 'SV3_SCND_TARGET']
}

# DR1
iron_info = {
    # QSO-Maker catalog from `merge_QSOmaker.ipynb`. DR1 version from after Edmond ran on all targets/all surveys
    'qso_maker': f'{DESI_ROOT_RO}/science/gqp/agncatalog/qsomaker/iron/QSO_cat_iron_healpix_all_targets_v1.fits',
    'qso_maker_cols': ['QN_C_LINE_BEST'],

    # FastSpecFit catalog
    'fast_spec': f'{DESI_ROOT_RO}/spectro/fastspecfit/iron/v2.1/catalogs/fastspec-iron.fits',
    'fast_spec_data_cols': ['LOGMSTAR'],

    # Redshift catalog
    'zcat': f'{DESI_ROOT_RO}/spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits',
    'zcat_cols': ['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX', 'ZERR', 'TSNR2_LRG', 'ZCAT_NSPEC',
                  'ZCAT_PRIMARY', 'SV_NSPEC', 'SV_PRIMARY', 'MAIN_PRIMARY', 'MAIN_NSPEC', 'MIN_MJD', 'MEAN_MJD',
                  'MAX_MJD', 'OBJTYPE'],

    # Output catalog extension 1 column names
    'output_cols_ext1': ['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX',
                         'Z', 'ZWARN', 'Z_RR', 'ZERR_RR', 'Z_QSOM', 'ZERR_QSOM', 'Z_QN', 'QN_C_LINE_BEST',
                         'SPECTYPE',
                         'AGN_MASKBITS', 'OPT_UV_TYPE', 'IR_TYPE',
                         'COADD_FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC', 'LS_ID',
                         'MIN_MJD', 'MEAN_MJD', 'MAX_MJD', 'COADD_NUMEXP', 'COADD_EXPTIME',
                         'SV_PRIMARY', 'MAIN_PRIMARY', 'ZCAT_PRIMARY',
                         'DESI_TARGET', 'SCND_TARGET', 'BGS_TARGET', 'CMX_TARGET',
                         'SV1_DESI_TARGET', 'SV2_DESI_TARGET', 'SV3_DESI_TARGET',
                         'SV1_BGS_TARGET', 'SV2_BGS_TARGET', 'SV3_BGS_TARGET',
                         'SV1_SCND_TARGET', 'SV2_SCND_TARGET', 'SV3_SCND_TARGET']
}

# DR2
loa_base_info = {
    # QSO-Maker
    'qso_maker_dir': Path(f'{DESI_ROOT_RO}/science/gqp/agncatalog/qsomaker/loa'),

    # FastSpecFit Catalog
    'fast_spec_dir': Path(f'{DESI_ROOT_RO}/vac/dr2/fastspecfit/loa/v1.0/catalogs'),

    'fast_spec_specphot_cols': ['TARGETID', 'PROGRAM', 'SURVEY', 'LOGMSTAR'],

    # Redshift Catalog
    'zcat_dir': Path(f'{DESI_ROOT_RO}/science/gqp/agncatalog/zpix_nside1/loa/v1'),
    'zcat_cols': ['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX', 'ZERR', 'TSNR2_LRG', 'ZCAT_NSPEC',
                  'ZCAT_PRIMARY', 'SV_NSPEC', 'SV_PRIMARY', 'MAIN_PRIMARY', 'MAIN_NSPEC', 'MIN_MJD', 'MEAN_MJD',
                  'MAX_MJD', 'OBJTYPE'],

    # Output catalog extension 1 column names
    'output_cols_ext1': ['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX',
                         'Z', 'ZWARN', 'Z_RR', 'ZERR_RR', 'Z_QSOM', 'ZERR_QSOM', 'Z_QN', 'QN_C_LINE_BEST',
                         'SPECTYPE',
                         'AGN_MASKBITS', 'OPT_UV_TYPE', 'IR_TYPE',
                         'COADD_FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC', 'LS_ID',
                         'MIN_MJD', 'MEAN_MJD', 'MAX_MJD', 'COADD_NUMEXP', 'COADD_EXPTIME',
                         'SV_PRIMARY', 'MAIN_PRIMARY', 'ZCAT_PRIMARY',
                         'DESI_TARGET', 'SCND_TARGET', 'BGS_TARGET', 'CMX_TARGET',
                         'SV1_DESI_TARGET', 'SV2_DESI_TARGET', 'SV3_DESI_TARGET',
                         'SV1_BGS_TARGET', 'SV2_BGS_TARGET', 'SV3_BGS_TARGET',
                         'SV1_SCND_TARGET', 'SV2_SCND_TARGET', 'SV3_SCND_TARGET']
}

all_data_release_info = {
    # AGN BitMask Definitions file
    'agn_bitmask_defs': '/dvs_ro/u2/b/bfloyd/agngal_dr2/AgnCats/py/agnmask.yaml',

    # Output file unit definitions files
    'output_ext1_unit_defs': '/dvs_ro/u2/b/bfloyd/agngal_dr2/AgnCats/py/configs/ext1_units.yaml',
    'output_ext2_unit_defs': '/dvs_ro/u2/b/bfloyd/agngal_dr2/AgnCats/py/configs/ext2_units.yaml',

    # Universal input catalog column names
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
                            'OIII_5007_FLUX', 'OIII_5007_FLUX_IVAR', 'OIII_5007_SIGMA', 'OIII_5007_EW', 'OIII_5007_EW_IVAR',
                            'OI_6300_FLUX', 'OI_6300_FLUX_IVAR',
                            'HALPHA_FLUX', 'HALPHA_FLUX_IVAR', 'HALPHA_EW', 'HALPHA_EW_IVAR',
                            'HALPHA_BROAD_FLUX', 'HALPHA_BROAD_FLUX_IVAR', 'HALPHA_BROAD_SIGMA',
                            'HALPHA_BROAD_VSHIFT',
                            'NII_6584_FLUX', 'NII_6584_FLUX_IVAR', 'NII_6584_EW', 'NII_6584_EW_IVAR',
                            'SII_6716_FLUX', 'SII_6716_FLUX_IVAR',
                            'SII_6731_FLUX', 'SII_6731_FLUX_IVAR'],

    'fast_spec_meta_cols': ['TARGETID', 'PROGRAM', 'SURVEY', 'LS_ID',
                            'PHOTSYS', 'SPECTYPE', 'DELTACHI2',
                            'Z', 'ZWARN', 'Z_RR',
                            'FLUX_W1', 'FLUX_W2', 'FLUX_W3',
                            'FLUX_IVAR_W1', 'FLUX_IVAR_W2', 'FLUX_IVAR_W3',
                            'EBV',
                            'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2', 'MW_TRANSMISSION_W3'],

    'qso_maker_cols': ['TARGETID', 'Z', 'ZERR', 'SPECTYPE', 'COADD_FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC',
                       'MORPHTYPE', 'MASKBITS', 'COADD_NUMEXP', 'COADD_EXPTIME', 'TSNR2_LYA', 'TSNR2_QSO',
                       'Z_QN', 'C_LYA', 'C_CIV', 'C_CIII', 'C_MgII', 'C_Hbeta', 'C_Halpha',
                       'QSO_MASKBITS', 'SURVEY', 'PROGRAM'],

    'zcat_cols': ['DESI_TARGET', 'BGS_TARGET', 'SCND_TARGET', 'CMX_TARGET',
                  'SV1_DESI_TARGET', 'SV1_BGS_TARGET', 'SV1_SCND_TARGET',
                  'SV2_DESI_TARGET', 'SV2_BGS_TARGET', 'SV2_SCND_TARGET',
                  'SV3_DESI_TARGET', 'SV3_BGS_TARGET', 'SV3_SCND_TARGET'],

    # Universal output catalog column names
    'output_cols_ext2': ['TARGETID', 'SURVEY', 'PROGRAM', 'LOGMSTAR',
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
                         'OIII_5007_FLUX', 'OIII_5007_FLUX_IVAR', 'OIII_5007_SIGMA', 'OIII_5007_EW', 'OIII_5007_EW_IVAR',
                         'OI_6300_FLUX', 'OI_6300_FLUX_IVAR',
                         'HALPHA_EW', 'HALPHA_EW_IVAR', 'HALPHA_FLUX', 'HALPHA_FLUX_IVAR',
                         'HALPHA_BROAD_FLUX', 'HALPHA_BROAD_FLUX_IVAR', 'HALPHA_BROAD_VSHIFT', 'HALPHA_BROAD_SIGMA',
                         'NII_6584_FLUX', 'NII_6584_FLUX_IVAR', 'NII_6584_EW', 'NII_6584_EW_IVAR',
                         'SII_6716_FLUX', 'SII_6716_FLUX_IVAR',
                         'SII_6731_FLUX', 'SII_6731_FLUX_IVAR']
}

if __name__ == '__main__':
    generate_fuji_iron_config(specprod_info=fuji_info, universal_info=all_data_release_info,
                              config_file_name=Path('/global/u2/b/bfloyd/agngal_dr2/AgnCats/py/configs/fuji_config.yaml'),
                              specprod_name='fuji')
    generate_fuji_iron_config(specprod_info=iron_info, universal_info=all_data_release_info,
                              config_file_name=Path('/global/u2/b/bfloyd/agngal_dr2/AgnCats/py/configs/iron_config.yaml'),
                              specprod_name='iron')
    generate_loa_config(specprod_info=loa_base_info, universal_info=all_data_release_info,
                        config_file_name=Path('/global/u2/b/bfloyd/agngal_dr2/AgnCats/py/configs/loa_config.yaml'),
                        specprod_name='loa')
