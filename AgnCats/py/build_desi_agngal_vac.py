"""
build_desi_agngal_vac.py
Author: Benjamin Floyd

This top-level script builds the DESI AGN/Galaxy Classification VAC. This supersedes the DR1 00_AGNQSO_summary_cat.ipynb
notebook and provides parallelized computation abilities in constructing the final catalog.
"""

import sys

import numpy as np
import yaml

sys.path.append('/global/homes/b/bfloyd/agngal_dr2')

import argparse
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass

import fitsio
from astropy.io import fits
from astropy.table import Table, hstack, join
from desiutil.annotate import annotate_fits, load_yml_units
from desiutil.log import get_logger

from AgnCats.py import set_agn_masksDESI as agn_masks

logger = get_logger()

@dataclass(kw_only=True)
class SpecProdInfo:
    """Data class for DESI specprod configuration information."""
    agn_bitmask_defs: str
    fast_spec: str
    fast_spec_data_cols: list[str]
    fast_spec_meta_cols: list[str]
    output_cols_ext1: list[str]
    output_cols_ext2: list[str]
    output_ext1_unit_defs: str
    output_ext2_unit_defs: str
    qso_maker: str
    qso_maker_cols: list[str]
    zcat: str
    zcat_cols: list[str]
    fast_spec_specphot_cols: list[str] = None


def read_config(config_path: Path | str) -> dict[str, dict[str, SpecProdInfo]]:
    """Read in the configuration file and parse it into a dictionary of data classes containing all configuration
    needed to create the catalog.

    Args:
        config_path:
            Path to the configuration file.
    Returns:
        Dictionary with configuration information with key named after the output catalog or sub-catalog.
    """

    with open(config_path, 'r') as f:
        config_info = yaml.safe_load(f)

    try:
        # Cast the nested dictionary in the configuration info as a SpecProdInfo data class to help with type checking.
        config_info = {survey_name: {survey_program: SpecProdInfo(**config)
                                     for survey_program, config in survey_config.items()}
                       for survey_name, survey_config in config_info.items()}
    except TypeError as e:
        # Casting into a data class also allows us to check all required data is present.
        raise KeyError(f'Required configuration keys were not found in the YAML file.') from e

    return config_info


def read_fastspecfit(specprod_info: SpecProdInfo) -> Table:
    """Reads and merges the FastSpecFit catalog extensions into a single table.

    Args:
        specprod_info:
            Data class with information about the targeted data release. Must include path to FastSpecFit catalog and
            lists of column names for each extension we wish to read in.

    Returns:
        Merged table of the two extensions.
    """

    # Read in the two extensions and cast as a table.
    fastspec_data_catalog = Table(fitsio.read(specprod_info.fast_spec,
                                              columns=specprod_info.fast_spec_data_cols, ext='FASTSPEC'))
    fastspec_meta_catalog = Table(fitsio.read(specprod_info.fast_spec,
                                              columns=specprod_info.fast_spec_meta_cols, ext='METADATA'))

    if specprod_info.fast_spec_specphot_cols is not None:
        # Only DR2/Loa will have this extension. At present, we only need the LOGMSTAR from it.
        fastspec_specphot_catalog = Table(fitsio.read(specprod_info.fast_spec,
                                                      columns=specprod_info.fast_spec_specphot_cols, ext='SPECPHOT'))
    else:
        # For non-DR2 catalogs, we'll just assign this catalog to an empty Table as it will pass through the hstack
        # without issue and minimizes special-case handling.
        fastspec_specphot_catalog = Table(data=None)

    # Remove any common columns between the extensions.
    fastspec_meta_catalog.remove_columns(set(specprod_info.fast_spec_data_cols)
                                         .intersection(specprod_info.fast_spec_meta_cols))

    if specprod_info.fast_spec_specphot_cols is not None:
        fastspec_specphot_catalog.remove_columns(set(specprod_info.fast_spec_data_cols)
                                                 .intersection(specprod_info.fast_spec_specphot_cols))

    # As all the extensions are already row-aligned we can do a fast hstack operation rather than a full join.
    fastspec_catalog = hstack([fastspec_data_catalog, fastspec_meta_catalog, fastspec_specphot_catalog])

    return fastspec_catalog


def read_input_catalogs(specprod_info: SpecProdInfo) -> Table:
    """Reads in the input catalogs and merges them into a single table to be used for AGN/Galaxy classification.

    Args:
        specprod_info:
            Data class with information about the targeted data release. Must include path names to relevant catalogs
            and associated data-release specific column names.

    Returns:
        Joined table of the three input catalogs.

    Raises:
        ValueError: Under any of the following conditions:

            - If the merged FastSpecFit + QSO-Maker catalog contains objects with redshifts :math:`z < 0.001`.
            - If the merged FastSpecFit + QSO-Maker catalog contains objects with zero coadd exposure time.
            - If the merged FastSpecFit + QSO-Maker + Redshift catalog contains non-"TGT" object types.

        OSError: On failure to open an input catalog file.

    """

    try:
        # Read in and merge the FastSpecFit catalog extensions into a combined table
        fastspec_catalog = read_fastspecfit(specprod_info)

        # Read in the QSO-Maker catalog
        qso_maker_catalog = Table(fitsio.read(specprod_info.qso_maker, ext=1, columns=specprod_info.qso_maker_cols))

        # Read in the Redshift catalog (columns used will be the data-release specific columns and global columns)
        redshift_catalog = Table(fitsio.read(specprod_info.zcat, ext=1, columns=specprod_info.zcat_cols))
    except OSError as e:
        raise OSError('Error on reading an input catalog.') from e

    # Add QN_C_LINE_BEST to QSO-Maker catalog if not processing Iron (pre-computed)
    if 'QN_C_LINE_BEST' not in specprod_info.qso_maker_cols:
        qn_c_lines = ['C_LYA', 'C_CIV', 'C_CIII', 'C_MgII', 'C_Hbeta', 'C_Halpha']
        all_c_lines = np.vstack([qso_maker_catalog[c_line_col] for c_line_col in qn_c_lines]).T
        qso_maker_catalog['QN_C_LINE_BEST'] = np.nanmax(all_c_lines, axis=1)

    # We want to preserve the redshift columns from QSO-Maker separately from FastSpecFit's columns
    qso_maker_catalog.rename_columns(['Z', 'ZERR', 'SPECTYPE', 'MORPHTYPE'],
                                     ['Z_QSOM', 'ZERR_QSOM', 'SPECTYPE_QSOM', 'MORPHTYPE_QSOM'])

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


def apply_agngal_class(input_table: Table, agnmask_defs: Path | str) -> Table:
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


def output_processing(input_table: Table, output_filename: str, specprod_info: SpecProdInfo) -> None:
    """Processes catalog for final write out.

    Args:
        input_table:
            Table with AGN/Galaxy classification bit masks present.
        output_filename:
            Path to the output FITS file.
        specprod_info:
            Data class with all information needed for targeted data release. Must include the output column lists and
            the paths to the unit definitions.

    """

    # Create the FITS HDU list structure and write out file
    primary_hdu = fits.PrimaryHDU()
    agn_gal_table_hdu = fits.BinTableHDU(input_table[specprod_info.output_cols_ext1], name='AGNCAT')
    flux_table_hdu = fits.BinTableHDU(input_table[specprod_info.output_cols_ext2], name='AUXDATA')
    hdu_list = fits.HDUList([primary_hdu, agn_gal_table_hdu, flux_table_hdu])
    hdu_list.writeto(output_filename, overwrite=True, checksum=True)

    # Read in unit definitions from file
    out_ext1_units, _ = load_yml_units(specprod_info.output_ext1_unit_defs)
    out_ext2_units, _ = load_yml_units(specprod_info.output_ext2_unit_defs)

    # We will use the ``annotate_fits`` function to add units to the extensions.
    annotate_fits(output_filename, extension=1, output=output_filename, units=out_ext1_units, validate=False, overwrite=True)
    annotate_fits(output_filename, extension=2, output=output_filename, units=out_ext2_units, validate=False, overwrite=True)


def build_agngal_catalog(specprod_info: SpecProdInfo, output_filename: str) -> None:
    """Builds the DESI AGN/Galaxy Classification VAC.

    Args:
        specprod_info:
            Data class containing all necessary configuration information to build the data release catalog.
        output_filename:
            Path to output FITS file.

    """

    # Build the initial input catalog
    desi_table = read_input_catalogs(specprod_info=specprod_info)

    # Apply all AGN/Galaxy classifications and build BitMask columns
    desi_table = apply_agngal_class(input_table=desi_table, agnmask_defs=specprod_info.agn_bitmask_defs)

    # Write out file to disk
    output_processing(input_table=desi_table, output_filename=output_filename, specprod_info=specprod_info)


if __name__ == "__main__":
    # Set multiprocessing spawn method per NERSC recommendation.
    mp.set_start_method('spawn')

    # Provide CLI arguments for easy execution via SLURM scripts.
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to configuration file.', type=Path)
    parser.add_argument("-o", "--output", default="desi_agngal.fits", required=True,
                        help="Path to output FITS file.", type=Path)
    parser.add_argument('--testing', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--testing-pp', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Read in the configuration file
    spec_prod_info = read_config(args.config)

    if 'fuji' in spec_prod_info.keys():
        spec_prod = 'fuji'
    elif 'iron' in spec_prod_info.keys():
        spec_prod = 'iron'
    elif 'loa' in spec_prod_info.keys():
        spec_prod = 'loa'
    else:
        raise ValueError(f'Invalid data release in configuration: {list(spec_prod_info.keys())}')

    # Due to size and complexity, DR2/Loa needs to be handled by parallel processing compared to previous DRs.
    if args.testing:
        cmx_other_info = spec_prod_info['loa']['cmx-other']
        build_agngal_catalog(cmx_other_info, args.output)

    elif args.testing_pp:
        # To test parallel processing we will regenerate a Loa version of EDR.
        edr_catalog_set = ['cmx-other',
                           'special-backup', 'special-bright', 'special-dark',
                           'sv1-backup', 'sv1-bright', 'sv1-dark', 'sv1-other',
                           'sv2-backup', 'sv2-bright', 'sv2-dark',
                           'sv3-backup', 'sv3-bright', 'sv3-dark']
        testing_info_set = {'loa': {survey_program: config_info
                                    for survey_program, config_info in spec_prod_info['loa'].items()
                                    if survey_program in edr_catalog_set}}
        output_filenames = [str(args.output / Path(f'desi_agngal_loa_{survey_program}.fits'))
                            for survey_program in testing_info_set['loa'].keys()]

        with mp.Pool() as pool:
            result = pool.starmap_async(build_agngal_catalog, zip(spec_prod_info['loa'].values(), output_filenames))
            result.get()

    elif spec_prod == 'loa':
        # We need to assign unique output filenames for Loa catalogs based on the input catalog names.
        output_filenames = [str(args.output / Path(f'desi_agngal_loa_{survey_program}.fits'))
                            for survey_program in spec_prod_info['loa'].keys()]

        # Run all catalog operations in parallel simultaneously
        with mp.Pool() as pool:
            result = pool.starmap_async(build_agngal_catalog, zip(spec_prod_info.values(), output_filenames))
            result.get()

    else:
        # For all previous data releases (EDR/Fuji, DR1/Iron) we will run the operations in serial.
        spec_prod_info = spec_prod_info[spec_prod][f'{spec_prod}_all']
        build_agngal_catalog(specprod_info=spec_prod_info, output_filename=str(args.output))
