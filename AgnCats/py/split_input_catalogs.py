"""
split_input_catalogs.py
Author: Benjamin Floyd

This script generalizes the split_zpix_Loa.ipynb and split_QSOmaker_Loa.ipynb notebooks to split the Redshift and
QSO-Maker catalogs respectively for the DR2/Loa DESI specprod to work on all DESI data releases.

This has become necessary to match the organization of the most recent FastSpec and FastPhot catalogs.
"""

import numpy as np
from astropy.table import Table
import fitsio
from pathlib import Path
from desiutil.healpix import radec2hpix
from astropy.io import fits

def split_by_healpix(catalog: Table, nside: int) -> Table:
    """Takes a catalog that is already grouped by Survey-Program and further groups by HEALPix of size ``nside``.

    Args:
        catalog:
            Table that is ideally already grouped by ``SURVEY`` and ``PROGRAM`` that we wish to further split by
            HEALPix of a specified size.
        nside:
            Size of HEALPix we want to group our catalog by.

    Returns:
        Grouped Table by ``SURVEY``, ``PROGRAM``, and ``new_healpix`` of size ``nside``.
    """

    # Determine the new nside HEALPix number based on the RA and Dec of the object
    catalog['new_healpix'] = radec2hpix(nside=nside, ra=catalog['TARGET_RA'], dec=catalog['TARGET_DEC'])

    # Group by our new HEALPix numbers
    catalog_grp = catalog.group_by(['SURVEY', 'PROGRAM', 'new_healpix'])

    return catalog_grp

def split_catalogs(catalog_file: str, out_prefix: str, out_dir: str, specprod: str, version: int | float) -> None:
    """Splits an input monolithic catalog into sub-catalogs split by Survey-Program(-nside1_hpXX).

    Most sub-catalogs will just be split by Survey-Program with the exception of ``Main-Bright`` and ``Main-Dark``.
    These sub-catalogs are often very large, and thus we further split by HEALPix of nside=1.
    This is largely to match the subdivision adopted by FastSpec and FastPhot catalogs.

    Args:
        catalog_file:
            Path to monolithic QSO-Maker catalog to split.
        out_prefix:
            Prefix to place in output file name. E.g., "QSO" or "zpix" for QSO-Maker or Redshift catalogs respectively.
        out_dir:
            Path to directory to save split catalogs to.
        specprod:
            DESI internal codename for data release.
        version:
            Version number to append to output catalogs.
    """

    # Read in the monolithic catalog
    catalog = Table(fitsio.read(catalog_file, ext=1))

    # Group the catalog by Survey-Program
    catalog = catalog.group_by(['SURVEY', 'PROGRAM'])

    # First, we will split the catalog by survey-program-nside1-hpXX
    subcats = {}
    for sp_grp, sp_grp_key in zip(catalog.groups, catalog.groups.keys.iterrows()):
        match sp_grp_key:
            # If catalog group is main-dark or main-bright, we need to further split by HEALPix (nside=1)
            case ('main', 'dark') | ('main', 'bright'):
                sp_hp_cat = split_by_healpix(sp_grp, nside=1)
                for sphp_grp, sphp_grp_key in zip(sp_hp_cat.groups, sp_hp_cat.groups.keys.iterrows()):
                    subcats[sphp_grp_key] = sphp_grp

            # For all other cases, we just need the survey-program split
            case _:
                subcats[sp_grp_key] = sp_grp

    # Iterate over the sub-catalogs to process and write out
    for subcat_key, subcat in subcats.items():
        # Prepare the output file
        primary_hdu = fits.PrimaryHDU()
        subcat_hdu = fits.BinTableHDU(subcat, name='QSO_CAT' if out_prefix == 'QSO' else 'ZCATALOG')
        hdu_list = fits.HDUList([primary_hdu, subcat_hdu])

        # Write the file out copying
        if len(subcat_key) == 3:
            survey, program, nside1_hp = subcat_key
            hdu_list.writeto(f'{out_dir}/{out_prefix}_cat_{specprod}_{survey}_{program}_nside1_hp{nside1_hp:02d}'
                             f'_healpix_all_targets_v{version}.fits', overwrite=True, checksum=True)
        else:
            survey, program = subcat_key
            hdu_list.writeto(f'{out_dir}/{out_prefix}_cat_{specprod}_{survey}_{program}'
                             f'_healpix_all_targets_v{version}.fits',
                             overwrite=True, checksum=True)

if __name__ == '__main__':
    # Split Fuji
    fuji_qsom = '/dvs_ro/cfs/cdirs/desi/users/edmondc/QSO_catalog/fuji/QSO_cat_fuji_healpix_all_targets_v2.fits'
    fuji_zcat = '/dvs_ro/cfs/cdirs/desi/public/edr/vac/edr/zcat/fuji/v1.0/zall-pix-edr-vac.fits'
    fuji_qso_outdir = '/pscratch/sd/b/bfloyd/agngal_incats_tmp/fuji/qsom'
    fuji_zcat_outdir = '/pscratch/sd/b/bfloyd/agngal_incats_tmp/fuji/zcat'

    split_catalogs(fuji_qsom, out_prefix='QSO', out_dir=fuji_qso_outdir, specprod='fuji', version=2)
    split_catalogs(fuji_zcat, out_prefix='zpix', out_dir=fuji_zcat_outdir, specprod='fuji', version=2)

    # Split Iron
    iron_qsom = '/dvs_ro/cfs/cdirs/desi/science/gqp/agncatalog/qsomaker/iron/QSO_cat_iron_healpix_all_targets_v1.fits'
    iron_zcat = '/dvs_ro/cfs/cdirs/desi/spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits'
    iron_qso_outdir = '/pscratch/sd/b/bfloyd/agngal_incats_tmp/iron/qsom'
    iron_zcat_outdir = '/pscratch/sd/b/bfloyd/agngal_incats_tmp/iron/zcat'

    split_catalogs(iron_qsom, out_prefix='QSO', out_dir=iron_qso_outdir, specprod='iron', version=2)
    split_catalogs(iron_zcat, out_prefix='zpix', out_dir=iron_zcat_outdir, specprod='iron', version=2)
