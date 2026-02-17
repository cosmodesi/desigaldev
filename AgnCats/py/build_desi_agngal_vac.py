"""
build_desi_agngal_vac.py
Author: Benjamin Floyd

This top-level script builds the DESI AGN/Galaxy Classification VAC. This supersedes the DR1 00_AGNQSO_summary_cat.ipynb
notebook and provides parallelized computation abilities in constructing the final catalog.
"""

from pathlib import Path

import fitsio
from astropy.table import Table, hstack, join

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

qso_maker_catalog = Table(fitsio.read(str(desi_specprod['SPECPROD']['qso_maker']), ext=1, columns=qso_maker_cols))

zcat_cols = ['DESI_TARGET','BGS_TARGET','SCND_TARGET','CMX_TARGET','SV1_DESI_TARGET','SV1_BGS_TARGET','SV1_SCND_TARGET',
               'SV2_DESI_TARGET','SV2_BGS_TARGET','SV2_SCND_TARGET','SV3_DESI_TARGET','SV3_BGS_TARGET','SV3_SCND_TARGET']

redshift_catalog = Table(fitsio.read(str(desi_specprod['SPECPROD']['zcat']), ext=1, columns=zcat_cols + desi_specprod['SPECPROD']['zcat_cols']))

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

fsf_data_catalog = Table(fitsio.read(str(desi_specprod['SPECPROD']['fast_spec']), columns=fsf_data_cols, ext=1))
fsf_meta_catalog = Table(fitsio.read(str(desi_specprod['SPECPROD']['fast_spec']), columns=fsf_meta_cols, ext=2))
fsf_meta_catalog.remove_columns(set(fsf_data_catalog.colnames).intersection(fsf_meta_catalog.colnames))

fsf_catalog = hstack([fsf_data_catalog, fsf_meta_catalog])

# Main identifiers for Joins
keys_for_join=['TARGETID','SURVEY','PROGRAM']

# Join FSF with qso-maker then zcat
fsf_qsom = join(fsf_catalog, qso_maker_catalog, keys=keys_for_join, join_type='left')

# Do asserts to check if joined table maintains consistency
assert all(fsf_qsom['OBJTYPE'] == 'TGT') # This was originally done after the qsom-zcat join
assert all(fsf_qsom['Z'] > 0.001)
assert all(fsf_qsom['COADD_EXPTIME'] > 0.0)

fsf_qsom_zcat = join(fsf_qsom, redshift_catalog, keys=keys_for_join, join_type='left')

# Final check to make sure that we retain the same number of rows (probably not necessary)
assert len(fsf_qsom_zcat) == len(fsf_qsom)

# def read_input_catalogs():

# Begin AGN/Galaxy classifications
