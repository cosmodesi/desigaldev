###############################
AGN QSO Summary Catalog for EDR
###############################

:Authors:
    Becky Canning,
    Stephanie Juneau,
    Mar Mezcua,
    Raga, 
    GQP, 
    et al.,

:Version: 1.0 of 2024/05/16


This is the Readme file for the Galaxy Quasar Physics EDR AGN / QSO Summary Catalogs.


Description
===========

The 


Version and reproduction
========================

Generating the catalog uses files:
- QSO_cat_fuji_healpix_all_targets.fits
- fastspec-fuji.fits
- fastphot-fuji.fits

To reproduce the catalog run:
>>> python agn_qso_wrapper.py


File location and structure
===========================

Files are located at NERSC. The parent directory is: /global/cfs/cdirs/desi/science/gqp/agncatalog 


Data model
==========
TARGETID: DESI target ID

SURVEY:

PROGRAM:

HEALPIX:

Z:

ZERR:

ZWARN:

SPECTYPE:

COADD_FIBERSTATUS:

TARGET_RA:

TARGET_DEC:

DESI_TARGET:

SCND_TARGET:

BGS_TARGET:

COADD_NUMEXP:

COADD_EXPTIME:

CMX_TARGET:

SV1_DESI_TARGET:

SV2_DESI_TARGET:

SV3_DESI_TARGET:

SV1_BGS_TARGET:

SV2_BGS_TARGET:

SV3_BGS_TARGET:

SV1_SCND_TARGET:

SV2_SCND_TARGET:

SV3_SCND_TARGET:

QSO_MASKBITS:

AGN_MASKBITS:

AGN_TYPE:

SV_PRIMARY:

ZCAT_PRIMARY:


AGN diagnostics
===============
- NII BPT regions defined as:
Kewley et al. (2001): starburst vs AGN classification
Kauffmann et al. (2003): starburst vs composites
Schawinsky et al. (2007): Seyferts vs LINERS

- SII BPT regions defined as:
Kewley et al. (2001): starburst vs AGN classification
Kewley et al. (2006): Seyferts vs LINERS

- OI BPT regions defined as:
Law et al. (2021): starburst vs AGN classification
Kewley et al. (2006): Seyferts vs LINERS

- WHAN diagram from Cid Fernandes et al. (2011)

- BLUE diagram from Lamareille et al (2004) & Lamareille (2010)

- HeII BPT diagram from Shirazi & Brinchmann (2012)

- NeV diagnostic: detection of [NeV]λ3426 implies hard radiation with photon energies above 96.6 eV, indicating AGN

- WISE diagnostics cuts from Jarrett et al. (2011), Mateos et al. (2012), Stern et al. (2012), Yao et al. (2020), and Hviding et al. (2022)
    

Example
=======

An example notebook *AGNQSO_summary_cat.ipynb* is presented which provides an interactive example of the catalog generation steps.

