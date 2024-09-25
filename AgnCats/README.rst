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
Modify this: 

Generating the catalog uses files:
- QSO_cat_fuji_healpix_all_targets.fits
- fastspec-fuji.fits
- fastphot-fuji.fits

To reproduce the catalog run:
>>> python agn_qso_wrapper.py

**AGN diagnostics used in the catalog:**

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
    

Data model
==========
Name  ||   Format   ||  Units  ||  Description

TARGETID ||  int64  ||  -  ||  Unique identifier for each object observed by DESI

SURVEY  ||  bytes3  ||  -  ||  Survey name

PROGRAM  ||  bytes6  ||  -   || Program name

HEALPIX  ||  int32  ||  -  ||  Healpix number

Z  ||  float64   || -  ||  Redshift

ZERR  ||  float64  ||  -  ||  Redshift error

ZWARN  ||  bytes3? ||   -  ||  Warning flags (0 is good)

SPECTYPE  ||  ? ||   -  ||  Spectype from Redrock file

COADD_FIBERSTATUS  ||  ? ||   -  ||  Bitwise-AND of input FIBERSTATUS

TARGET_RA ||   float64 ||   degree ||   Right Ascension in decimal degrees (J2000)

TARGET_DEC ||   float64  ||  degree  ||  Declination in decimal degrees (J2000)

DESI_TARGET ||   int64 ||   -  ||  DESI (dark time program) target selection bitmask

SCND_TARGET ||   int64 ||   -  ||  SCND (secondary program) target selection bitmask

BGS_TARGET ||   int64  ||  -  ||  BGS (bright time program) target selection bitmask

COADD_NUMEXP

COADD_EXPTIME

CMX_TARGET

SV1_DESI_TARGET  || ? ||    -  ||  DESI (dark time program) target selection bitmask for SV1

SV2_DESI_TARGET ||   ? ||   - ||   DESI (dark time program) target selection bitmask for SV2

SV3_DESI_TARGET ||   ? ||   -  ||  DESI (dark time program) target selection bitmask for SV3

SV1_BGS_TARGET ||  ? ||    -  ||  BGS (bright time program) target selection bitmask for SV1

SV2_BGS_TARGET  || ? ||   -  ||  BGS (bright time program) target selection bitmask for SV2

SV3_BGS_TARGET ||  ?  ||   -  ||  BGS (bright time program) target selection bitmask for SV3

SV1_SCND_TARGET ||  ? ||    -  ||  Secondary target selection bitmask for SV1

SV2_SCND_TARGET  || ? ||    -  ||  Secondary target selection bitmask for SV2

SV3_SCND_TARGET ||  ?  ||   -  ||  Secondary target selection bitmask for SV3

QSO_MASKBITS   || ?  ||  -  ||  QSO selection bitmask

AGN_MASKBITS

AGN_TYPE

SV_PRIMARY

ZCAT_PRIMARY


Example
=======

An example notebook *AGNQSO_summary_cat.ipynb* is presented which provides an interactive example of the catalog generation steps.


File location and structure
===========================

Files are located at NERSC. The parent directory is: /global/cfs/cdirs/desi/science/gqp/agncatalog 

