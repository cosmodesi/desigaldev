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

:Version: 1.0 of 2024/10/04


This is the Readme file for the Galaxy Quasar Physics EDR AGN / QSO Summary Catalogs.


Description
===========

This AGN/QSO catalog has been created based on the emission line measurements from FastSpecFit v3.2, correspongind to the Fuji value-added catalog which was publicly released in December 2023 as the DESI Early Data Release (DESI/EDR). Most of the redshifts come from the Redrock catalog. For some QSO, the Redrock redshiftd are incorrect and have been updated using the machine-learning algorithm QuasarNet or the MgII Afterburner.

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

SURVEY  ||  bytes7  ||  -  ||  Survey name

PROGRAM  ||  bytes6  ||  -   || Program name

HEALPIX  ||  int32  ||  -  ||  Healpix number

Z  ||  float64   || -  ||  Redshift

ZERR  ||  float64  ||  -  ||  Redshift error

ZWARN  ||  int64 ||   -  ||  Warning flags (0 is good)

SPECTYPE  ||  bytes6 ||   -  ||  Spectype from Redrock file

COADD_FIBERSTATUS  ||  int32 ||   -  ||  Bitwise-AND of input FIBERSTATUS

TARGET_RA ||   float64 ||   degree ||   Right Ascension in decimal degrees (J2000)

TARGET_DEC ||   float64  ||  degree  ||  Declination in decimal degrees (J2000)

DESI_TARGET ||   int64 ||   -  ||  DESI (dark time program) target selection bitmask

SCND_TARGET ||   int64 ||   -  ||  SCND (secondary program) target selection bitmask

BGS_TARGET ||   int64  ||  -  ||  BGS (bright time program) target selection bitmask

COADD_NUMEXP ||   int16  ||  ?  ||     ?

COADD_EXPTIME ||   float32  ||  ?  ||     ?

CMX_TARGET ||   int64  ||  ?  ||     ?

SV1_DESI_TARGET  || int64 ||    -  ||  DESI (dark time program) target selection bitmask for SV1

SV2_DESI_TARGET ||   int64 ||   - ||   DESI (dark time program) target selection bitmask for SV2

SV3_DESI_TARGET ||   int64 ||   -  ||  DESI (dark time program) target selection bitmask for SV3

SV1_BGS_TARGET ||  int64 ||    -  ||  BGS (bright time program) target selection bitmask for SV1

SV2_BGS_TARGET  || int64 ||   -  ||  BGS (bright time program) target selection bitmask for SV2

SV3_BGS_TARGET ||  int64  ||   -  ||  BGS (bright time program) target selection bitmask for SV3

SV1_SCND_TARGET ||  int64 ||    -  ||  Secondary target selection bitmask for SV1

SV2_SCND_TARGET  || int64 ||    -  ||  Secondary target selection bitmask for SV2

SV3_SCND_TARGET ||  int64  ||   -  ||  Secondary target selection bitmask for SV3

QSO_MASKBITS   || int32  ||  -  ||  QSO selection bitmask

AGN_MASKBITS || int64  ||  -  ||  AGN selection bitmask

AGN_TYPE || int64  ||  -  ||  AGN type: UNKNOWN, TYPE1, TYPE2

SV_PRIMARY || boolean  ||  -  || ??

ZCAT_PRIMARY || boolean  ||  -  || ??


Example
=======

An example notebook *AGNQSO_summary_cat.ipynb* is presented which provides an interactive example of the catalog generation steps.


File location and structure
===========================

Files are located at NERSC. The parent directory is: /global/cfs/cdirs/desi/science/gqp/agncatalog 

