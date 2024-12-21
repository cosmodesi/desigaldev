AGN QSO Summary Catalog for DR1
======================

Authors:
    Becky Canning,
    Stephanie Juneau,
    Mar Mezcua,
    Raga Pucha, 
    GQP, 
    et al.,

Version: 0.0 of 2024/12/20


This is the Readme file for the Galaxy Quasar Physics EDR AGN / QSO Summary Catalogs.


Description
===========

This AGN/QSO catalog has been created based on the emission line measurements from FastSpecFit v2.1, corresponding to the Iron value-added catalog which will publicly released in 2025 as the DESI First Data Release (DESI/DR1): https://fastspecfit.readthedocs.io/en/latest/iron.html. Most of the redshifts come from the Redrock catalog. For some QSO, the Redrock redshifts are incorrect and have been updated using the machine-learning algorithm QuasarNet or the MgII Afterburner.

**AGN diagnostics used in the catalog:**

- NII BPT regions defined as:

  Kewley et al. (2001): starburst vs AGN classification 

  Kauffmann et al. (2003): starburst vs composites

  Schawinsky et al. (2007): Seyferts vs LINERS

- SII BPT regions defined as:
  Kewley et al. (2001): starburst vs AGN classification

  Law et al. (2021): Seyferts vs LINERS

- OI BPT regions defined as:
  Law et al. (2021): starburst vs AGN classification

  Law et al. (2021): Seyferts vs LINERS

- WHAN diagram from Cid Fernandes et al. (2011)

- BLUE diagram from Lamareille et al (2004) & Lamareille (2010)

- HeII BPT diagram from Shirazi & Brinchmann (2012)

- NeV diagnostic: detection of [NeV]λ3426 implies hard radiation with photon energies above 96.6 eV, indicating AGN

- WISE diagnostics cuts from Jarrett et al. (2011), Mateos et al. (2012), Stern et al. (2012), Assef et al. (2018), Yao et al. (2020), and Hviding et al. (2022)
    

Data model
==========
**EXTENSIONS**
| Number   | EXTNAME   |   Ver  |  Type    |  Dimensions   | Contents   |
| ----- | ------ |------ | ------    |    ------  |    ------  | 
|  HDU00  | PRIMARY  |     1  | PrimaryHDU |   ()    |  Keywords only |   
|  HDU01  | AGNCAT   |     1  | BinTableHDU |  1397603R x 35C  | Table with summary catalog |
|  HDU02  | AUXDATA  |    1   | BinTableHDU |    1397603R x 58C   | Table with fluxes |

**HDU01**

EXTNAME = AGNCAT

| Name  | Format | Units | Description |
| ----- | ------ |------ | ------ |
| TARGETID |  int64  |  -  |  Unique identifier for each object observed by DESI |
| SURVEY  |  char[7]  |  -  |  Survey name |
| PROGRAM |  char[6]  |  -   | DESI program type - BRIGHT, DARK, BACKUP, OTHER |
| HEALPIX  |  int32  |  -  |  Healpix number |
| Z  |  float64   | -  |  Redshift from the FastSpecFit catalog |
| ZERR  |  float64  |  -  |  Redshift error|
| ZWARN |  int64 |   -  |  Warning flags (0 is good) |
| SPECTYPE  |  char[6] |   -  |  Spectral type of Redrock best fit template (e.g. GALAXY, QSO, STAR) |
| COADD_FIBERSTATUS  |  int32 |  -  |  Bitwise-AND of input FIBERSTATUS |
| TARGET_RA |   float64 |   degr |   Right Ascension in decimal degrees (J2000) |
| TARGET_DEC |   float64  |  deg  |  Declination in decimal degrees (J2000) |
| DESI_TARGET |   int64 |   -  |  DESI (dark time program) target selection bitmask |
| SCND_TARGET |   int64 |  -  | SCND (secondary program) target selection bitmask |
| BGS_TARGET |   int64  |  -  |  BGS (bright time program) target selection bitmask |
| COADD_NUMEXP |   int16  |  -   |     Number of exposures in coadd |
| COADD_EXPTIME |   float32  |  s |     Summed exposure time for coadd |
| CMX_TARGET |   int64  | -  |   Target selection bitmask for commissioning |
| SV1_DESI_TARGET  | int64 |    -  |  DESI (dark time program) target selection bitmask for SV1 |
| SV2_DESI_TARGET |   int64 |   - |   DESI (dark time program) target selection bitmask for SV2 |
| SV3_DESI_TARGET |   int64 |   -  |  DESI (dark time program) target selection bitmask for SV3 |
| SV1_BGS_TARGET |  int64 |    -  |  BGS (bright time program) target selection bitmask for SV1 |
| SV2_BGS_TARGET  | int64 |   -  |  BGS (bright time program) target selection bitmask for SV2 |
| SV3_BGS_TARGET |  int64  |   -  |  BGS (bright time program) target selection bitmask for SV3 |
| SV1_SCND_TARGET |  int64 |    -  |  Secondary target selection bitmask for SV1 |
| SV2_SCND_TARGET  | int64 |    -  |  Secondary target selection bitmask for SV2 |
| SV3_SCND_TARGET |  int64  |   -  |  Secondary target selection bitmask for SV3 |
| AGN_MASKBITS | int64  |  -  |  AGN selection bitmask. AGN_ANY: any AGN classification is set, RR: Redrock determines this to be a QSO from template fitting, MGII: MgII Afterburner detects broad line, QN: QuasarNet reclassifies as a QSO, QN_NEW_RR: QuasarNet prompts different Redrock redshift, QN_BGS: QuasarNet reclassifies BGS target as a QSO, QN_ELG: QuasarNet reclassifies ELG target as a QSO, QN_VAR_WISE: QuasarNet reclassifies VAR_WISE_QSO target as a QSO, BPT_ANY_SY: At least one BPT diagnostic indicates SEYFERT, BPT_ANY_AGN: At least one BPT diagnostic indicates SEYFERT, LINER or COMPOSITE, BROAD_LINE: Lines with FWHN >=1200 km/s in Halpha, Hbeta, MgII and/or CIV line, OPT_OTHER_AGN: Rest frame optical emission lines diagnostic not BPT (4000-10000 ang) indicate AGN, UV: Rest frame UV emission lines indicate AGN, WISE: At least one infrared (WISE) colour diagnostic indicates AGN |
| AGN_TYPE | int64  |  -  |  AGN detailed type information. NII_BPT: NII BPT diagnostic is available, NII_SF: NII BPT Star-forming, NII_COMP: NII BPT Composite, NII_SY: NII BPT Seyfert, NII_LINER: NII BPT LINER, SII_BPT: SII BPT diagnostic is available, SII_SF: SII BPT Star-forming, SII_SY: SII BPT Seyfert, SII_LINER: SII BPT LINER, OI_BPT: OI BPT diagnostic is available, OI_SF: OI BPT Star-forming, OI_SY: OI BPT Seyfert, OI_LINER: OI BPT LINER, WHAN: WHAN is available (Halpha and [NII]), WHAN_SF: WHAN Star-forming, WHAN_SAGN: WHAN Strong AGN, WHAN_WAGN: WHAN Weak AGN, WHAN_RET: WHAN Retired, WHAN_PASS: WHAN Passive, BLUE: Blue diagram available, BLUE_AGN: Blue diagram AGN, BLUE_SLC: Blue diagram Star-forming/LINER/Composite, BLUE_LINER: Blue diagram LINER, BLUE_SF: Blue diagram Star-forming, BLUE_SFAGN: Blue diagram Star-forming/AGN, HEII_BPT: He II BPT diagnostic is available, HEII_AGN: He II BPT AGN, HEII_SF: He II BPT Star-forming, NEV: Ne V is available, NEV_AGN: NE V AGN, NEV_SF: Ne V Star-forming, WISE_W12: WISE W1 and W2 available, WISE_W123: WISE W1, W2 and W3 available, WISE_AGN_J11: WISE diagnostic Jarrett et al. 2011 is AGN (based on W1,W2,W3), WISE_SF_J11:  WISE diagnostic Jarrett et al. 2011 is not an AGN (based on W1,W2,W3), WISE_AGN_S12: WISE diagnostic Stern et al. 2012 is AGN (based on W1,W2), WISE_SF_S12: WISE diagnostic Stern et al. 2012 is not an AGN (based on W1,W2), WISE_AGN_M12: WISE diagnostic Mateos et al. 2012 is AGN (based on W1,W2,W3), WISE_SF_M12: WISE diagnostic Mateos et al. 2012 is not an AGN (based on W1,W2,W3), WISE_AGN_A18: WISE diagnostic Assef et al. 2018 is AGN (based on W1,W2), WISE_SF_A18: WISE diagnostic Assef et al. 2018 is not an AGN (based on W1,W2), WISE_AGN_Y20: WISE diagnostic Yao et al. 2020 is AGN (based on W1,W2,W3), WISE_SF_Y20: WISE diagnostic Yao et al. 2020 is not an AGN (based on W1,W2,W3), WISE_AGN_H22: WISE diagnostic Hviding et al. 2022 is AGN (based on W1,W2,W3), WISE_SF_H22: WISE diagnostic Hviding et al. 2022 is not an AGN (based on W1,W2,W3) |
| SV_PRIMARY | logical  |  -  | Boolean flag (True/False) for the primary coadded spectrum in SV (SV1+2+3) |
| ZCAT_PRIMARY | logical  |  -  | Boolean flag (True/False) for the primary coadded spectrum in the zcatalog |

**HDU02**

EXTNAME = AUXDATA

| Name  | Format | Units | Description |
| ----- | ------ |------ | ------ |
| TARGETID |  int64  |  -  |  Unique identifier for each object observed by DESI |
| SURVEY  |  char[7]  |  -  |  Survey name |
| PROGRAM |  char[6]  |  -   | DESI program type - BRIGHT, DARK, BACKUP, OTHER |
| LOGMSTAR | float32  |  Msun | Logarithmic stellar mass (h=1.0, Chabrier+2003 initial mass function) |
| FLUX_W1  |  float32  | nanomaggy | WISE flux in W1 (AB) |
| FLUX_W2  |  float32  | nanomaggy | WISE flux in W2 (AB) |
| FLUX_W3  |  float32  | nanomaggy | WISE flux in W3 (AB) |
| FLUX_IVAR_W1	| float32 |	nanomaggy^-2 | Inverse variance of FLUX_W1 (AB) |
| FLUX_IVAR_W2	| float32 |	nanomaggy^-2 | Inverse variance of FLUX_W2 (AB) |
| FLUX_IVAR_W3	| float32 |	nanomaggy^-2 | Inverse variance of FLUX_W3 (AB) |
| CIV_1549_FLUX	| float32	| 10**-17 erg/(s cm2)| 	Gaussian-integrated emission-line flux |
| CIV_1549_FLUX_IVAR	| float32| 	10**+34 (s2 cm4) / erg2	| Inverse variance of integrated flux |
| CIV_1549_SIGMA	| float32	| km / s	| Gaussian emission-line width |
| MGII_2796_FLUX	| float32	| 10**-17 erg/(s cm2)| 	Gaussian-integrated emission-line flux |
| MGII_2796_FLUX_IVAR	| float32| 	10**+34 (s2 cm4) / erg2	| Inverse variance of integrated flux |
| MGII_2796_SIGMA	| float32	| km / s	| Gaussian emission-line width |
| MGII_2803_FLUX	| float32	| 10**-17 erg/(s cm2)| 	Gaussian-integrated emission-line flux |
| MGII_2803_FLUX_IVAR	| float32| 	10**+34 (s2 cm4) / erg2	| Inverse variance of integrated flux |
| MGII_2803_SIGMA	| float32	| km / s	| Gaussian emission-line width |
| OII_3726_FLUX	| float32	| 10**-17 erg/(s cm2)| 	Gaussian-integrated emission-line flux |
| OII_3726_FLUX_IVAR	| float32| 	10**+34 (s2 cm4) / erg2	| Inverse variance of integrated flux |
| OII_3726_EW	| float32| 	Angstrom	| Rest-frame emission-line equivalent width | 
| OII_3726_EW_IVAR	| float32	| 1 / Angstrom2	| Inverse variance of equivalent width |
| OII_3729_FLUX	| float32	| 10**-17 erg/(s cm2)| 	Gaussian-integrated emission-line flux |
| OII_3729_FLUX_IVAR	| float32| 	10**+34 (s2 cm4) / erg2	| Inverse variance of integrated flux |
| OII_3729_EW	| float32| 	Angstrom	| Rest-frame emission-line equivalent width | 
| OII_3729_EW_IVAR	| float32	| 1 / Angstrom2	| Inverse variance of equivalent width |
| NEV_3426_FLUX	| float32	| 10**-17 erg/(s cm2)| 	Gaussian-integrated emission-line flux |
| NEV_3426_FLUX_IVAR	| float32| 	10**+34 (s2 cm4) / erg2	| Inverse variance of integrated flux |
| HEII_4686_FLUX	| float32	| 10**-17 erg/(s cm2)| 	Gaussian-integrated emission-line flux |
| HEII_4686_FLUX_IVAR	| float32| 	10**+34 (s2 cm4) / erg2	| Inverse variance of integrated flux |
| HBETA_FLUX	| float32	| 10**-17 erg/(s cm2)| 	Gaussian-integrated emission-line flux |
| HBETA_FLUX_IVAR	| float32| 	10**+34 (s2 cm4) / erg2	| Inverse variance of integrated flux|
| HBETA_EW	| float32| 	Angstrom	| Rest-frame emission-line equivalent width | 
| HBETA_EW_IVAR	| float32	| 1 / Angstrom2	| Inverse variance of equivalent width |
| HBETA_BROAD_FLUX	| float32	| 10**-17 erg/(s cm2)| 	Gaussian-integrated emission-line flux |
| HBETA_BROAD_FLUX_IVAR	| float32| 	10**+34 (s2 cm4) / erg2	| Inverse variance of integrated flux|
| HBETA_BROAD_SIGMA	| float32| 	km / s	| Gaussian emission-line width | 
| HBETA_BROAD_CHI2	| float32	| Angstrom	| Chi-squared of the line-fit |
| OIII_5007_FLUX	| float32	| 10**-17 erg/(s cm2)| 	Gaussian-integrated emission-line flux |
| OIII_5007_FLUX_IVAR	| float32| 	10**+34 (s2 cm4) / erg2	| Inverse variance of integrated flux |
| OIII_5007_SIGMA	| float32	| km / s	| Gaussian emission-line width |
| OI_6300_FLUX	| float32	| 10**-17 erg/(s cm2)| 	Gaussian-integrated emission-line flux |
| OI_6300_FLUX_IVAR	| float32| 	10**+34 (s2 cm4) / erg2	| Inverse variance of integrated flux |
| HALPHA_FLUX	| float32	| 10**-17 erg/(s cm2)| 	Gaussian-integrated emission-line flux |
| HALPHA_FLUX_IVAR	| float32| 	10**+34 (s2 cm4) / erg2	| Inverse variance of integrated flux |
| HALPHA_EW	| float32| 	Angstrom	| Rest-frame emission-line equivalent width | 
| HALPHA_SIGMA	| float32	| km / s	| Gaussian emission-line width |
| HALPHA_BROAD_FLUX	| float32	| 10**-17 erg/(s cm2)| 	Gaussian-integrated emission-line flux |
| HALPHA_BROAD_FLUX_IVAR	| float32| 	10**+34 (s2 cm4) / erg2	| Inverse variance of integrated flux|
| HALPHA_BROAD_SIGMA	| float32| 	km / s	| Gaussian emission-line width | 
| HALPHA_BROAD_VSHIFT	| float32	| km / s	| Velocity shift relative to Z |
| NII_6584_FLUX	| float32	| 10**-17 erg/(s cm2)| 	Gaussian-integrated emission-line flux |
| NII_6584_FLUX_IVAR	| float32| 	10**+34 (s2 cm4) / erg2	| Inverse variance of integrated flux |    
| SII_6716_FLUX	| float32	| 10**-17 erg/(s cm2)| 	Gaussian-integrated emission-line flux |
| SII_6716_FLUX_IVAR	| float32| 	10**+34 (s2 cm4) / erg2	| Inverse variance of integrated flux |
| SII_6731_FLUX	| float32	| 10**-17 erg/(s cm2)| 	Gaussian-integrated emission-line flux |
| SII_6731_FLUX_IVAR	| float32| 	10**+34 (s2 cm4) / erg2	| Inverse variance of integrated flux |


Example
=======

An example notebook *how_to_use_AGN_summary.ipynb* is presented which provides an interactive example of the catalog generation steps.


File location and structure
===========================

Files are located at NERSC. The parent directory is: /global/cfs/cdirs/desi/science/gqp/agncatalog 
