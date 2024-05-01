###############################
AGN QSO Summary Catalog for EDR
###############################

:Authors:
    Becky Canning,
    Stephanie Juneau,
    Raga, 
    Mar, 
    GQP, 
    et al.,

:Version: 1.0 of 2023/04/04


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

