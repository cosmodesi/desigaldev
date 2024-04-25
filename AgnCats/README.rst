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


Example
=======

An example notebook *AGNQSO_summary_cat.ipynb* is presented which provides an interactive example of the catalog generation steps.

