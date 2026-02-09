"""
xray_agn_diagnostics.py
Author: Benjamin Floyd

Library containing all X-ray AGN/galaxy diagnostic functions.

Original version of code written by:
Becky Canning (University of Portsmouth)
Stephanie Juneau (NOIRlab)
Mar Mezcula (Institut de Ciencies de l'Espai)
"""
import numpy as np


# TODO: BenFloyd - This selection still needs to be refactored
def Xray(input, H0=67.4, Om0=0.315, snr=3):
    ## X-ray diagnostic ##
    # 2-10 keV X-ray luminosity equal or above 1e42 erg/s indicates AGN
    thres = 1e42

    # Fiducial Cosmology used in DESI from Planck 2018 results: https://ui.adsabs.harvard.edu/abs/2020A%26A...641A...6P/abstract
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    DL = cosmo.luminosity_distance(input['z'].values)  # in Mpc
    DL_cm = 3.08567758e24 * DL.value

    # Convert the CSC 2-7 keV flux to 2-10 keV and compute the LX2-10 keV
    # Conversion factor =  1.334E-15 erg cm^-2 s^-1 for an unabsorbed flux of 1E-15 erg cm^-2 s^-1 using PIMMS (https://cxc.harvard.edu/toolkit/pimms.jsp) and assuming gamma=1.8
    factor = 1.334E-15  # for gamma=1.8
    flux_2_10 = (input['FLUX_2_7'] / 1E-15) * factor
    LX210 = 4 * np.pi * DL_cm ** 2 * flux_2_10  # in erg/s

    # Apply K-correction:
    gamma = 1.8
    k = (1 + input['z']) ** (gamma - 2)
    LX210_Kcorr = k * LX210

    # Mask for zero flux
    zero_flux_xray = input['FLUX_2_7'] == 0

    # Mask for SNR
    snr = snr
    SNR_Xray = input['FLUX_2_7'] / input['FLUX_2_7_err']

    ## Xray diagnostic is available if flux is not zero and SNR_Xray >= 3
    xray = (SNR_Xray >= snr) & (~zero_flux_xray)

    ## Xray-AGN, SF, footprint
    agn_xray = (xray) & (LX210_Kcorr >= thres)
    sf_xray = (xray) & ~agn_xray
    fp_xray = (zero_flux_xray) & (input['FLUX_2_7_err'] > 0)

    return (agn_xray, sf_xray, fp_xray)
