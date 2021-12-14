### Raga Pucha: Modified from the code written by Mar Mezcua to work with DESI Everest data
### To-do: Add detailed comments and function information.
### Version: 2021 December 14

import numpy as np

from astropy.io import fits
from astropy.table import Table, Column, vstack, join
import fitsio
from join_tables import join_zcat_fastspec
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.27)


def Classify_NII_BPT(table):
    fastspec_mask = table['HALPHA_FLUX'].mask     # Depends on the availability of Fastspecfit results
    
    snr = 3

    SNR_Ha = table['HALPHA_FLUX']*np.sqrt(table['HALPHA_FLUX_IVAR'])
    SNR_Hb = table['HBETA_FLUX']*np.sqrt(table['HBETA_FLUX_IVAR'])
    SNR_OIII = table['OIII_5007_FLUX']*np.sqrt(table['OIII_5007_FLUX_IVAR'])
    SNR_NII = table['NII_6584_FLUX']*np.sqrt(table['NII_6584_FLUX_IVAR'])
    
    zero_fluxes = (table['HALPHA_FLUX'] == 0) | (table['HBETA_FLUX'] == 0) | (table['OIII_5007_FLUX']  == 0) |(table['NII_6584_FLUX'] == 0) | fastspec_mask
    
    ## BPT DIAGRAM: NII ##
    #Kewley et al. 2001: starburst vs AGN classification. Solid lines in BPT
    #log10(flux_oiii_5006/flux_hbeta)=0.61/(log10(flux_nii_6583/flux_halpha)-0.47)+1.19
    #Kauffmann et al. 2003: starburst vs composites. Dashed line in BPT
    #log10(flux_oiii_5006/flux_hbeta)=0.61/(log10(flux_nii_6583/flux_halpha)-0.05)+1.3
    #Schawinsky et al. 2007: Seyferts vs LINERS
    #log10(flux_oiii_5006/flux_hbeta)=1.05*log10(flux_nii_6583/flux_halpha)+0.45
    i_bptnii = np.log10(table['NII_6584_FLUX']/table['HALPHA_FLUX'])
    j_bptnii = np.log10(table['OIII_5007_FLUX']/table['HBETA_FLUX'])
    Kew01_nii = 0.61/(i_bptnii-0.47)+1.19
    Scha07 = 1.05*i_bptnii+0.45
    Ka03 = 0.61/(i_bptnii-0.05)+1.3
    
    ## NII-BPT is available (All SNR >= 3)
    nii_bpt = (SNR_Ha >= snr) & (SNR_Hb >= snr) & (SNR_OIII >= snr) & (SNR_NII >= snr) & (~zero_fluxes)

    ## NII-Quiescent -- (SNR < 3 for one or more lines)
    quiescent_nii = ((SNR_Ha<snr) | (SNR_Hb<snr) | (SNR_OIII<snr) | (SNR_NII<snr)) & (~zero_fluxes)
    
    agn_nii = (~quiescent_nii) & ((j_bptnii>=Kew01_nii) & (j_bptnii>Scha07) | (i_bptnii>=0.47))
    liner_nii = (~quiescent_nii) & ((j_bptnii>=Kew01_nii) & (j_bptnii<Scha07) | (i_bptnii>=0.47))
    composite_nii = (~quiescent_nii) & ((j_bptnii>=Ka03) | (i_bptnii>=0.05)) & (~agn_nii)
    sf_nii = (~quiescent_nii) & (~agn_nii) & (~composite_nii) & (~liner_nii)
    
    return (nii_bpt, sf_nii, agn_nii, liner_nii, composite_nii, quiescent_nii)

def Classify_SII_BPT(table):
    fastspec_mask = table['HALPHA_FLUX'].mask       # Fastspec Results Not Available = TRUE
    
    snr = 3
    
    SNR_Ha = table['HALPHA_FLUX']*np.sqrt(table['HALPHA_FLUX_IVAR'])
    SNR_Hb = table['HBETA_FLUX']*np.sqrt(table['HBETA_FLUX_IVAR'])
    SNR_OIII = table['OIII_5007_FLUX']*np.sqrt(table['OIII_5007_FLUX_IVAR'])
    SII_VAR = (1/table['SII_6716_FLUX_IVAR'])+(1/table['SII_6731_FLUX_IVAR'])
    SNR_SII = (table['SII_6716_FLUX']+table['SII_6731_FLUX'])/np.sqrt(SII_VAR)
    
    zero_fluxes = (table['HALPHA_FLUX'] == 0) | (table['HBETA_FLUX'] == 0) | (table['OIII_5007_FLUX'] == 0) | (table['SII_6716_FLUX']+table['SII_6731_FLUX'] == 0) | fastspec_mask
    
    ## BPT DIAGRAM: SII ##
	#Kewley et al. 2001: starburst vs AGN classification. Solid lines in BPT
	#log10(flux_oiii_5006/flux_hbeta)=0.72/(log10(flux_sii_6716,6731/flux_halpha)-0.32)+1.30
	#Kewley et al. 2006: Seyferts vs LINERS
	#log10(flux_oiii_5006/flux_hbeta)=1.89*log10(flux_sii_6716,6731/flux_halpha)+0.76
    i_bptsii = np.log10((table['SII_6716_FLUX']+table['SII_6731_FLUX'])/table['HALPHA_FLUX'])
    j_bptsii = np.log10(table['OIII_5007_FLUX']/table['HBETA_FLUX'])
    Kew01_sii=0.72/(i_bptsii-0.32)+1.30
    Kew06_sii=1.89*i_bptsii+0.76
    
    ## SII-BPT is available (All SNR >= 3)
    sii_bpt = (SNR_Ha >= snr) & (SNR_Hb >= snr) & (SNR_OIII >= snr) & (SNR_SII >= snr) & (~zero_fluxes)
    
    ## SII-Quiescent -- (SNR < 3 for one or more lines)
    quiescent_sii = ((SNR_Ha < snr) | (SNR_Hb < snr) | (SNR_OIII < snr) | (SNR_SII < snr)) & (~zero_fluxes)
    
    agn_sii = (~quiescent_sii) & ((j_bptsii>=Kew01_sii) & (j_bptsii>Kew06_sii) | (i_bptsii>=0.32))
    liner_sii = (~quiescent_sii) & ((j_bptsii>=Kew01_sii) & (j_bptsii<Kew06_sii) | (i_bptsii>=0.32))
    sf_sii = (~quiescent_sii) & (~agn_sii) & (~liner_sii)
    
    return (sii_bpt, sf_sii, agn_sii, liner_sii)

def Classify_OI_BPT(table):
    fastspec_mask = table['HALPHA_FLUX'].mask        # Fastspec Results Not Available = TRUE
    
    snr = 3
    SNR_Ha = table['HALPHA_FLUX']*np.sqrt(table['HALPHA_FLUX_IVAR'])
    SNR_Hb = table['HBETA_FLUX']*np.sqrt(table['HBETA_FLUX_IVAR'])
    SNR_OIII = table['OIII_5007_FLUX']*np.sqrt(table['OIII_5007_FLUX_IVAR'])
    
    zero_fluxes = (table['HALPHA_FLUX'] == 0) | (table['HBETA_FLUX'] == 0) | (table['OIII_5007_FLUX'] == 0) | (table['OI_6300_FLUX'] == 0) | fastspec_mask
    
    ## BPT DIAGRAM: OI ##
    #Kewley et al. 2001: starburst vs AGN classification. Solid lines in BPT
    #log10(flux_oiii_5006/flux_hbeta)=0.73/(log10(flux_oi_6300/flux_halpha)+0.59)+1.33
    #Kewley et al. 2006: Seyferts vs LINERS
    #log10(flux_oiii_5006/flux_hbeta)=1.18*log10(flux_oi_6300/flux_halpha)+1.30
    i_bptoi = np.log10(table['OI_6300_FLUX']/table['HALPHA_FLUX'])
    j_bptoi = np.log10(table['OIII_5007_FLUX']/table['HBETA_FLUX'])
    
    Kew01_oi=0.73/(i_bptoi+0.59)+1.33
    Kew06_oi=1.18*i_bptoi+1.30
    
    ## OI-BPT is available (SNR for 3 lines other than OI >= 3)
    oi_bpt = (SNR_Ha >= snr) & (SNR_Hb >= snr) & (SNR_OIII >= snr) & (~zero_fluxes)
    
    agn_oi = (((j_bptoi>=Kew01_oi) | (i_bptoi>=-0.59)) & (j_bptoi>Kew06_oi)) & oi_bpt
    liner_oi = (((j_bptoi>=Kew01_oi) | (i_bptoi>=-0.59)) & (j_bptoi<Kew06_oi)) & oi_bpt
    sf_oi = (~agn_oi) & (~liner_oi) & oi_bpt
    
    return (oi_bpt, sf_oi, agn_oi, liner_oi)