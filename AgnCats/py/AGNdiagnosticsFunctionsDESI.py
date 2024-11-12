import numpy as np

'''notes for us:
Find/replace: Mar_&_Steph_2023 with correct reference 
Find/replace: Summary_ref_2023 with correct reference
Find/replace: FastSpecFit_ref with correct reference
'''

##########################################################################################################
##########################################################################################################

###
### BC: let's discuss 'unknown' once more 3rd Nov 2024 and also which fluxes to use when FSF does fit _BROAD_
###

def AGN_OPTICAL_UV_TYPE(input, snr=3, mask=None):
    '''
    If using these diagnostic fuctions please ref the appropriate references given below.
    
    If using DESI please reference Summary_ref_2023 and the apprpriate
    emission line catalog (e.g. FastSpecFit ref FastSpecFit_ref)

    Inputs:
    'table' including Ha, Hb, MgII and CIV emission lines.
    'snr' is the snr cut applied to all axes. Default is 3.
    'mask' is an optional mask (e.g. from masked column array). Default is None.

    Outputs:
    Vectors of same dimension as rows in table which include flags for:
    type_1, type_2, type_unknown
    
    'type_1' Optical/UV Type 1 AGN (FWHM>=1200 km/s in Halpha, Hbeta, MgII and/or CIV line)
    
    'type_2' Optical/UV Type 2 AGN (FWHM<1200 km/s in Halpha, Hbeta, MgII and/or CIV line)
    
    'type_unknown' Optical/UV AGN lacking Halpha, Hbeta, MgII and CIV line constraints as 
    the lines are either not int he spectra or they are not bright enough 
    '''
    
    # Mask for zero fluxes
    zero_flux_nii = (input['HALPHA_FLUX'] == 0) | (input['HBETA_FLUX'] == 0) | \
                    (input['OIII_5007_FLUX']  == 0) |(input['NII_6584_FLUX'] == 0)
    if mask != None:
        # Mask for flux avalibility - included as fastspecfit columns are maskedcolumn data
        mask = mask     
        zero_flux_nii = (input['HALPHA_FLUX'] == 0) | (input['HBETA_FLUX'] == 0) | \
                        (input['OIII_5007_FLUX']  == 0) |(input['NII_6584_FLUX'] == 0) | mask

    #If ivar=0 set it to NaN to avoid infinites when computing the error:
    input['HALPHA_FLUX_IVAR']=np.where(input['HALPHA_FLUX_IVAR']==0,np.nan,input['HALPHA_FLUX_IVAR'])
    input['HBETA_FLUX_IVAR']=np.where(input['HBETA_FLUX_IVAR']==0,np.nan,input['HBETA_FLUX_IVAR'])
    input['MGII_2796_FLUX_IVAR']=np.where(input['MGII_2796_FLUX_IVAR']==0,np.nan,input['MGII_2796_FLUX_IVAR'])
    input['MGII_2803_FLUX_IVAR']=np.where(input['MGII_2803_FLUX_IVAR']==0,np.nan,input['MGII_2803_FLUX_IVAR'])
    input['CIV_1549_FLUX_IVAR']=np.where(input['CIV_1549_FLUX_IVAR']==0,np.nan,input['CIV_1549_FLUX_IVAR'])

    # Mask for SNR. Default is TYPE is available to determine if one of the lines SNR >= 3
    snr = snr
    SNR_Ha=input['HALPHA_FLUX']*np.sqrt(input['HALPHA_FLUX_IVAR'])

    SNR_Hb=input['HBETA_FLUX']*np.sqrt(input['HBETA_FLUX_IVAR'])
    #HBETA_BROAD_FLUX
    #HALPHA_BROAD_FLUX
    
    SNR_MG_2796=input['MGII_2796_FLUX']*np.sqrt(input['MGII_2796_FLUX_IVAR'])
    SNR_MG_2803=input['MGII_2803_FLUX']*np.sqrt(input['MGII_2803_FLUX_IVAR'])
    SNR_CIV=input['CIV_1549_FLUX']*np.sqrt(input['CIV_1549_FLUX_IVAR'])

    # Define breadth in FWHM in kmps
    broad_fwhm_HALPHA = HALPHA_SIGMA * (2. * np.sqrt(2. * np.log(2.)))
    broad_fwhm_HBETA = HBETA_SIGMA * (2. * np.sqrt(2. * np.log(2.)))
    broad_fwhm_MGII_2796 = MGII_2796_SIGMA * (2. * np.sqrt(2. * np.log(2.)))
    broad_fwhm_MGII_2803 = MGII_2803_SIGMA * (2. * np.sqrt(2. * np.log(2.)))
    broad_fwhm_CIV = CIV_1549_SIGMA * (2. * np.sqrt(2. * np.log(2.)))
    
    ## Type is unknown if all lines SNR <= 3 otherwise type is known and we classify as type 1 or type 2 based on velocities
    type_known = (SNR_Ha >= snr) & (SNR_Hb >= snr) & (SNR_MG_2796 >= snr) & (SNR_MG_2803 >= snr) & (SNR_CIV >= snr) & (~zero_flux_nii)
    type_unknown = (SNR_Ha < snr) & (SNR_Hb < snr) & (SNR_MG_2796 < snr) & (SNR_MG_2803 < snr) & (SNR_CIV < snr) & (zero_flux_nii)
    
    ## SJ: Type UNKNOWN should be an Optical/UV AGN (but lacking the S/N for all Ha, Hb, MgII, CIV)

    ## Broad lines classifying an AGN Type 1 or 2
    max_fwhm = max([broad_fwhm_HALPHA,broad_fwhm_HBETA,broad_fwhm_MGII_2796,broad_fwhm_MGII_2803,broad_fwhm_CIV])
    type_1 = max_fwhm >= 1200.
    type_2 = max_fwhm < 1200.
    
    return (type_1, type_2, type_unknown)

##########################################################################################################
##########################################################################################################
def NII_BPT(input, snr=3, mask=None):
    '''
    If using these diagnostic fuctions please ref the appropriate references given below.
    
    If using DESI please reference Summary_ref_2023 and the apprpriate
    emission line catalog (e.g. FastSpecFit ref FastSpecFit_ref)
    
    --NII diagram originally from BPT81--
    
    Inputs:
    'table' including NII, Ha, OIII, Hb and inverse variances.
    'snr' is the snr cut applied to all axes. Default is 3.
    'mask' is an optional mask (e.g. from masked column array). Default is None.
        
    Outputs:
    Vectors of same dimension as rows in table which include flags for:
    nii_bpt, sf_nii, agn_nii, liner_nii, composite_nii, quiescent_nii
    
    'nii_bpt' flag for SNR in all lines higher than snr and no zerofluxes
    'quiescent_nii' flag for SNR less than snr in one or many lines no zerofluxes

    'agn_nii' a Kew01 AGN and Scha07 Seyfert
    flag SNR>snr & [ log(oiii/hb)>=Kew01_nii & log(oiii/hb)>Scha07 | log(nii/ha)>=0.47 ]

    'liner_nii' a Kew01 AGN and Scha07 Liner
    flag SNR>snr & [ log(oiii/hb)>=Kew01_nii & log(oiii/hb)<Scha07 | log(nii/ha)>=0.47 ]

    'composite_nii a Ka03 composite but not 'agn_nii'
    flag SNR>snr & not agn_nii & [ log(oiii/hb)>=Ka03 | log(nii/ha)>=0.05 ] 
        
    'sf_nii'
    flag SNR>snr & not agn_nii & not liner & not composite
 
    BPT regions defined as:
    #Kewley et al. 2001: starburst vs AGN classification.
    Kew01_nii: log10(flux_oiii_5006/flux_hbeta)=0.61/(log10(flux_nii_6583/flux_halpha)-0.47)+1.19
    #Kauffmann et al. 2003: starburst vs composites.
    Ka03: log10(flux_oiii_5006/flux_hbeta)=0.61/(log10(flux_nii_6583/flux_halpha)-0.05)+1.3
    #Schawinsky et al. 2007: Seyferts vs LINERS
    Scha07: log10(flux_oiii_5006/flux_hbeta)=1.05*log10(flux_nii_6583/flux_halpha)+0.45
    
    Other BPT regions not implemented here:
    #Law et al. 2021 proposed revised lines based on MaNGA observation (not implemented b/c similar to Ka03):
    log10(flux_oiii_5006/flux_hbeta)=0.438/(log10(flux_nii_6583/flux_halpha)+0.023)+1.222
    #Law et al. define an extra "intermediate" region (not yet implemented)
    '''
    # Mask for zero fluxes
    zero_flux_nii = (input['HALPHA_FLUX'] == 0) | (input['HBETA_FLUX'] == 0) | \
                    (input['OIII_5007_FLUX']  == 0) |(input['NII_6584_FLUX'] == 0)
    if mask != None:
        # Mask for flux avalibility - included as fastspecfit columns are maskedcolumn data
        mask = mask     
        zero_flux_nii = (input['HALPHA_FLUX'] == 0) | (input['HBETA_FLUX'] == 0) | \
                        (input['OIII_5007_FLUX']  == 0) |(input['NII_6584_FLUX'] == 0) | mask
   
    #If ivar=0 set it to NaN to avoid infinites when computing the error:
    input['HALPHA_FLUX_IVAR']=np.where(input['HALPHA_FLUX_IVAR']==0,np.nan,input['HALPHA_FLUX_IVAR'])
    input['HBETA_FLUX_IVAR']=np.where(input['HBETA_FLUX_IVAR']==0,np.nan,input['HBETA_FLUX_IVAR'])
    input['OIII_5007_FLUX_IVAR']=np.where(input['OIII_5007_FLUX_IVAR']==0,np.nan,input['OIII_5007_FLUX_IVAR'])
    input['NII_6584_FLUX_IVAR']=np.where(input['NII_6584_FLUX_IVAR']==0,np.nan,input['NII_6584_FLUX_IVAR'])

    # Mask for SNR. Default is NII-BPT is available if all SNR >= 3
    snr = snr
    SNR_Ha=input['HALPHA_FLUX']*np.sqrt(input['HALPHA_FLUX_IVAR'])
    SNR_Hb=input['HBETA_FLUX']*np.sqrt(input['HBETA_FLUX_IVAR'])
    SNR_OIII=input['OIII_5007_FLUX']*np.sqrt(input['OIII_5007_FLUX_IVAR'])
    SNR_NII=input['NII_6584_FLUX']*np.sqrt(input['NII_6584_FLUX_IVAR'])

    # Define regions
    log_nii_ha=np.log10(input['NII_6584_FLUX']/input['HALPHA_FLUX'])
    log_oiii_hb=np.log10(input['OIII_5007_FLUX']/input['HBETA_FLUX'])
    Kew01_nii=0.61/(log_nii_ha-0.47)+1.19
    Scha07=1.05*log_nii_ha+0.45
    Ka03=0.61/(log_nii_ha-0.05)+1.3

    ## NII-BPT is available (All lines SNR >= 3)
    nii_bpt = (SNR_Ha >= snr) & (SNR_Hb >= snr) & (SNR_OIII >= snr) & (SNR_NII >= snr) & (~zero_flux_nii)

    ## NII-Quiescent (SNR < 3 for one or more lines)
    quiescent_nii=((SNR_Ha<snr) | (SNR_Hb<snr) | (SNR_OIII<snr) | (SNR_NII<snr)) & (~zero_flux_nii) 

    ## NII-AGN, LINER, COMP, SF
    agnliner_nii=(nii_bpt) & ((log_oiii_hb>=Kew01_nii) | (log_nii_ha>=0.47))
    agn_nii=(agnliner_nii) & (log_oiii_hb>=Scha07) 
    liner_nii=(agnliner_nii) & (log_oiii_hb<Scha07) 
    composite_nii=(nii_bpt) & ((log_oiii_hb>=Ka03) | (log_nii_ha>=0.05)) & (~agnliner_nii)
    sf_nii=(nii_bpt) & (~agnliner_nii) & (~composite_nii)
    
    return (nii_bpt, sf_nii, agn_nii, liner_nii, composite_nii, quiescent_nii)
    
def NII_BPT_lines(x_axes):
    '''
    This function draws the lines for the BPT regions int he NII_BPT plot
    
    Kewley et al. 2001: starburst vs AGN classification.
    Kew01_nii: log10(flux_oiii_5006/flux_hbeta)=0.61/(log10(flux_nii_6583/flux_halpha)-0.47)+1.19

    Kauffmann et al. 2003: starburst vs composites.
    Ka03: log10(flux_oiii_5006/flux_hbeta)=0.61/(log10(flux_nii_6583/flux_halpha)-0.05)+1.3
    
    Schawinsky et al. 2007: Seyferts vs LINERS
    Scha07: log10(flux_oiii_5006/flux_hbeta)=1.05*log10(flux_nii_6583/flux_halpha)+0.45
    
    Other BPT regions not implemented here yet:
    
    Law et al. 2021 proposed revised lines based on MaNGA observation (not implemented b/c similar to Ka03):
    log10(flux_oiii_5006/flux_hbeta)=0.438/(log10(flux_nii_6583/flux_halpha)+0.023)+1.222
    
    Law et al. define an extra "intermediate" region (not yet implemented)
    '''
    Kew01_nii=0.61/(x_axes-0.47)+1.19
    n=np.where(x_axes >= 0.47)
    Kew01_nii[n]=np.nan

    Ka03=0.61/(x_axes-0.05)+1.3
    n=np.where(x_axes >= 0.05)
    Ka03[n]=np.nan

    Scha07=1.05*x_axes+0.45
    n=np.where(Scha07 < Kew01_nii)
    Scha07[n]=np.nan
    return Kew01_nii, Ka03, Scha07
    
    
##########################################################################################################
##########################################################################################################

def SII_BPT(input, snr=3, Kewley01=False, mask=None):
    '''
    If using these diagnostic fuctions please ref the appropriate references given below.
    
    If using DESI please reference Summary_ref_2023 and the apprpriate
    emission line catalog (e.g. FastSpecFit ref FastSpecFit_ref)

    --SII diagram originally from VO87--
    
    By default here we use the Law+21 line for SF/AGN separation and the Kewley+06 line 
    for LINER/Seyfert separation on the AGN side. Optionally, can set Kewley01=True to 
    use the Kewley+01 line instead of Law+21
    
    Inputs:
    'table' including SII, Ha, OIII, Hb and inverse variances.
    'snr' is the snr cut applied to all axes. Default is 3.
    'Kewley01' provides and optional diagnostic line. Default is False.
    'mask' is an optional mask (e.g. from masked column array). Default is None.
        
    Outputs:
    Output vectors of same dimension as rows in table which include flags for:
    sii_bpt, sf_sii, agn_sii, liner_sii, quiescent_sii
 
    BPT regions defined as:
    #Kewley et al. 2001: starburst vs AGN classification. Solid lines in BPT
    Kew01_sii: log10(flux_oiii_5006/flux_hbeta)=0.72/(log10(flux_sii_6716,6731/flux_halpha)-0.32)+1.30
    #Law et al. 2021 proposed revised lines based on MaNGA observation:
    Law21_sii: log10(flux_oiii_5006/flux_hbeta)=0.648/(log10(flux_sii_6716,6731/flux_halpha)-0.324)+1.349

    Optional:
    #Kewley et al. 2006: Seyferts vs LINERS
    Kew06_sii: log10(flux_oiii_5006/flux_hbeta)=1.89*log10(flux_sii_6716,6731/flux_halpha)+0.76

    Other BPT regions not implemented here:
    #Law et al. define an extra "intermediate" region (not yet implemented)    
    '''
    # Mask for zero fluxes
    zero_flux_sii = (input['HALPHA_FLUX'] == 0) | (input['HBETA_FLUX'] == 0) | \
                    (input['OIII_5007_FLUX'] == 0) | \
                    (input['SII_6716_FLUX']+input['SII_6731_FLUX'] == 0)
    if mask != None:
        # Mask for flux avalibility - included as fastspecfit columns are maskedcolumn data
        mask = mask     
        zero_flux_sii = (input['HALPHA_FLUX'] == 0) | (input['HBETA_FLUX'] == 0) | \
                        (input['OIII_5007_FLUX'] == 0) | \
                        (input['SII_6716_FLUX']+input['SII_6731_FLUX'] == 0) | mask     

    #If ivar=0 set it to NaN to avoid infinites when computing the error:
    input['HALPHA_FLUX_IVAR']=np.where(input['HALPHA_FLUX_IVAR']==0,np.nan,input['HALPHA_FLUX_IVAR'])
    input['HBETA_FLUX_IVAR']=np.where(input['HBETA_FLUX_IVAR']==0,np.nan,input['HBETA_FLUX_IVAR'])
    input['OIII_5007_FLUX_IVAR']=np.where(input['OIII_5007_FLUX_IVAR']==0,np.nan,input['OIII_5007_FLUX_IVAR'])
    input['SII_6716_FLUX_IVAR']=np.where(input['SII_6716_FLUX_IVAR']==0,np.nan,input['SII_6716_FLUX_IVAR'])
    input['SII_6731_FLUX_IVAR']=np.where(input['SII_6731_FLUX_IVAR']==0,np.nan,input['SII_6731_FLUX_IVAR'])

    # Mask for SNR. Default is SII-BPT is available if all SNR >= 3
    snr = snr
    SNR_Ha=input['HALPHA_FLUX']*np.sqrt(input['HALPHA_FLUX_IVAR'])
    SNR_Hb=input['HBETA_FLUX']*np.sqrt(input['HBETA_FLUX_IVAR'])
    SNR_OIII=input['OIII_5007_FLUX']*np.sqrt(input['OIII_5007_FLUX_IVAR'])
    SNR_SII=(input['SII_6716_FLUX']+input['SII_6731_FLUX']) / \
            (1/np.sqrt(input['SII_6716_FLUX_IVAR'])+1/np.sqrt(input['SII_6731_FLUX_IVAR']))

    # Define regions    
    log_sii_ha=np.log10((input['SII_6716_FLUX']+input['SII_6731_FLUX'])/input['HALPHA_FLUX'])
    log_oiii_hb=np.log10(input['OIII_5007_FLUX']/input['HBETA_FLUX'])
    Kew01_sii=0.72/(log_sii_ha-0.32)+1.30
    Kew06_sii=1.89*log_sii_ha+0.76
    Law21_sii=0.648/(log_sii_ha-0.324)+1.43 #modified (+1.349 was original)
    if Kewley01=='True':
        line_sii = Kew01_sii
    else:
        line_sii = Law21_sii

    ## SII-BPT is available (All lines SNR >= 3)
    sii_bpt = (SNR_Ha >= snr) & (SNR_Hb >= snr) & (SNR_OIII >= snr) & (SNR_SII >= snr) & (~zero_flux_sii)

    ## SII-Quiescent -- (SNR < 3 for one or more lines)
    quiescent_sii = ((SNR_Ha < snr) | (SNR_Hb < snr) | (SNR_OIII < snr) | (SNR_SII < snr)) & (~zero_flux_sii)

    ## SII-AGN, LINER, SF
    agnliner_sii=(sii_bpt) & ((log_oiii_hb>=line_sii) | (log_sii_ha>=0.32))
    agn_sii=(agnliner_sii) & (log_oiii_hb>=Kew06_sii)
    liner_sii=(agnliner_sii) & (log_oiii_hb<Kew06_sii)
    sf_sii=(sii_bpt) & (~agnliner_sii)

    return (sii_bpt, sf_sii, agn_sii, liner_sii, quiescent_sii)

##########################################################################################################
##########################################################################################################

def OI_BPT(input, snr=3, snrOI=1, Kewley01=False, mask=None):
    '''
    If using these diagnostic fuctions please ref the appropriate references given below.
    
    If using DESI please reference Summary_ref_2023 and the apprpriate
    emission line catalog (e.g. FastSpecFit ref FastSpecFit_ref)

    --OI diagram originally from VO87--
    
    By default here we use the Law+21 line for SF/AGN separation and the Kewley+06 line 
    for LINER/Seyfert separation on the AGN side. Optionally, can set Kewley01=True to 
    use the Kewley+01 line instead of Law+21
    
    Inputs:
    'table' including OI, Ha, OIII, Hb and inverse variances.
    'snr' is the snr cut applied to Ha, Hb and OIII. Default is 3.
    'snrOI' is the snr cut applied to the [OI]6300 emission line. Default is 1. 
    'Kewley01' provides and optional diagnostic line. Default is False.
    'mask' is an optional mask (e.g. from masked column array). Default is None.
      
    Outputs:
    Output vectors of same dimension as rows in table which include flags for:
    oi_bpt, sf_oi, agn_oi, liner_oi
 
    BPT regions defined as:
    #Law et al. 2021: By default, use the Law+21 line for SF/AGN separation
    Law21_oi: log10(flux_oiii_5006/flux_hbeta)=0.884/(log10(flux_oi_6300/flux_halpha)+0.124)+1.291
    #Kewley et al. 2006: By default Kewley+06 line for LINER/Seyfert separation on the AGN side
    Kew06_oi: log10(flux_oiii_5006/flux_hbeta)=1.18*log10(flux_oi_6300/flux_halpha)+1.30

    Optional:
    Optionally, can set Kewley01=True to use the Kewley+01 line instead of Law+21
    #Kewley et al. 2001: starburst vs AGN classification. Solid lines in BPT
    Kew01_oi: log10(flux_oiii_5006/flux_hbeta)=0.73/(log10(flux_oi_6300/flux_halpha)+0.59)+1.33

    Other BPT regions not implemented here:
    #Law et al. define an extra "intermediate" region (not yet implemented)    
    '''    
    # Mask for zero fluxes
    zero_flux_oi = (input['HALPHA_FLUX'] == 0) | (input['HBETA_FLUX'] == 0) | \
                   (input['OIII_5007_FLUX'] == 0) | (input['OI_6300_FLUX'] == 0)    
    if mask != None:
        # Mask for flux avalibility - included as fastspecfit columns are maskedcolumn data
        mask = mask     
        zero_flux_oi = (input['HALPHA_FLUX'] == 0) | (input['HBETA_FLUX'] == 0) | \
                       (input['OIII_5007_FLUX'] == 0) | (input['OI_6300_FLUX'] == 0) | \
                       mask

    #If ivar=0 set it to NaN to avoid infinites when computing the error:
    input['HALPHA_FLUX_IVAR']=np.where(input['HALPHA_FLUX_IVAR']==0,np.nan,input['HALPHA_FLUX_IVAR'])
    input['HBETA_FLUX_IVAR']=np.where(input['HBETA_FLUX_IVAR']==0,np.nan,input['HBETA_FLUX_IVAR'])
    input['OIII_5007_FLUX_IVAR']=np.where(input['OIII_5007_FLUX_IVAR']==0,np.nan,input['OIII_5007_FLUX_IVAR'])
    input['OI_6300_FLUX_IVAR']=np.where(input['OI_6300_FLUX_IVAR']==0,np.nan,input['OI_6300_FLUX_IVAR'])

    # Mask for SNR. Default is OI-BPT is available if Ha, Hb, OIII SNR >= 3 and OI SNR >= 1.
    snr = snr
    snrOI=snrOI
    SNR_Ha=input['HALPHA_FLUX']*np.sqrt(input['HALPHA_FLUX_IVAR'])
    SNR_Hb=input['HBETA_FLUX']*np.sqrt(input['HBETA_FLUX_IVAR'])
    SNR_OIII=input['OIII_5007_FLUX']*np.sqrt(input['OIII_5007_FLUX_IVAR'])
    SNR_OI=input['OI_6300_FLUX']*np.sqrt(input['OI_6300_FLUX_IVAR'])
        
    # Define regions
    log_oi_ha=np.log10(input['OI_6300_FLUX']/input['HALPHA_FLUX'])
    log_oiii_hb=np.log10(input['OIII_5007_FLUX']/input['HBETA_FLUX'])
    Kew01_oi=0.73/(log_oi_ha+0.59)+1.33
    Kew06_oi=1.18*log_oi_ha+1.30
    Law21_oi=0.884/(log_oi_ha+0.124)+1.4   #modified (original was +1.291)
    if Kewley01=='True':
        line_oi = Kew01_oi
    else:
        line_oi = Law21_oi

    ## OI-BPT is available (SNR for the 3 lines other than OI >= 3)
    oi_bpt = (SNR_Ha >= snr) & (SNR_Hb >= snr) & (SNR_OIII >= snr) & (SNR_OI >= snrOI) & (~zero_flux_oi)

    ## OI-AGN, LINER, SF
    agnliner_oi=(oi_bpt) & ((log_oiii_hb>=line_oi) | (log_oi_ha>=-0.59))
    agn_oi = (agnliner_oi) & (log_oiii_hb>=Kew06_oi)
    liner_oi = (agnliner_oi) & (log_oiii_hb<Kew06_oi)
    sf_oi = oi_bpt & (~agnliner_oi) 

    return (oi_bpt, sf_oi, agn_oi, liner_oi)

##########################################################################################################
##########################################################################################################

def WHAN(input, snr=3, mask=None):
    '''
    If using these diagnostic fuctions please ref Mar_&_Steph_2023
    and the appropriate references given below.
    
    If using DESI please reference Summary_ref_2023 and the apprpriate
    emission line catalog (e.g. FastSpecFit ref FastSpecFit_ref)

    --original diagram WHAN diagram (Cid Fernandes et al. 2011)--
    
    Inputs:
    'input' including Ha, NII fluexes and inverse variances and the Ha equivanlent width.
    'snr' is the snr cut applied to all axes. Default is 3.
    'mask' is an optional mask (e.g. from masked column array). Default is None.
      
    Outputs:
    Output vectors of same dimension as rows in table which include flags for:
    whan, whan_sf, whan_sagn, whan_wagn, whan_retired, whan_passive
    
    WHAN regions defined as:
    # Cid Fernandes et al. 2011
    # pure star-forming galaxies
    whan_sf: log10(flux_nii_6583/flux_halpha) < -0.4 and ew_ha_6562 > 3A
    # strong AGN (e.g. Seyferts)
    whan_sagn: log10(flux_nii_6583/flux_halpha) > -0.4 and ew_ha_6562 > 6A
    #weak AGN: 
    whan_wagn: log10(flux_nii_6583/flux_halpha) > -0.4 and 3A < ew_ha_6562 < 6A
    #retired galaxies (fake AGN, i.e. galaxies that have stopped forming stars 
    # and are ionized by their hot low-mass evolved stars): 
    whan_retired: 0.5 A < ew_ha_6562 < 3A
    #passive: 
    whan_passive: ew_ha_6562 < 0.5A
    '''
    # Mask for zero fluxes
    zero_flux_whan= (input['HALPHA_FLUX']==0) | (input['NII_6584_FLUX']==0)
    if mask != None:
        # Mask for flux avalibility - included as fastspecfit columns are maskedcolumn data
        mask = mask     
        zero_flux_whan= (input['HALPHA_FLUX']==0) | (input['NII_6584_FLUX']==0) | mask    
        
    #If ivar=0 set it to NaN to avoid infinites when computing the error:
    input['HALPHA_FLUX_IVAR']=np.where(input['HALPHA_FLUX_IVAR']==0,np.nan,input['HALPHA_FLUX_IVAR'])
    input['NII_6584_FLUX_IVAR']=np.where(input['NII_6584_FLUX_IVAR']==0,np.nan,input['NII_6584_FLUX_IVAR'])

    # Mask for SNR. Default is OI-BPT is available if Ha, Hb, OIII SNR >= 3 and OI SNR >= 1.
    snr = snr
    SNR_Ha=input['HALPHA_FLUX']*np.sqrt(input['HALPHA_FLUX_IVAR'])
    SNR_NII=input['NII_6584_FLUX']*np.sqrt(input['NII_6584_FLUX_IVAR'])

    # Define regions
    ew_ha_6562=input['HALPHA_EW']
    log_nii_ha=np.log10(input['NII_6584_FLUX']/input['HALPHA_FLUX'])

    ## WHAN is available (NII and Halpha lines SNR >= 3)
    whan = (SNR_Ha >= snr) & (SNR_NII >= snr) & (~zero_flux_whan)

    ## WHAN-SF, strong AGN, weak AGN, retired, passive
    whan_sf=(log_nii_ha<-0.4) & (ew_ha_6562>=3)
    whan_sagn=(log_nii_ha>=-0.4) & (ew_ha_6562>=6)
    whan_wagn=(log_nii_ha>=-0.4) & (ew_ha_6562<6) & (ew_ha_6562>=3)
    whan_retired=(ew_ha_6562<3) & (ew_ha_6562>=0.5)
    whan_passive=ew_ha_6562<0.5

    return (whan, whan_sf, whan_sagn, whan_wagn, whan_retired, whan_passive)

##########################################################################################################
##########################################################################################################

def BLUE(input, snr=3, snrOII=1, mask=None):
    '''
    If using these diagnostic fuctions please ref the appropriate references given below.
    
    If using DESI please reference Summary_ref_2023 and the apprpriate
    emission line catalog (e.g. FastSpecFit ref FastSpecFit_ref)

    --BLUE diagram from Lamareille et al (2004) & Lamareille (2010)--
    
    Inputs:
    'input' including OII, OIII and Hb fluxes and inverse variances and the Hb equivanlent width.
    'snr' is the snr cut applied to Hb and OIII. Default is 3.
    'snrOII' is the cut applied to [OII]3727
    'mask' is an optional mask (e.g. from masked column array). Default is None.
      
    Outputs:
    Output vectors of same dimension as rows in table which include flags for:
    blue, agn_blue, sflin_blue, liner_blue, sf_blue, sfagn_blue
    
    Blue diagram regions defined as:
    #Main division between SF/AGN (eq. 1 of Lamareille 2010): 
    log10(flux_oiii_5006/flux_hbeta) = 0.11/(log10(ew_oii_3727/ew_hb_4861) - 0.92) + 0.85
    #Division between SF and "mixed" SF/Sy2 (eq. 2 of Lamareille 2010): 
    log10(flux_oiii_5006/flux_hbeta) > 0.3
    #Divisions for the SF-LIN/Comp overlap region (eq. 3 of Lamareille 2010): 
    blue1: log10(flux_oiii_5006/flux_hbeta) = -(log10(ew_oii_3727/ew_hb_4861)-1.0)**2 - 0.1*log10(ew_oii_3727/ew_hb_4861) + 0.25
    blue2: log10(flux_oiii_5006/flux_hbeta) = (log10(ew_oii_3727/ew_hb_4861)-0.2)**2 - 0.6
    #Division between Sy2/LINER (eq. 4 of Lamareille 2010): 
    log10(flux_oiii_5006/flux_hbeta) = 0.95*log10(ew_oii_3727/ew_hb_4861) - 0.4
    '''
    # Mask for zero fluxes
    zero_flux_blue= (input['HBETA_FLUX']==0) | (input['OIII_5007_FLUX']==0) | (input['OII_3726_FLUX']==0)
    if mask != None:
        # Mask for flux avalibility - included as fastspecfit columns are maskedcolumn data
        mask = mask     
        zero_flux_blue= (input['HBETA_FLUX']==0) | (input['OIII_5007_FLUX']==0) | \
                        (input['OII_3726_FLUX']==0) | mask

    #If ivar=0 set it to NaN to avoid infinites when computing the error:
    input['HBETA_FLUX_IVAR']=np.where(input['HBETA_FLUX_IVAR']==0,np.nan,input['HBETA_FLUX_IVAR'])
    input['OIII_5007_FLUX_IVAR']=np.where(input['OIII_5007_FLUX_IVAR']==0,np.nan,input['OIII_5007_FLUX_IVAR'])

    # Mask for SNR. Default is OI-BPT is available if Ha, Hb, OIII SNR >= 3 and OI SNR >= 1.
    snr = snr
    snrOII=snrOII
    SNR_Hb=input['HBETA_FLUX']*np.sqrt(input['HBETA_FLUX_IVAR'])
    SNR_OIII=input['OIII_5007_FLUX']*np.sqrt(input['OIII_5007_FLUX_IVAR'])
    SNR_OII=input['OII_3726_FLUX']*np.sqrt(input['OII_3726_FLUX_IVAR'])

    # Define regions
    log_ewoii_ewhb = np.log10(input['OII_3726_EW']/input['HBETA_EW'])
    log_oiii_hb=np.log10(input['OIII_5007_FLUX']/input['HBETA_FLUX'])
    main_blue = 0.11/(log_ewoii_ewhb-0.92)+0.85
    eq3_blue1 = -(log_ewoii_ewhb-1.0)**2-0.1*log_ewoii_ewhb+0.25
    eq3_blue2 = (log_ewoii_ewhb-0.2)**2-0.6
    eq4_blue = 0.95*log_ewoii_ewhb - 0.4

    ## BLUE is available (SNR for the 3 lines other than OII >= 3)
    blue = (SNR_Hb >= snr) & (SNR_OIII >= snr) & (SNR_OII >= snrOII) & (~zero_flux_blue)

    ## BLUE-AGN, SF/LINER/Composite, LINER, SF, SF/AGN
    agn_blue = blue & ((log_oiii_hb>=main_blue) | (log_ewoii_ewhb>=0.92))
    sflin_blue = blue & ((log_oiii_hb<=eq3_blue1) | (log_oiii_hb>=eq3_blue2))
    liner_blue = (agn_blue) & (log_oiii_hb<eq4_blue) & (~sflin_blue)
    sf_blue = blue & (~agn_blue) & (~liner_blue) & (~sflin_blue) & (log_oiii_hb<0.3)
    sfagn_blue = blue & (~agn_blue) & (~liner_blue) & (~sflin_blue) & (log_oiii_hb>=0.3)
    
    return (blue, agn_blue, sflin_blue, liner_blue, sf_blue, sfagn_blue)

##########################################################################################################
##########################################################################################################

def HeII_BPT(input, snr=3, mask=None):
    '''
    If using these diagnostic fuctions please ref Mar_&_Steph_2023
    and the appropriate references given below.
    
    If using DESI please reference Summary_ref_2023 and the apprpriate
    emission line catalog (e.g. FastSpecFit ref FastSpecFit_ref)

    --original diagram BPT DIAGRAM Shirazi & Brinchmann 2012--
    
    Inputs:
    'input' including Ha, Hb, HeII, NII fluxes and inverse variances.
    'snr' is the snr cut applied to all axes. Default is 3.
    'mask' is an optional mask (e.g. from masked column array). Default is None.
      
    Outputs:
    Output vectors of same dimension as rows in table which include flags for:
    heii_bpt, agn_heii, sf_nii
    
    Regions defined as:
    log10(flux_heii_4685/flux_hbeta)=-1.22+1/(8.92*log10(flux_nii_6583/flux_halpha)+1.32)
    '''
    # Mask for zero fluxes
    zero_flux_heii= (input['HALPHA_FLUX']==0) | (input['HBETA_FLUX']==0) | \
                    (input['HEII_4686_FLUX']==0) | (input['NII_6584_FLUX']==0)    
    if mask != None:
        # Mask for flux avalibility - included as fastspecfit columns are maskedcolumn data
        mask = mask
        zero_flux_heii= (input['HALPHA_FLUX']==0) | (input['HBETA_FLUX']==0) | \
                        (input['HEII_4686_FLUX']==0) | (input['NII_6584_FLUX']==0) | \
                        mask

    #If ivar=0 set it to NaN to avoid infinites when computing the error:
    input['HALPHA_FLUX_IVAR']=np.where(input['HALPHA_FLUX_IVAR']==0,np.nan,input['HALPHA_FLUX_IVAR'])
    input['HBETA_FLUX_IVAR']=np.where(input['HBETA_FLUX_IVAR']==0,np.nan,input['HBETA_FLUX_IVAR'])
    input['HEII_4686_FLUX_IVAR']=np.where(input['HEII_4686_FLUX_IVAR']==0,np.nan,input['HEII_4686_FLUX_IVAR'])
    input['NII_6584_FLUX_IVAR']=np.where(input['NII_6584_FLUX_IVAR']==0,np.nan,input['NII_6584_FLUX_IVAR'])

    # Mask for SNR. Default is OI-BPT is available if Ha, Hb, OIII SNR >= 3 and OI SNR >= 1.
    snr = snr
    SNR_Ha=input['HALPHA_FLUX']*np.sqrt(input['HALPHA_FLUX_IVAR'])
    SNR_Hb=input['HBETA_FLUX']*np.sqrt(input['HBETA_FLUX_IVAR'])
    SNR_HeII=input['HEII_4686_FLUX']*np.sqrt(input['HEII_4686_FLUX_IVAR'])
    SNR_NII=input['NII_6584_FLUX']*np.sqrt(input['NII_6584_FLUX_IVAR'])

    # Define regions
    log_nii_ha=np.log10(input['NII_6584_FLUX']/input['HALPHA_FLUX'])
    log_heii_hb=np.log10(input['HEII_4686_FLUX']/input['HBETA_FLUX'])
    Shir12=-1.22+1/(8.92*log_nii_ha+1.32)

    ## HeII-BPT is available (All lines SNR >= 3)
    heii_bpt = (SNR_Ha >= snr) & (SNR_Hb >= snr) & (SNR_HeII >= snr) & (SNR_NII >= snr) & (~zero_flux_heii)

    ## HeII-AGN, SF
    agn_heii=(heii_bpt) & (log_heii_hb>=Shir12)
    sf_heii=(heii_bpt) & ~agn_heii

    return (heii_bpt, agn_heii, sf_heii)

##########################################################################################################
##########################################################################################################

def NeV(input, snr=2.5, mask=None):
    '''
    If using these diagnostic fuctions please ref Mar_&_Steph_2023
    and the appropriate references given below.
    
    If using DESI please reference Summary_ref_2023 and the apprpriate
    emission line catalog (e.g. FastSpecFit ref FastSpecFit_ref)

    --NeV diagnostic--
    
    Inputs:
    'input' including NeV flux and inverse variance.
    'snr' is the snr cut applied to NeV. Default is 2.5.
    'mask' is an optional mask (e.g. from masked column array). Default is None.
      
    Outputs:
    Output vectors of same dimension as rows in table which include flags for:
    agn_nev, sf_nev
    
    Regions defined as:
    Detection of [NeV]λ3426 implies hard radiation with photon energies above 96.6 eV, indicating AGN
    '''
    # Mask for zero fluxes
    zero_flux_nev= input['NEV_3426_FLUX']==0
    if mask != None:
        # Mask for flux avalibility - included as fastspecfit columns are maskedcolumn data
        mask = mask
        zero_flux_nev= input['NEV_3426_FLUX']==0 | mask
    
    #If ivar=0 set it to NaN to avoid infinites when computing the error:
    input['NEV_3426_FLUX_IVAR']=np.where(input['NEV_3426_FLUX_IVAR']==0,np.nan,input['NEV_3426_FLUX_IVAR'])

    # Mask for SNR. Default is OI-BPT is available if Ha, Hb, OIII SNR >= 3 and OI SNR >= 1.
    snr = snr
    SNR_NeV=input['NEV_3426_FLUX']*np.sqrt(input['NEV_3426_FLUX_IVAR'])

    ## NeV diagnostic is available if flux is not zero
    nev = (~zero_flux_nev)

    ## NeV-AGN, SF
    agn_nev= (nev) & (SNR_NeV >= snr)
    sf_nev= (nev) & ~agn_nev

    return (nev, agn_nev, sf_nev)

##########################################################################################################
##########################################################################################################

def WISE_colors(input, snr=3, mask=None, diag='All'):
    '''
    If using these diagnostic fuctions please ref Mar_&_Steph_2025
    and the appropriate references given below.
    
    If using DESI please reference Summary_ref_2023 and the apprpriate
    photometry catalog (e.g., Tractor or Photometry VAC)

    --WISE Color diagnostic--
    
    Inputs:
    'input' including WISE fluxes and inverse variance: 
            FLUX_W1, FLUX_IVAR_W1, FLUX_W2, FLUX_IVAR_W2, FLUX_W3, FLUX_IVAR_W3
    'snr' is the snr cut applied to WISE magnitudes. Default is 3.
    'mask' is an optional mask (e.g. from masked column array). Default is None.
      
    Outputs:
    Output vectors of same dimension as rows in table which include flags for:
    W1W2_avail, agn_ir, sf_ir (unavail_ir)
    
    Regions defined as:
    Region defined in WISE infrared color space, indicating AGN. Note of caution:
    The points outside the AGN region may still include a significant fraction of AGN and 
    are best considered as "uncertain" rather than "star-forming" or "non-AGN"
    '''
    # Mask for zero fluxes
    zero_flux_wise = (input['FLUX_W1']==0)|(input['FLUX_W2']==0)
    zero_flux_w3 = (input['FLUX_W3']==0)
    if mask != None:
        # Mask for flux avalibility - included if input photometry is missing/masked
        mask = mask
        zero_flux_wise = (input['FLUX_W1']==0) | (input['FLUX_W2']==0) | mask
        zero_flux_w3 = (input['FLUX_W3']==0) | mask
    
    #If ivar=0 set it to NaN to avoid infinites when computing the error:
    input['FLUX_IVAR_W1'] = np.where(input['FLUX_IVAR_W1']==0, np.nan, input['FLUX_IVAR_W1'])
    input['FLUX_IVAR_W2'] = np.where(input['FLUX_IVAR_W2']==0, np.nan, input['FLUX_IVAR_W2'])
    input['FLUX_IVAR_W3'] = np.where(input['FLUX_IVAR_W3']==0, np.nan, input['FLUX_IVAR_W3'])

    # Mask for SNR.
    snr = snr
    SNR_W1 = input['FLUX_W1']*np.sqrt(input['FLUX_IVAR_W1'])
    SNR_W2 = input['FLUX_W2']*np.sqrt(input['FLUX_IVAR_W2'])
    SNR_W3 = input['FLUX_W3']*np.sqrt(input['FLUX_IVAR_W3'])

    ## IR diagnostic based on W1W2 is available if flux is not zero
    ## Note: Yao+2020 used S/N>5 for W1, W2 and S/N>2 for W3 due to being much less sensitive
    W1W2_avail = (~zero_flux_wise)&(SNR_W1>snr)&(SNR_W2>snr)
    W2W3_avail = (~zero_flux_wise)&(~zero_flux_w3)&(SNR_W2>snr)&(SNR_W3>snr)

    # Convert to AB magnitudes (most diagnostics use Vega mags so need to apply offsets)
    W1 = 22.5 - 2.5*np.log10(input['FLUX_W1'])
    W2 = 22.5 - 2.5*np.log10(input['FLUX_W2'])
    W3 = 22.5 - 2.5*np.log10(input['FLUX_W3'])
    W1W2 = W1 - W2
    W2W3 = W2 - W3

    # Offsets from Vega to AB magnitudes (Jarrett+2011) 
    W1_vega2ab = 2.699
    W2_vega2ab = 3.339
    W3_vega2ab = 5.174

    # Offsets from Vega to AB WISE colors
    W1W2_vega2ab = W1_vega2ab - W2_vega2ab
    W2W3_vega2ab = W2_vega2ab - W3_vega2ab

    # Subtract offsets to go from AB to Vega (add to go from Vega to AB)
    W1_Vega = W1 - W1_vega2ab
    W2_Vega = W2 - W2_vega2ab
    W3_Vega = W3 - W3_vega2ab
    W1W2_Vega = W1_Vega - W2_Vega  #W1W2 - W1W2_vega2ab
    W2W3_Vega = W2_Vega - W3_Vega  #W2W3 - W2W3_vega2ab
    
    ## Jarrett et al. 2011 box in W1-W2 vs. W2-W3 space in Vega mags
    y_top = 1.7 
    y_bot = 0.1*W2W3_Vega + 0.38
    x_left = 2.2
    x_right = 4.2
    
    agn_jarrett11 = W1W2_avail&W2W3_avail&(W2W3_Vega>x_left)&(W2W3_Vega<x_right)&(W1W2_Vega>y_bot)&(W1W2_Vega<y_top)
    sf_jarrett11 = W1W2_avail&W2W3_avail&(~agn_jarrett11)
    unavail_jarrett11 = (~W1W2_avail)|(~W2W3_avail)  #unavailable
                    
    ## Stern et al. 2012 cut along just W1-W2 color
    agn_stern12 = W1W2_avail&(W1W2_Vega>0.8)
    sf_stern12 = W1W2_avail&(~agn_stern12)
    unavail_stern12 = ~W1W2_avail  #unavailable

    ## Mateos et al. 2012 box in W1-W2 vs. W2-W3 space
    x_M12 = W2W3 / (2.5)  #from eqn 1 using AB mags
    y_M12 = W1W2 / (2.5)
        
    # top/bottom around the power-law
    y_top = 0.315*x_M12 + 0.297   #eqn 1 + offset
    y_bot = 0.315*x_M12 - 0.110   #eqn 1 - offset
    y_pl = -3.172*x_M12 + 0.436   #eqn 2 for the power-law

    agn_mateos12 = W1W2_avail&W2W3_avail&(y_M12>y_bot)&(y_M12>y_pl)&(y_M12<y_top)
    sf_mateos12 = W1W2_avail&W2W3_avail&(~agn_mateos12)
    unavail_mateos12 = (~W1W2_avail)|(~W2W3_avail)  #unavailable
        
    ## Assef et al. 2018: https://ui.adsabs.harvard.edu/abs/2018ApJS..234...23A/abstract
    
    # equation 2 (simplistic from Stern+12): (W1W2_Vega >= 0.8)&((W2 - W2_vega2ab)<15.05)
    # equation 3: W1W2_Vega > alpha* exp(beta*(W2_Vega-gamma)**2)
                    
    ## 90% reliability
    alpha_90 = 0.65
    beta_90 = 0.153
    gamma_90 = 13.86        
                    
    ## 75% reliability
    alpha_75 = 0.486
    beta_75 = 0.092
    gamma_75 = 13.07 
     
    ## Choose here:
    alpha = alpha_90
    beta = beta_90
    gamma = gamma_90
    
    bright_a18 = W2_Vega<=gamma
                    
    agn_assef18 = W1W2_avail&((W1W2_Vega > alpha* np.exp(beta*(W2_Vega-gamma)**2))|
                              ((W1W2_Vega > alpha)&bright_a18))
    sf_assef18 = W1W2_avail&(~agn_assef18)
    unavail_assef18 = ~W1W2_avail  #unavailable

    ## Yao et al. 2020 cuts
    # Vega mags: w1w2 = (0.015 * exp(w2w3/1.38)) - 0.08 + offset
    # where offset of 0.3 is reported in paper as the 2*sigma cut to create a demarcation line
    line_yao20 = (0.015 * np.exp(W2W3_Vega/1.38)) - 0.08 + 0.3
    agn_yao20 = W1W2_avail&W2W3_avail&(W1W2_Vega>line_yao20)
    unavail_yao20 = (~W1W2_avail)|(~W2W3_avail)  #unavailable

    ## Hviding et al. 2022 cuts in (y=)W1-W2 vs. (x=)W2-W3 space in Vega mags (eq. 3)
    x_left = 1.734
    x_right = 3.916
    y_bot1 = 0.0771*W2W3_Vega + 0.319
    y_bot2 = 0.261*W2W3_Vega -0.260
    agn_hviding22 = W1W2_avail&W2W3_avail&(W2W3_Vega>x_left)&(W2W3_Vega<x_right)&(W1W2_Vega>y_bot1)&(W1W2_Vega>y_bot2)
    unavail_hviding22 = (~W1W2_avail)|(~W2W3_avail)  #unavailable
    
    ## Set the choice here for individual diagnostics or our default combination # agn_hviding22 not yet implemented
    if diag=='Stern12':
        agn_ir = agn_stern12
        avail_ir = W1W2_avail
    if diag=='Assef18':
        agn_ir = agn_assef18
        avail_ir = W1W2_avail
    if diag=='Jarrett11':
        agn_ir = agn_jarrett11
        avail_ir = W1W2_avail&W2W3_avail
    if diag=='Mateos12':
        agn_ir = agn_mateos12
        avail_ir = W1W2_avail&W2W3_avail
    if diag=='Yao20':
        agn_ir = agn_yao20
        avail_ir = W1W2_avail&W2W3_avail
    if diag=='Hviding22':
        agn_ir = agn_hviding22
        avail_ir = W1W2_avail&W2W3_avail
    ## By default, combine the diagnostics based on W1W2W3 when all 3 bands available;
    #  otherwise use the Stern cut on W1-W2 only
    if diag=='All':
        agn_ir = agn_mateos12 | agn_jarrett11 | (agn_stern12&~W2W3_avail) | agn_assef18 | agn_hviding22
        avail_ir = W1W2_avail
    
    # SF defined based on the above
    sf_ir = avail_ir & (~agn_ir)
    
    return (avail_ir, agn_ir, sf_ir)

##########################################################################################################
##########################################################################################################

def Xray(input, H0=67.4, Om0=0.315, snr=3):
	## X-ray diagnostic ##
	#2-10 keV X-ray luminosity equal or above 1e42 erg/s indicates AGN	
	thres=1e42

	#Fiducial Cosmology used in DESI from Planck 2018 results: https://ui.adsabs.harvard.edu/abs/2020A%26A...641A...6P/abstract
	from astropy.cosmology import FlatLambdaCDM
	cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
	DL  = cosmo.luminosity_distance(input['z'].values) #in Mpc
	DL_cm = 3.08567758e24*DL.value

	#Convert the CSC 2-7 keV flux to 2-10 keV and compute the LX2-10 keV
	#Conversion factor =  1.334E-15 erg cm^-2 s^-1 for an unabsorbed flux of 1E-15 erg cm^-2 s^-1 using PIMMS (https://cxc.harvard.edu/toolkit/pimms.jsp) and assuming gamma=1.8
	factor =     1.334E-15 #for gamma=1.8
	flux_2_10=(input['FLUX_2_7']/1E-15)*factor
	LX210= 4*pi*DL_cm**2*flux_2_10  #in erg/s

	#Apply K-correction:
	gamma=1.8
	k = (1+input['z'])**(gamma-2)
	LX210_Kcorr=k*LX210

	#Mask for zero flux
	zero_flux_xray= input['FLUX_2_7']==0

	#Mask for SNR
	snr=snr
	SNR_Xray=input['FLUX_2_7']/input['FLUX_2_7_err']

	## Xray diagnostic is available if flux is not zero and SNR_Xray >= 3
	xray = (SNR_Xray >= snr) & (~zero_flux_xray)

	## Xray-AGN, SF, footprint
	agn_xray= (xray) & (LX210_Kcorr >= thres)
	sf_xray= (xray) & ~agn_xray
	fp_xray = (zero_flux_xray) & (input['FLUX_2_7_err'] > 0) 

	return (agn_xray, sf_xray, fp_xray)
