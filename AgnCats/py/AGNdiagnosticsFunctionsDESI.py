import numpy as np

'''notes for us:
Find/replace: Mar_&_Steph_2025 with correct reference 
Find/replace: Summary_ref_2025 with correct reference
Find/replace: FastSpecFit_ref with correct reference
'''

##########################################################################################################
##########################################################################################################

def BROAD_LINE(input, snr=3, mask=None, vel_thres=1200.):
    '''
    If using these diagnostic fuctions please ref the appropriate references given below.
    
    If using DESI please reference Summary_ref_2023 and the apprpriate
    emission line catalog (e.g. FastSpecFit ref FastSpecFit_ref)

    Inputs:
    'table' including Ha, Hb, MgII and CIV emission lines (fluxes and widths)
    'snr' is the snr cut applied to all axes. Default is 3.
    'mask' is an optional mask (e.g. from masked column array). Default is None.

    Outputs:
    Vectors of same dimension as rows in table which include flags for:
    broad_line
    
    If (FWHM>=1200 km/s in Halpha, Hbeta, MgII and/or CIV line)
    '''
    
    # Mask for zero fluxes when NONE of the lines are available
    zero_flux = (input['HALPHA_BROAD_FLUX']==0) & (input['HBETA_BROAD_FLUX']==0) & \
                (input['MGII_2796_FLUX']==0) & (input['MGII_2803_FLUX']==0) & (input['CIV_1549_FLUX']==0)
    if mask != None:
        # Mask for flux avalibility - included as fastspecfit columns are maskedcolumn data
        mask = mask     
        zero_flux = mask | ((input['HALPHA_BROAD_FLUX']==0) & (input['HBETA_BROAD_FLUX']==0) & \
                     (input['MGII_2796_FLUX']==0) & (input['MGII_2803_FLUX']==0) & (input['CIV_1549_FLUX']==0))

    #If ivar=0 set it to NaN to avoid infinites when computing the error:
    input['HALPHA_BROAD_FLUX_IVAR']=np.where(input['HALPHA_BROAD_FLUX_IVAR']==0,np.nan,input['HALPHA_BROAD_FLUX_IVAR'])
    input['HBETA_BROAD_FLUX_IVAR']=np.where(input['HBETA_BROAD_FLUX_IVAR']==0,np.nan,input['HBETA_BROAD_FLUX_IVAR'])
    input['MGII_2796_FLUX_IVAR']=np.where(input['MGII_2796_FLUX_IVAR']==0,np.nan,input['MGII_2796_FLUX_IVAR'])
    input['MGII_2803_FLUX_IVAR']=np.where(input['MGII_2803_FLUX_IVAR']==0,np.nan,input['MGII_2803_FLUX_IVAR'])
    input['CIV_1549_FLUX_IVAR']=np.where(input['CIV_1549_FLUX_IVAR']==0,np.nan,input['CIV_1549_FLUX_IVAR'])

    # Mask for SNR. Default is TYPE is available to determine if one of the lines SNR >= 3
    snr = snr

    # Broad components for Balmer lines
    SNR_Ha=input['HALPHA_BROAD_FLUX']*np.sqrt(input['HALPHA_BROAD_FLUX_IVAR'])
    SNR_Hb=input['HBETA_BROAD_FLUX']*np.sqrt(input['HBETA_BROAD_FLUX_IVAR'])

    # For MgII, sum the doublet
    MGII_FLUX = input['MGII_2796_FLUX']+input['MGII_2803_FLUX']
    MGII_FLUX_IVAR = 1./(1./input['MGII_2796_FLUX_IVAR'] + 1./input['MGII_2803_FLUX_IVAR'])
    SNR_MGII = MGII_FLUX*np.sqrt(MGII_FLUX_IVAR)
    
    # CIV
    SNR_CIV = input['CIV_1549_FLUX']*np.sqrt(input['CIV_1549_FLUX_IVAR'])

    # Factor to convert from Gaussian sigma to FWHM
    sig2fwhm = 2. * np.sqrt(2. * np.log(2.))
    
    # Define breadth in FWHM in km/s
    broad_fwhm_HALPHA = input['HALPHA_BROAD_SIGMA'] * sig2fwhm
    broad_fwhm_HBETA = input['HBETA_BROAD_SIGMA'] * sig2fwhm
    broad_fwhm_MGII_2796 = input['MGII_2796_SIGMA'] * sig2fwhm
    broad_fwhm_MGII_2803 = input['MGII_2803_SIGMA'] * sig2fwhm
    broad_fwhm_CIV = input['CIV_1549_SIGMA'] * sig2fwhm
        
    # Velocity threshold for FWHM in km/s to identify a BL (FWHM >= 1200 km/s by default)
    
    # Check for each line separately first
    is_broad_Ha = (SNR_Ha>=snr) & (broad_fwhm_HALPHA>=vel_thres) & (~zero_flux)
    is_broad_Hb = (SNR_Hb>=snr) & (broad_fwhm_HBETA>=vel_thres) & (~zero_flux)
    is_broad_MgII = (SNR_MGII>=snr) & (broad_fwhm_MGII_2796>=vel_thres) & (~zero_flux)
    is_broad_CIV = (SNR_CIV>=snr) & (broad_fwhm_CIV>=vel_thres) & (~zero_flux)
    
    # Decision: flag a BL if any of the 4 lines meet the criteria
    is_broad = is_broad_Ha | is_broad_Hb | is_broad_MgII | is_broad_CIV
    
    return (is_broad)

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

    ## NII-AGN, LINER, COMP, SF
    agnliner_nii=(nii_bpt) & ((log_oiii_hb>=Kew01_nii) | (log_nii_ha>=0.47))
    agn_nii=(agnliner_nii) & (log_oiii_hb>=Scha07) 
    liner_nii=(agnliner_nii) & (log_oiii_hb<Scha07) 
    composite_nii=(nii_bpt) & ((log_oiii_hb>=Ka03) | (log_nii_ha>=0.05)) & (~agnliner_nii)
    sf_nii=(nii_bpt) & (~agnliner_nii) & (~composite_nii)
    
    return (nii_bpt, sf_nii, agn_nii, liner_nii, composite_nii)
    
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
    SII_FLUX = input['SII_6716_FLUX'] + input['SII_6731_FLUX']
    SII_FLUX_IVAR = 1/ (1/input['SII_6716_FLUX_IVAR'] + 1/input['SII_6731_FLUX_IVAR'])

    # Mask for SNR. Default is SII-BPT is available if all SNR >= 3
    snr = snr
    SNR_Ha=input['HALPHA_FLUX']*np.sqrt(input['HALPHA_FLUX_IVAR'])
    SNR_Hb=input['HBETA_FLUX']*np.sqrt(input['HBETA_FLUX_IVAR'])
    SNR_OIII=input['OIII_5007_FLUX']*np.sqrt(input['OIII_5007_FLUX_IVAR'])
    SNR_SII = SII_FLUX*np.sqrt(SII_FLUX_IVAR)

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

    ## SII-AGN, LINER, SF
    agnliner_sii=(sii_bpt) & ((log_oiii_hb>=line_sii) | (log_sii_ha>=0.32))
    agn_sii=(agnliner_sii) & (log_oiii_hb>=Kew06_sii)
    liner_sii=(agnliner_sii) & (log_oiii_hb<Kew06_sii)
    sf_sii=(sii_bpt) & (~agnliner_sii)

    return (sii_bpt, sf_sii, agn_sii, liner_sii)

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

def WHAN(input, snr=3, snr_ew=1, mask=None):
    '''
    If using these diagnostic fuctions please ref Mar_&_Steph_2025
    and the appropriate references given below.
    
    If using DESI please reference Summary_ref_2025 and the apprpriate
    emission line catalog (e.g. FastSpecFit ref FastSpecFit_ref)

    --original diagram WHAN diagram (Cid Fernandes et al. 2011)--
    
    Inputs:
    'input' including Ha, NII fluxes, Ha equivalent width and inverse variances.
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

    # Mask for SNR. Default is WHAN is available if Ha, NII SNR >= 3.
    snr = snr
    SNR_Ha=input['HALPHA_FLUX']*np.sqrt(input['HALPHA_FLUX_IVAR'])
    SNR_NII=input['NII_6584_FLUX']*np.sqrt(input['NII_6584_FLUX_IVAR'])
    SNR_HaEW=input['HALPHA_EW']*np.sqrt(input['HALPHA_EW_IVAR'])

    # Define regions
    ew_ha_6562=input['HALPHA_EW']
    log_nii_ha=np.log10(input['NII_6584_FLUX']/input['HALPHA_FLUX'])

    ## WHAN is available: 
    # - NII and Halpha line flux SNR >= snr (=3 by default) when using the [NII]/Ha ratio
    # - Halpha EW measured at > snr_ew (=1 by default) sigma significance when cutting just on EW
    whan_ew_cut = (SNR_HaEW >= snr_ew) & (~zero_flux_whan)
    whan_flux_cut = (SNR_Ha >= snr) & (SNR_NII >= snr) & (~zero_flux_whan)
    whan = whan_ew_cut | whan_flux_cut

    ## WHAN-SF, strong AGN, weak AGN, retired, passive
    whan_sf=whan_flux_cut & (log_nii_ha<-0.4) & (ew_ha_6562>=3)
    whan_sagn=whan_flux_cut & (log_nii_ha>=-0.4) & (ew_ha_6562>=6)
    whan_wagn=whan_flux_cut & (log_nii_ha>=-0.4) & (ew_ha_6562<6) & (ew_ha_6562>=3)
    whan_retired=whan_ew_cut & (ew_ha_6562<3) & (ew_ha_6562>=0.5)
    whan_passive=whan_ew_cut & ew_ha_6562<0.5

    return (whan, whan_sf, whan_sagn, whan_wagn, whan_retired, whan_passive)

##########################################################################################################
##########################################################################################################

def BLUE(input, snr=3, snrOII=3, mask=None):
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

    ## SJ: DO WE NEED THIS?? We only take sqrt(IVAR) so a zero is fine (it gives SNR=0)
    #If ivar=0 set it to NaN to avoid infinites when computing the error:
#    input['HBETA_FLUX_IVAR']=np.where(input['HBETA_FLUX_IVAR']==0,np.nan,input['HBETA_FLUX_IVAR'])
#    input['OIII_5007_FLUX_IVAR']=np.where(input['OIII_5007_FLUX_IVAR']==0,np.nan,input['OIII_5007_FLUX_IVAR'])

    # Mask for SNR. Default is BLUE is available if Hb, OIII SNR >= 3 and OII SNR >= 1.
    snr = snr
    snrOII=snrOII
    SNR_Hb=input['HBETA_FLUX']*np.sqrt(input['HBETA_FLUX_IVAR'])
    SNR_OIII=input['OIII_5007_FLUX']*np.sqrt(input['OIII_5007_FLUX_IVAR'])
    SNR_Hb_EW = input['HBETA_EW']*np.sqrt(input['HBETA_EW_IVAR'])

    # [OII]3727 is the sum of the doublet [OII]3726,3729
    OII_EW = input['OII_3726_EW']+input['OII_3729_EW']
    OII_EW_IVAR = 1./ (1./input['OII_3726_EW_IVAR'] + 1./input['OII_3729_EW_IVAR'])
    SNR_OII_EW = OII_EW*np.sqrt(OII_EW_IVAR)
    
    # Parameters for horizontal and vertical axes
    log_ewoii_ewhb = np.log10(OII_EW/input['HBETA_EW'])
    log_oiii_hb=np.log10(input['OIII_5007_FLUX']/input['HBETA_FLUX'])
    
    # Define regions
    main_blue = 0.11/(log_ewoii_ewhb-0.92)+0.85

    # Mixed region
    eq3_blue1 = -(log_ewoii_ewhb-1.0)**2-0.1*log_ewoii_ewhb+0.25
    eq3_blue2 = (log_ewoii_ewhb-0.2)**2-0.6
    
    # Seyfert/LINER
    eq4_blue = 0.95*log_ewoii_ewhb - 0.4

    ## BLUE is available (SNR for the 3 lines other than OII >= 3)
    blue = (SNR_Hb >= snr) & (SNR_OIII >= snr) & (SNR_Hb_EW >= snr) & (SNR_OII_EW >= snrOII) & (~zero_flux_blue)

    ## BLUE-AGN, SF/LINER/Composite, LINER, SF, SF/AGN
    # Region that overlaps with other classes (set an extra bit for info)
    sflin_blue = blue & ((log_oiii_hb<=eq3_blue1) & (log_oiii_hb>=eq3_blue2))

    # AGN will be sub-divided between Seyfert2 & LINER
    agnlin_blue = blue & ((log_oiii_hb>=main_blue) | (log_ewoii_ewhb>=0.92))
    agn_blue = agnlin_blue & (log_oiii_hb>=eq4_blue)
    liner_blue = agnlin_blue & (log_oiii_hb<eq4_blue)

    # SF 
    sf_blue = blue & (~agnlin_blue) & (log_oiii_hb<0.3)
    sfagn_blue = blue & (~agnlin_blue) & (log_oiii_hb>=0.3) 
    
    return (blue, agn_blue, sflin_blue, liner_blue, sf_blue, sfagn_blue)

##########################################################################################################
##########################################################################################################

def MEX(input, snr=3, mask=None):
    '''
    MEx diagnostic diagram (Juneau et al. 2014)
    
    Inputs:
    'input' including OIII and Hb fluxes and inverse variances and stellar mass (Chabrier or Kroupa IMF).
    'snr' is the snr cut applied to Hb and OIII. Default is 3.
    'mask' is an optional mask (e.g. from masked column array). Default is None.
      
    Outputs:
    Output vectors of same dimension as rows in table which include flags for:
    'mex', 'mex_sf', 'mex_interm', 'mex_agn'
    
    MEx diagram regions defined as:
    #Top division between SF/AGN (eq. 1 of Juneau et al. 2014): 
    log10(flux_oiii_5006/flux_hbeta) = 0.375/(log10(M*) - 10.5) + 1.14 for logM*<=10
    #Division between SF and "intermediate" (eq. 2 of juneau et al. 2014): 
    log10(flux_oiii_5006/flux_hbeta) > a0+a1*x+a2*x**2+a3*x**3
    
    where x = log10(M*)
    '''
    
    # Mask for zero fluxes
    zero_flux_mex = (input['HBETA_FLUX'] == 0) | (input['OIII_5007_FLUX'] == 0)
    if mask != None:
        # Mask for flux avalibility - included as fastspecfit columns are maskedcolumn data
        mask = mask     
        zero_flux_mex = (input['HBETA_FLUX'] == 0) | (input['OIII_5007_FLUX'] == 0) | mask
   
    #If ivar=0 set it to NaN to avoid infinites when computing the error:
    input['HBETA_FLUX_IVAR']=np.where(input['HBETA_FLUX_IVAR']==0,np.nan,input['HBETA_FLUX_IVAR'])
    input['OIII_5007_FLUX_IVAR']=np.where(input['OIII_5007_FLUX_IVAR']==0,np.nan,input['OIII_5007_FLUX_IVAR'])

    # Mask for SNR. Default is MEx is available if all SNR >= 3
    snr = snr
    SNR_Hb=input['HBETA_FLUX']*np.sqrt(input['HBETA_FLUX_IVAR'])
    SNR_OIII=input['OIII_5007_FLUX']*np.sqrt(input['OIII_5007_FLUX_IVAR'])
    
    ## MEx is available (line fluxes SNR >= 3 and valid mass)
    mex = (SNR_Hb >= snr) & (SNR_OIII >= snr) & (input['LOGMSTAR']>4.) & (~zero_flux_mex)
    
    # Define variables for equations 1 & 2
    x = input['LOGMSTAR']
    y = np.log10(input['OIII_5007_FLUX']/input['HBETA_FLUX'])
    
    # upper MEx
    a0, a1, a2, a3 = 410.24, -109.333, 9.71731, -0.288244
    mex_agn = ((y>0.375/(x-10.5)+1.14)&(x<=10)) | ((y>a0+a1*x+a2*x**2+a3*x**3)&(x>10))
        
    # lower MEx
    a0, a1, a2, a3 = 352.066, -93.8249, 8.32651, -0.246416
    mex_sf = ((y<0.375/(x-10.5)+1.14)&(x<=9.6)) | ((y<a0+a1*x+a2*x**2+a3*x**3)&(x>9.6))
        
    # MEX intermediate
    mex_interm = (x>9.6)&(y>=a0+a1*x+a2*x**2+a3*x**3)&(~mex_agn)
    
    # Return whether it's available and then the 3 classes when also available
    return (mex, mex&mex_agn, mex&mex_sf, mex&mex_interm)
    
##########################################################################################################
##########################################################################################################

def KEX(input, snr=3, mask=None):
    '''
    KEx diagnostic diagram (Zhang & Hao 2018)
    
    Inputs:
    'input' including OIII and Hb fluxes and OIII width.
    'snr' is the snr cut applied to Hb and OIII. Default is 3.
    'mask' is an optional mask (e.g. from masked column array). Default is None.
      
    Outputs:
    Output vectors of same dimension as rows in table which include flags for:
    'kex', 'kex_sf', 'kex_interm', 'kex_agn'
    
    KEx diagram regions defined as:
    #Main division between SF/AGN (eq. 1 of Zhang & Hao 2018): 
    log10(flux_oiii_5006/flux_hbeta) = -2*sigma_oiii + 4.2
    
    #Division between SF and "intermediate" (eq. 2 of Zhang & Hao 2018): 
    log10(flux_oiii_5006/flux_hbeta) = 3
    '''
    
    # Masks:
    if mask is None:
        mask = np.zeros_like(input['HBETA_FLUX'], dtype=bool)
    
    # Mask zero fluxes
    zero_flux_kex = (input['HBETA_FLUX'] <= 0.) | (input['OIII_5007_FLUX'] <= 0.) | mask    # Mask for zero fluxes
   
    #If ivar=0 set it to NaN to avoid infinites when computing the error:
    input['HBETA_FLUX_IVAR']=np.where(input['HBETA_FLUX_IVAR']==0,np.nan,input['HBETA_FLUX_IVAR'])
    input['OIII_5007_FLUX_IVAR']=np.where(input['OIII_5007_FLUX_IVAR']==0,np.nan,input['OIII_5007_FLUX_IVAR'])

    # Mask for SNR. Default is KEx is available if all SNR >= 3
    snr = snr
    SNR_Hb=input['HBETA_FLUX']*np.sqrt(input['HBETA_FLUX_IVAR'])
    SNR_OIII=input['OIII_5007_FLUX']*np.sqrt(input['OIII_5007_FLUX_IVAR'])
    
    ## KEx is available (line fluxes SNR >= 3 and valid OIII width)
    kex = (SNR_Hb >= snr) & (SNR_OIII >= snr) & (input['OIII_5007_SIGMA']>0) & (~zero_flux_kex)
    
    # Define variables for equations 1 & 2
    x = np.log10(input['OIII_5007_SIGMA'])
    y = np.log10(input['OIII_5007_FLUX']/input['HBETA_FLUX'])
    
    # Upper KEX
    kex_agn=(y>=-2.*x + 4.2) & (y>=0.3)
  
    # Lower KEX
    kex_sf=y < -2.*x + 4.2

    # KEX intermediate
    kex_interm= (y>=-2.*x+4.2) & (y<0.3) & (~kex_agn)
    
    # Return whether it's available and then the 3 classes when also available
    return (kex, kex&kex_agn, kex&kex_sf, kex&kex_interm)
    
##########################################################################################################
##########################################################################################################

def HeII_BPT(input, snr=3, mask=None):
    '''
    If using these diagnostic fuctions please ref Mar_&_Steph_2025
    and the appropriate references given below.
    
    If using DESI please reference Summary_ref_2025 and the apprpriate
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

    # Mask for SNR. Default is HeII-BPT is available if Ha, Hb, NII, HeII SNR >= 3
    snr = snr
    SNR_Ha=input['HALPHA_FLUX']*np.sqrt(input['HALPHA_FLUX_IVAR'])
    SNR_Hb=input['HBETA_FLUX']*np.sqrt(input['HBETA_FLUX_IVAR'])
    SNR_HeII=input['HEII_4686_FLUX']*np.sqrt(input['HEII_4686_FLUX_IVAR'])
    SNR_NII=input['NII_6584_FLUX']*np.sqrt(input['NII_6584_FLUX_IVAR'])

    # Define regions
    log_nii_ha = np.log10(input['NII_6584_FLUX']/input['HALPHA_FLUX'])
    log_heii_hb = np.log10(input['HEII_4686_FLUX']/input['HBETA_FLUX'])
    Shir12 = -1.22+1/(8.92*log_nii_ha+1.32)

    # Value where denominator goes to zero (non-finite)
    log_nii_ha_0 = -1.32/8.92

    ## HeII-BPT is available (All lines SNR >= 3)
    heii_bpt = (SNR_Ha >= snr) & (SNR_Hb >= snr) & (SNR_HeII >= snr) & (SNR_NII >= snr) & (~zero_flux_heii)

    ## HeII-AGN, SF
    agn_heii=(heii_bpt) & ((log_heii_hb>=Shir12) | (log_nii_ha>=log_nii_ha_0))
    sf_heii=(heii_bpt) & ~agn_heii

    return (heii_bpt, agn_heii, sf_heii)

##########################################################################################################
##########################################################################################################

def NeV(input, snr=2.5, mask=None):
    '''
    If using these diagnostic fuctions please ref Mar_&_Steph_2025
    and the appropriate references given below.
    
    If using DESI please reference Summary_ref_2025 and the apprpriate
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

def WISE_colors(input, snr=3, mask=None, diag='All', weak_agn=False):
    '''
    If using these diagnostic fuctions please ref Mar_&_Steph_2025
    and the appropriate references given below.
    
    If using DESI please reference Summary_ref_2025 and the apprpriate
    photometry catalog (e.g., Tractor or Photometry VAC)

    --WISE Color diagnostic--
    
    Inputs:
    'input' including WISE fluxes and inverse variance: 
            FLUX_W1, FLUX_IVAR_W1, FLUX_W2, FLUX_IVAR_W2, FLUX_W3, FLUX_IVAR_W3
    'snr' is the snr cut applied to WISE magnitudes. Default is 3.
    'mask' is an optional mask (e.g. from masked column array). Default is None.
    'weak_agn' to optionally return weak (low-power) AGN from Yao+20 (only works for Yao20 or All)
      
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
    strong_agn_yao20 = W1W2_avail&W2W3_avail&(agn_jarrett11|agn_stern12)
    # Line for low-power AGN
    weak_agn_yao20 = W1W2_avail&W2W3_avail&(W1W2_Vega>line_yao20)&~strong_agn_yao20
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
        agn_ir = strong_agn_yao20
        wagn_ir = weak_agn_yao20  # not used for now (would need code changes)
        avail_ir = W1W2_avail&W2W3_avail
    if diag=='Hviding22':
        agn_ir = agn_hviding22
        avail_ir = W1W2_avail&W2W3_avail
    ## By default, combine the diagnostics based on W1W2W3 when all 3 bands available;
    #  otherwise use the Stern cut on W1-W2 only
    if diag=='All':
        agn_ir = agn_mateos12 | agn_jarrett11 | (agn_stern12&~W2W3_avail) | agn_assef18 | agn_hviding22
        avail_ir = W1W2_avail
        # By default, not considering weak (low-power) AGN; only return if specified
        wagn_ir = weak_agn_yao20 & ~agn_ir
     
    # SF defined based on the above
    sf_ir = avail_ir & (~agn_ir)
    
    # By default, not considering weak (low-power) AGN from Yao+20; only return if specified
    if weak_agn==False:
        return (avail_ir, agn_ir, sf_ir)
    else:
        return (avail_ir, agn_ir, sf_ir&~wagn_ir, wagn_ir)

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
