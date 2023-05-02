import numpy as np

'''notes for us:
Find/replace: Mar_&_Steph_2023 with correct reference 
Find/replace: Summary_ref_2023 with correct reference
Find/replace: FastSpecFit_ref with correct reference
'''

##########################################################################################################
##########################################################################################################
def NII_BPT(input, snr=3, mask=None):
    '''
    If using these diagnostic fuctions please ref Mar_&_Steph_2023
    and the appropriate references given below.
    
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

##########################################################################################################
##########################################################################################################

def SII_BPT(input, snr=3, Kewley01=False, mask=None):
    '''
    If using these diagnostic fuctions please ref Mar_&_Steph_2023
    and the appropriate references given below.
    
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
    agnliner_sii=(sii_bpt) & ((log_oiii_hb>=Kew01_sii) | (log_sii_ha>=0.32))
    agn_sii=(agnliner_sii) & (log_oiii_hb>=Kew06_sii)
    liner_sii=(agnliner_sii) & (log_oiii_hb<Kew06_sii)
    sf_sii=(sii_bpt) & (~agnliner_sii)

    return (sii_bpt, sf_sii, agn_sii, liner_sii, quiescent_sii)

##########################################################################################################
##########################################################################################################

def OI_BPT(input, snr=3, snrOI=1, Kewley01=False, mask=None):
    '''
    If using these diagnostic fuctions please ref Mar_&_Steph_2023
    and the appropriate references given below.
    
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
    
    WHAN regions defeined as:
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
    If using these diagnostic fuctions please ref Mar_&_Steph_2023
    and the appropriate references given below.
    
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
    agn_heii=(heii_bpt) & log_heii_hb>=Shir12
    sf_heii=(heii_bpt) & ~agn_heii

    return (heii_bpt, agn_heii, sf_nii)

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
    input['NEV_3426_FLUX']=np.where(input['NEV_3426_FLUX_IVAR']==0,np.nan,input['NEV_3426_FLUX_IVAR'])

    # Mask for SNR. Default is OI-BPT is available if Ha, Hb, OIII SNR >= 3 and OI SNR >= 1.
    snr = snr
    SNR_NeV=input['NEV_3426_FLUX']*np.sqrt(input['NEV_3426_FLUX_IVAR'])

    ## NeV diagnostic is available if flux is not zero
    nev = (~zero_flux_nev)

    ## NeV-AGN, SF
    agn_nev= (nev) & (SNR_NeV >= snr)
    sf_nev= (nev) & ~agn_nev

    return (agn_nev, sf_nev)

##########################################################################################################
##########################################################################################################
