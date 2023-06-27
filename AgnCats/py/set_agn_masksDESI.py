## Raga Pucha: First draft of the function 
## Need to add detailed comments to the script
## Returns a lot of warnings because of division by zero - have to use filterwarnings after checking everything
## Version: 2021 December 21
## Edited: B. Canning 24th April 2022, 2023

import numpy as np
from astropy.table import Column

###
def get_qso_maskbits(file):
    import yaml
    from desiutil.bitmask import BitMask
    file_yaml = open(file, 'r')
    yaml_defs = yaml.safe_load(file_yaml)
    QSO_MASKBITS = BitMask('QSO_MASKBITS', yaml_defs)
    AGN_MASKBITS = BitMask('AGN_MASKBITS', yaml_defs)
    AGN_TYPE = BitMask('AGN_TYPE', yaml_defs)
    return QSO_MASKBITS, AGN_MASKBITS, AGN_TYPE
###

###
def update_AGN_MASKBITS(T, QSO_MASKBITS, AGN_MASKBITS):

    ## EC doesn't use yaml - no QN_NEW_RR but we add this
    qsom_RR = T['QSO_MASKBITS'] & QSO_MASKBITS.RR != 0
    qsom_mgii = (T['QSO_MASKBITS'] & QSO_MASKBITS.MGII != 0)  
    qsom_QN = (T['QSO_MASKBITS'] & QSO_MASKBITS.QN != 0)
    qsom_QN_RR = (T['QSO_MASKBITS'] & QSO_MASKBITS.QN_NEW_RR != 0)

    agn_bits = np.zeros(len(T))
    agn_mask = AGN_MASKBITS

    agn_bits = qsom_RR * agn_mask.RR 
    agn_bits |= qsom_mgii * agn_mask.MGII
    agn_bits |= qsom_QN * agn_mask.QN
    agn_bits |= qsom_QN_RR * agn_mask.QN_NEW_RR

    agnmaskbits_column = Column(agn_bits, name = 'AGN_MASKBITS')
    if 'AGN_MASKBITS' in T.columns:
        T['AGN_MASKBITS']=agn_bits
    else:
        T.add_column(agnmaskbits_column)
    return T
###

###
def update_AGNTYPE_NIIBPT(T, AGN_TYPE, snr=3, mask=None):
    
    from AGNdiagnosticsFunctionsDESI import NII_BPT
    '''
    --[NII]-BPT masks--

    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    AGN_TYPE - bitmask structure from yaml 
    
    outputs:
    T - table with new column 'AGN_TYPE'
    '''    
    nii_bpt, sf_nii, agn_nii, liner_nii, composite_nii, quiescent_nii = NII_BPT(T, snr=snr, mask=mask)
    bpt_mask = np.zeros(len(T))    
    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    bpt_mask = nii_bpt * AGN_TYPE.NII_BPT_AV           ## All the emission lines have S/N >= 3
    bpt_mask |= sf_nii * AGN_TYPE.NII_SF               ## [NII] - Star Forming
    bpt_mask |= agn_nii * AGN_TYPE.NII_SY              ## [NII] - Seyfert
    bpt_mask |= liner_nii * AGN_TYPE.NII_LINER         ## [NII] - LINER
    bpt_mask |= composite_nii * AGN_TYPE.NII_COMP      ## [NII] - Composite
    bpt_mask |= quiescent_nii * AGN_TYPE.NII_QUIES     ## [NII] - Quiescent -- S/N one of the lines < 3
    #  
    bptmask_column = Column(bpt_mask, name = 'AGN_TYPE')
    if 'AGN_TYPE' in T.columns:
        T['AGN_TYPE']|=bpt_mask
    else:
        T.add_column(bptmask_column)
    
    return T
###

###
def update_AGNTYPE_SIIBPT(T, AGN_TYPE, snr=3, Kewley01=False, mask=None):
    
    from AGNdiagnosticsFunctionsDESI import SII_BPT
    '''
    --[SII]-BPT masks--

    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    AGN_TYPE - bitmask structure from yaml 
    
    outputs:
    T - table with new column 'AGN_TYPE'
    '''    
    sii_bpt, sf_sii, agn_sii, liner_sii, quiescent_sii = SII_BPT(T, snr=snr, Kewley01=Kewley01, mask=mask)
    bpt_mask = np.zeros(len(T))    
    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    bpt_mask = sii_bpt * AGN_TYPE.SII_BPT_AV          ## All the emission lines have S/N >= 3
    bpt_mask |= sf_sii * AGN_TYPE.SII_SF               ## [SII] - Star Forming
    bpt_mask |= agn_sii * AGN_TYPE.SII_SY              ## [SII] - Seyfert
    bpt_mask |= liner_sii * AGN_TYPE.SII_LINER         ## [SII] - LINER
    bpt_mask |= quiescent_sii * AGN_TYPE.SII_QUIES     ## [SII] - Quiescent -- S/N for one of the lines < 3
    #  
    bptmask_column = Column(bpt_mask, name = 'AGN_TYPE')
    if 'AGN_TYPE' in T.columns:
        T['AGN_TYPE']|=bpt_mask
    else:
        T.add_column(bptmask_column)
    
    return T
###

###
def update_AGNTYPE_OIBPT(T, AGN_TYPE, snr=3, snrOI=1, Kewley01=False, mask=None):
    
    from AGNdiagnosticsFunctionsDESI import OI_BPT
    '''
    --[OI]-BPT masks--

    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    AGN_TYPE - bitmask structure from yaml 
    
    outputs:
    T - table with new column 'AGN_TYPE'
    '''    
    oi_bpt, sf_oi, agn_oi, liner_oi = OI_BPT(T)
    bpt_mask = np.zeros(len(T))    
    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    bpt_mask = oi_bpt * AGN_TYPE.OI_BPT_AV            ## Except [OI] - other emission lines have S/N >= 3
    bpt_mask |= sf_oi * AGN_TYPE.OI_SF                 ## [OI] - Star Forming
    bpt_mask |= agn_oi * AGN_TYPE.OI_SY                ## [OI] - Seyfert
    bpt_mask |= liner_oi * AGN_TYPE.OI_LINER           ## [OI] - LINER
    #  
    bptmask_column = Column(bpt_mask, name = 'AGN_TYPE')
    if 'AGN_TYPE' in T.columns:
        T['AGN_TYPE']|=bpt_mask
    else:
        T.add_column(bptmask_column)
    
    return T
###

###
def update_AGNTYPE_WHAN(T, AGN_TYPE, snr=3, mask=None):
    
    from AGNdiagnosticsFunctionsDESI import WHAN
    '''
    --WHAN masks--

    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    AGN_TYPE - bitmask structure from yaml 
    
    outputs:
    T - table with new column 'AGN_TYPE'
    '''    
    whan, whan_sf, whan_sagn, whan_wagn, whan_retired, whan_passive = WHAN(T, snr=snr, mask=mask)
    agn_mask = np.zeros(len(T))    
    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    agn_mask = whan * AGN_TYPE.WHAN_AV            ## WHAN is available (Halpha and [NII])
    agn_mask |= whan_sf * AGN_TYPE.WHAN_SF         ## WHAN Star-forming
    agn_mask |= whan_sagn * AGN_TYPE.WHAN_SAGN     ## WHAN Strong AGN
    agn_mask |= whan_wagn * AGN_TYPE.WHAN_WAGN          ## WHAN Weak AGN
    agn_mask |= whan_retired * AGN_TYPE.WHAN_RET        ## WHAN Retired
    agn_mask |= whan_passive * AGN_TYPE.WHAN_PASS       ## WHAN Passive
    #  
    agnmask_column = Column(agn_mask, name = 'AGN_TYPE')
    if 'AGN_TYPE' in T.columns:
        T['AGN_TYPE']|=agn_mask
    else:
        T.add_column(agnmask_column)
    
    return T
###

###
def update_AGNTYPE_BLUE(T, AGN_TYPE, snr=3, snrOII=1, mask=None):
    
    from AGNdiagnosticsFunctionsDESI import BLUE
    '''
    --BLUE masks--

    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    AGN_TYPE - bitmask structure from yaml
        
    outputs:
    T - table with new column 'AGN_TYPE'
    '''    
    blue, agn_blue, sflin_blue, liner_blue, sf_blue, sfagn_blue = BLUE(T, snr=snr, snrOII=snrOII, mask=None)
    agn_mask = np.zeros(len(T))    
    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    agn_mask = blue * AGN_TYPE.BLUE_AV           ## 
    agn_mask |= agn_blue * AGN_TYPE.BLUE_AGN         ## 
    agn_mask |= sflin_blue * AGN_TYPE.BLUE_SLC     ## 
    agn_mask |= liner_blue * AGN_TYPE.BLUE_LINER          ## 
    agn_mask |= sf_blue * AGN_TYPE.BLUE_SF        ## 
    agn_mask |= sfagn_blue * AGN_TYPE.BLUE_SFAGN      ##
    #  
    agnmask_column = Column(agn_mask, name = 'AGN_TYPE')
    if 'AGN_TYPE' in T.columns:
        T['AGN_TYPE']|=agn_mask
    else:
        T.add_column(agnmask_column)
    
    return T
###

###
def update_AGNTYPE_HeII(T, AGN_TYPE, snr=3, mask=None):
    
    from AGNdiagnosticsFunctionsDESI import HeII_BPT
    '''
    --He II masks--

    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    AGN_TYPE - bitmask structure from yaml
       
    outputs:
    T - table with new column 'AGN_TYPE'
    '''    
    heii_bpt, agn_heii, sf_heii = HeII_BPT(T, snr=snr, mask=None)
    agn_mask = np.zeros(len(T))    
    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    agn_mask = heii_bpt * AGN_TYPE.HEII_BPT_AV           ## 
    agn_mask |= agn_heii * AGN_TYPE.HEII_AGN         ## 
    agn_mask |= sf_heii * AGN_TYPE.HEII_SF     ## 
    #  
    agnmask_column = Column(agn_mask, name = 'AGN_TYPE')
    if 'AGN_TYPE' in T.columns:
        T['AGN_TYPE']|=agn_mask
    else:
        T.add_column(agnmask_column)
    
    return T
###

###
def update_AGNTYPE_NeV(T, AGN_TYPE, snr=2.5, mask=None):
    
    from AGNdiagnosticsFunctionsDESI import NeV
    '''
    --NeV masks--

    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    AGN_TYPE - bitmask structure from yaml
    
    outputs:
    T - table with new column 'AGN_TYPE'
    '''    
    nev, agn_nev, sf_nev = NeV(T, snr=snr, mask=None)
    agn_mask = np.zeros(len(T))    
    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    agn_mask = nev * AGN_TYPE.NEV_AV           ## 
    agn_mask |= agn_nev * AGN_TYPE.NEV_AGN         ## 
    agn_mask |= sf_nev * AGN_TYPE.NEV_SF     ## 
    #  
    agnmask_column = Column(agn_mask, name = 'AGN_TYPE')
    if 'AGN_TYPE' in T.columns:
        T['AGN_TYPE']|=agn_mask
    else:
        T.add_column(agnmask_column)
    
    return T
###

###
def update_AGNTYPE_WISE_colors(T, AGN_TYPE, snr=3, mask=None):
    
    from AGNdiagnosticsFunctionsDESI import WISE_colors
    '''
    --WISE colour masks--

    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    AGN_TYPE - bitmask structure from yaml
    
    outputs:
    T - table with new column 'AGN_TYPE'
    '''    
    W1W2_avail, agn_ir, sf_ir = WISE_colors(T, snr=snr, mask=None)
    agn_mask = np.zeros(len(T))    
    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    agn_mask = W1W2_avail * AGN_TYPE.WISE_AV           ## 
    agn_mask |= agn_ir * AGN_TYPE.WISE_AGN         ## 
    agn_mask |= sf_ir * AGN_TYPE.WISE_SF     ## 
    #  
    agnmask_column = Column(agn_mask, name = 'AGN_TYPE')
    if 'AGN_TYPE' in T.columns:
        T['AGN_TYPE']|=agn_mask
    else:
        T.add_column(agnmask_column)
    
    return T
###




###
def test_bpt_mask(tab, AGN_TYPE, directory):
    '''
    BC modified from RP code May 22
    Plots a test bpt in NII, SII and OI to make sure all is working well
    This code require AGNTYPE maskbits to be present in the data

    inputs: 
    tab - table joined with FastSpecFit (or other emline cat)
    AGN_TYPE - bitmask structure from yaml 
    
    outputs:
    image in directory/test_bpt.jpg
    '''
    import matplotlib.pyplot as plt
    if 'AGN_TYPE' in tab.columns:
        print("Found mask")
        bptmask = tab['AGN_TYPE']
    
        bpt_type = AGN_TYPE

        nii_sf = (bptmask & bpt_type.NII_SF != 0)
        nii_sy = (bptmask & bpt_type.NII_SY != 0)
        nii_lin = (bptmask & bpt_type.NII_LINER != 0)
        nii_comp = (bptmask & bpt_type.NII_COMP != 0)

        sii_sf = (bptmask & bpt_type.SII_SF != 0)
        sii_sy = (bptmask & bpt_type.SII_SY != 0)
        sii_lin = (bptmask & bpt_type.SII_LINER != 0)

        oi_sf = (bptmask & bpt_type.OI_SF != 0)
        oi_sy = (bptmask & bpt_type.OI_SY != 0)
        oi_lin = (bptmask & bpt_type.OI_LINER != 0)

        xx_bptnii=np.log10(tab['NII_6584_FLUX']/tab['HALPHA_FLUX'])
        xx_bptsii = np.log10((tab['SII_6716_FLUX']+tab['SII_6731_FLUX'])/tab['HALPHA_FLUX'])
        xx_bptoi = np.log10(tab['OI_6300_FLUX']/tab['HALPHA_FLUX'])

        yy_bpt=np.log10(tab['OIII_5007_FLUX']/tab['HBETA_FLUX'])
        plt.figure(figsize = (24,8))

        plt.subplot2grid((1,3), (0,0))
        plt.title('[NII]-BPT')
        plt.scatter(xx_bptnii[nii_sf], yy_bpt[nii_sf], color = 'b', s = 10)
        plt.scatter(xx_bptnii[nii_sy], yy_bpt[nii_sy], color = 'r', s = 10)
        plt.scatter(xx_bptnii[nii_comp], yy_bpt[nii_comp], color = 'g', s = 10)
        plt.scatter(xx_bptnii[nii_lin], yy_bpt[nii_lin], color = 'purple', s = 10)
        plt.xlabel('log ([NII]/H$\\alpha$)')
        plt.ylabel('log ([OIII]/H$\\beta$)')

        plt.subplot2grid((1,3), (0,1))
        plt.title('[SII]-BPT')
        plt.scatter(xx_bptsii[sii_sf], yy_bpt[sii_sf], color = 'b', s = 10)
        plt.scatter(xx_bptsii[sii_sy], yy_bpt[sii_sy], color = 'r', s = 10)
        plt.scatter(xx_bptsii[sii_lin], yy_bpt[sii_lin], color = 'purple', s = 10)
        plt.xlabel('log ([SII]/H$\\alpha$)')
        plt.ylabel('log ([OIII]/H$\\beta$)')

        plt.subplot2grid((1,3), (0,2))
        plt.title('[OI]-BPT')
        plt.scatter(xx_bptoi[oi_sf], yy_bpt[oi_sf], color = 'b', s = 10)
        plt.scatter(xx_bptoi[oi_sy], yy_bpt[oi_sy], color = 'r', s = 10)
        plt.scatter(xx_bptoi[oi_lin], yy_bpt[oi_lin], color = 'purple', s = 10)
        plt.xlabel('log ([OI]/H$\\alpha$)')
        plt.ylabel('log ([OIII]/H$\\beta$)')

        plt.tight_layout()
        plt.savefig(directory+'test_bpt.jpg')
    else:
        print("Please run create_bpt_mask_all or create_bpt_mask first")

###################################################################################################################

def update_AGNTYPE_BPT_all(T, AGN_TYPE):
    
    from AGNdiagnosticsFunctionsDESI import NII_BPT, SII_BPT, OI_BPT, WHAN
    '''
    BC modified to take from AGNDiagnsticFunctions
    BC modified from RP code May 22
    
    Calls emline_classification to make ratios and test against standard lines
    Makes AGN_TYPE
    
    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    AGN_TYPE - bitmask structure from yaml 
    
    outputs:
    T - table with new column 'AGN_TYPE'
    '''    
    nii_bpt, sf_nii, agn_nii, liner_nii, composite_nii, quiescent_nii = NII_BPT(T)
    sii_bpt, sf_sii, agn_sii, liner_sii, quiescent_sii = SII_BPT(T)
    oi_bpt, sf_oi, agn_oi, liner_oi = OI_BPT(T)

    bpt_type = AGN_TYPE
    
    bpt_mask = np.zeros(len(T))
    
    ## If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    ## [NII]-BPT masks
    bpt_mask = nii_bpt * bpt_type.NII_BPT_AV           ## All the emission lines have S/N >= 3
    bpt_mask |= sf_nii * bpt_type.NII_SF               ## [NII] - Star Forming
    bpt_mask |= agn_nii * bpt_type.NII_SY              ## [NII] - Seyfert
    bpt_mask |= liner_nii * bpt_type.NII_LINER         ## [NII] - LINER
    bpt_mask |= composite_nii * bpt_type.NII_COMP      ## [NII] - Composite
    bpt_mask |= quiescent_nii * bpt_type.NII_QUIES     ## [NII] - Quiescent -- S/N one of the lines < 3
    
    ## [SII]-BPT masks
    bpt_mask |= sii_bpt * bpt_type.SII_BPT_AV          ## All the emission lines have S/N >= 3
    bpt_mask |= sf_sii * bpt_type.SII_SF               ## [SII] - Star Forming
    bpt_mask |= agn_sii * bpt_type.SII_SY              ## [SII] - Seyfert
    bpt_mask |= liner_sii * bpt_type.SII_LINER         ## [SII] - LINER
    bpt_mask |= quiescent_sii * bpt_type.SII_QUIES     ## [SII] - Quiescent -- S/N for one of the lines < 3

    ## [OI]-BPT masks
    bpt_mask |= oi_bpt * bpt_type.OI_BPT_AV            ## Except [OI] - other emission lines have S/N >= 3
    bpt_mask |= sf_oi * bpt_type.OI_SF                 ## [OI] - Star Forming
    bpt_mask |= agn_oi * bpt_type.OI_SY                ## [OI] - Seyfert
    bpt_mask |= liner_oi * bpt_type.OI_LINER           ## [OI] - LINER
    
    bptmask_column = Column(bpt_mask, name = 'AGN_TYPE')
    if 'AGN_TYPE' in T.columns:
        T['AGN_TYPE']=bpt_mask
    else:
        T.add_column(bptmask_column)
        
    return T

def create_bpt_mask_all(T, AGN_TYPE):
    
    from AGNdiagnosticsFunctionsDESI import NII_BPT, SII_BPT, OI_BPT, WHAN
    '''
    BC modified to take from AGNDiagnsticFunctions
    BC modified from RP code May 22
    
    Calls emline_classification to make ratios and test against standard lines
    Makes AGN_TYPE
    
    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    AGN_TYPE - bitmask structure from yaml 
    
    outputs:
    T - table with new column 'AGN_TYPE'
    '''    
    nii_bpt, sf_nii, agn_nii, liner_nii, composite_nii, quiescent_nii = NII_BPT(T)
    sii_bpt, sf_sii, agn_sii, liner_sii, quiescent_sii = SII_BPT(T)
    oi_bpt, sf_oi, agn_oi, liner_oi = OI_BPT(T)

    bpt_type = AGN_TYPE
    
    bpt_mask = np.zeros(len(T))
    
    ## If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    ## [NII]-BPT masks
    bpt_mask = nii_bpt * bpt_type.NII_BPT_AV           ## All the emission lines have S/N >= 3
    bpt_mask |= sf_nii * bpt_type.NII_SF               ## [NII] - Star Forming
    bpt_mask |= agn_nii * bpt_type.NII_SY              ## [NII] - Seyfert
    bpt_mask |= liner_nii * bpt_type.NII_LINER         ## [NII] - LINER
    bpt_mask |= composite_nii * bpt_type.NII_COMP      ## [NII] - Composite
    bpt_mask |= quiescent_nii * bpt_type.NII_QUIES     ## [NII] - Quiescent -- S/N one of the lines < 3
    
    ## [SII]-BPT masks
    bpt_mask |= sii_bpt * bpt_type.SII_BPT_AV          ## All the emission lines have S/N >= 3
    bpt_mask |= sf_sii * bpt_type.SII_SF               ## [SII] - Star Forming
    bpt_mask |= agn_sii * bpt_type.SII_SY              ## [SII] - Seyfert
    bpt_mask |= liner_sii * bpt_type.SII_LINER         ## [SII] - LINER
    bpt_mask |= quiescent_sii * bpt_type.SII_QUIES     ## [SII] - Quiescent -- S/N for one of the lines < 3

    ## [OI]-BPT masks
    bpt_mask |= oi_bpt * bpt_type.OI_BPT_AV            ## Except [OI] - other emission lines have S/N >= 3
    bpt_mask |= sf_oi * bpt_type.OI_SF                 ## [OI] - Star Forming
    bpt_mask |= agn_oi * bpt_type.OI_SY                ## [OI] - Seyfert
    bpt_mask |= liner_oi * bpt_type.OI_LINER           ## [OI] - LINER
    
    bptmask_column = Column(bpt_mask, name = 'AGN_TYPE')
    if 'AGN_TYPE' in T.columns:
        T['AGN_TYPE']=bpt_mask
    else:
        T.add_column(bptmask_column)
    
    return T
