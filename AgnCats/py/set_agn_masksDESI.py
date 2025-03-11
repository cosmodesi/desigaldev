## Raga Pucha: First draft of the function (2021)
## Need to add detailed comments to the script
## Returns a lot of warnings because of division by zero - have to use filterwarnings after checking everything
## Version: 2025 February 25
## Edited: B. Canning 24th April 2022, 2023, S. Juneau November 2024, Feb 2025

import numpy as np
from astropy.table import Column

###
# Retrieve the bitmasks definitions from the yaml file
# Note: QSO_MASKBITS are applied to the first 9 bits of AGN_MASKBITS
#       OPT_UV_TYPE and IR_TYPE include definitions for detailed classification
def get_agn_maskbits(file):
    import yaml
    from desiutil.bitmask import BitMask
    file_yaml = open(file, 'r')
    yaml_defs = yaml.safe_load(file_yaml)
    
    AGN_MASKBITS = BitMask('AGN_MASKBITS', yaml_defs)
    OPT_UV_TYPE = BitMask('OPT_UV_TYPE', yaml_defs)
    IR_TYPE = BitMask('IR_TYPE', yaml_defs)
    
    return AGN_MASKBITS, OPT_UV_TYPE, IR_TYPE
###

###
## SJ: removed QSO_MASKBITS from yaml (still exist as a column in the input file though!)
def update_AGN_MASKBITS(T, AGN_MASKBITS, snr=3, snrOI=1, snrOII=1, Kewley01=False, mask=None):

    from AGNdiagnosticsFunctionsDESI import NII_BPT
    from AGNdiagnosticsFunctionsDESI import SII_BPT
    from AGNdiagnosticsFunctionsDESI import OI_BPT
    from AGNdiagnosticsFunctionsDESI import BROAD_LINE
    from AGNdiagnosticsFunctionsDESI import WISE_colors
    from AGNdiagnosticsFunctionsDESI import MEX
    from AGNdiagnosticsFunctionsDESI import KEX
    from AGNdiagnosticsFunctionsDESI import BLUE
    from AGNdiagnosticsFunctionsDESI import WHAN
    
    ## SJ: need to use AGN_MASKBITS instead
    qsom_RR = T['QSO_MASKBITS'] & AGN_MASKBITS.RR != 0
    qsom_mgii = (T['QSO_MASKBITS'] & AGN_MASKBITS.MGII != 0)  
    qsom_QN = (T['QSO_MASKBITS'] & AGN_MASKBITS.QN != 0)
    qsom_QN_RR = (T['QSO_MASKBITS'] & AGN_MASKBITS.QN_NEW_RR != 0)
    qsom_QN_BGS = (T['QSO_MASKBITS'] & AGN_MASKBITS.QN_BGS != 0)
    qsom_QN_ELG = (T['QSO_MASKBITS'] & AGN_MASKBITS.QN_ELG != 0)
    qsom_QN_VAR_WISE = (T['QSO_MASKBITS'] & AGN_MASKBITS.QN_VAR_WISE != 0)
     
    agn_bits = np.zeros(len(T))
    agn_mask = AGN_MASKBITS

    agn_bits = qsom_RR * agn_mask.RR 
    agn_bits |= qsom_mgii * agn_mask.MGII
    agn_bits |= qsom_QN * agn_mask.QN
    agn_bits |= qsom_QN_RR * agn_mask.QN_NEW_RR
    # SJ: added these
    agn_bits |= qsom_QN_BGS * agn_mask.QN_BGS
    agn_bits |= qsom_QN_ELG * agn_mask.QN_ELG
    agn_bits |= qsom_QN_VAR_WISE * agn_mask.QN_VAR_WISE

    # BPT classifications from individual diagnostics
    nii_bpt, sf_nii, agn_nii, liner_nii, composite_nii = NII_BPT(T, snr=snr, mask=mask)
    sii_bpt, sf_sii, agn_sii, liner_sii = SII_BPT(T, snr=snr, Kewley01=Kewley01, mask=mask)
    oi_bpt, sf_oi, agn_oi, liner_oi = OI_BPT(T, snr=snr, snrOI=snrOI, Kewley01=Kewley01, mask=mask)

    # Combined BPT classification
    bpt_any_sy = agn_nii | agn_sii | agn_oi
    bpt_any_agn = agn_nii | agn_sii | agn_oi | liner_nii | composite_nii | liner_sii | liner_oi
    agn_bits |= bpt_any_sy * agn_mask.BPT_ANY_SY 
    agn_bits |= bpt_any_agn * agn_mask.BPT_ANY_AGN

    # Whether there is a broad line (FWHM>= 1200 km/s)
    bl = BROAD_LINE(T, snr=snr, mask=mask, vel_thres=1200.)
    agn_bits |= bl * agn_mask.BROAD_LINE 
   
    # Other (non-BPT) optical diagnostics: WHAN, MEx, KEx, Blue
    whan, whan_sf, whan_sagn, whan_wagn, whan_retired, whan_passive = WHAN(T, snr=snr, mask=mask)
    mex, mex_agn, mex_sf, mex_interm = MEX(T, snr=snr, mask=mask)
    blue, agn_blue, sflin_blue, liner_blue, sf_blue, sfagn_blue = BLUE(T, snr=snr, snrOII=snrOII, mask=mask)
    kex, kex_agn, kex_sf, kex_interm = KEX(T, snr=snr, mask=mask)

    # Combine them for the OPT_OTHER_AGN (keeping mostly more confident ones and 
    # exclusing possible weak AGN / blended classes)
    opt_other_agn = whan_sagn | mex_agn | agn_blue | kex_agn
    agn_bits |= opt_other_agn * agn_mask.OPT_OTHER_AGN

    # Overall WISE classification (combining all diagnostics
    snr_wise = 3.
    wise, agn_wise, sf_wise = WISE_colors(T, snr=snr_wise, mask=mask)
    agn_bits |= agn_wise * agn_mask.WISE_ANY_AGN
    
    # uv, xray, radio =
    #agn_bits |= uv * agn_mask.UV
    #agn_bits |= xray * agn_mask.XRAY
    #agn_bits |= radio * agn_mask.RADIO

    # Finally, add the AGN_ANY definition based on confident AGNs
    agn_any = qsom_RR | qsom_mgii | qsom_QN | qsom_QN_BGS | qsom_QN_ELG | qsom_QN_VAR_WISE | bpt_any_sy | agn_wise | opt_other_agn
    agn_bits |= agn_any * agn_mask.AGN_ANY

    agnmaskbits_column = Column(agn_bits, name = 'AGN_MASKBITS')
    if 'AGN_MASKBITS' in T.columns:
        T['AGN_MASKBITS']|=agn_bits
    else:
        T.add_column(agnmaskbits_column)
    return T
###

###
def update_AGNTYPE_NIIBPT(T, OPT_UV_TYPE, snr=3, mask=None):
    
    from AGNdiagnosticsFunctionsDESI import NII_BPT
    '''
    --[NII]-BPT masks--

    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    OPT_UV_TYPE - bitmask structure from yaml 
    
    outputs:
    T - table with new column 'OPT_UV_TYPE'
    '''    
    nii_bpt, sf_nii, agn_nii, liner_nii, composite_nii = NII_BPT(T, snr=snr, mask=mask)
    bpt_mask = np.zeros(len(T))    
    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    bpt_mask = nii_bpt * OPT_UV_TYPE.NII_BPT              ## All the emission lines have S/N >= 3
    bpt_mask |= sf_nii * OPT_UV_TYPE.NII_SF               ## [NII] - Star Forming
    bpt_mask |= agn_nii * OPT_UV_TYPE.NII_SY              ## [NII] - Seyfert
    bpt_mask |= liner_nii * OPT_UV_TYPE.NII_LINER         ## [NII] - LINER
    bpt_mask |= composite_nii * OPT_UV_TYPE.NII_COMP      ## [NII] - Composite
    #  
    bptmask_column = Column(bpt_mask, name = 'OPT_UV_TYPE')
    if 'OPT_UV_TYPE' in T.columns:
        T['OPT_UV_TYPE']|=bpt_mask
    else:
        T.add_column(bptmask_column)
    
    return T
###

###
def update_AGNTYPE_SIIBPT(T, OPT_UV_TYPE, snr=3, Kewley01=False, mask=None):
    
    from AGNdiagnosticsFunctionsDESI import SII_BPT
    '''
    --[SII]-BPT masks--

    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    OPT_UV_TYPE - bitmask structure from yaml 
    
    outputs:
    T - table with new column 'OPT_UV_TYPE'
    '''    
    sii_bpt, sf_sii, agn_sii, liner_sii = SII_BPT(T, snr=snr, Kewley01=Kewley01, mask=mask)
    bpt_mask = np.zeros(len(T))    
    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    bpt_mask = sii_bpt * OPT_UV_TYPE.SII_BPT              ## All the emission lines have S/N >= 3
    bpt_mask |= sf_sii * OPT_UV_TYPE.SII_SF               ## [SII] - Star Forming
    bpt_mask |= agn_sii * OPT_UV_TYPE.SII_SY              ## [SII] - Seyfert
    bpt_mask |= liner_sii * OPT_UV_TYPE.SII_LINER         ## [SII] - LINER
    #  
    bptmask_column = Column(bpt_mask, name = 'OPT_UV_TYPE')
    if 'OPT_UV_TYPE' in T.columns:
        T['OPT_UV_TYPE']|=bpt_mask
    else:
        T.add_column(bptmask_column)
    
    return T
###

###
def update_AGNTYPE_OIBPT(T, OPT_UV_TYPE, snr=3, snrOI=1, Kewley01=False, mask=None):
    
    from AGNdiagnosticsFunctionsDESI import OI_BPT
    '''
    --[OI]-BPT masks--

    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    OPT_UV_TYPE - bitmask structure from yaml 
    
    outputs:
    T - table with new column 'OPT_UV_TYPE'
    '''    
    oi_bpt, sf_oi, agn_oi, liner_oi = OI_BPT(T, snr=snr, snrOI=snrOI, Kewley01=Kewley01, mask=mask)
    bpt_mask = np.zeros(len(T))    
    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    bpt_mask = oi_bpt * OPT_UV_TYPE.OI_BPT                ## Except [OI] - other em lines have S/N >= 3
    bpt_mask |= sf_oi * OPT_UV_TYPE.OI_SF                 ## [OI] - Star Forming
    bpt_mask |= agn_oi * OPT_UV_TYPE.OI_SY                ## [OI] - Seyfert
    bpt_mask |= liner_oi * OPT_UV_TYPE.OI_LINER           ## [OI] - LINER
    #  
    bptmask_column = Column(bpt_mask, name = 'OPT_UV_TYPE')
    if 'OPT_UV_TYPE' in T.columns:
        T['OPT_UV_TYPE']|=bpt_mask
    else:
        T.add_column(bptmask_column)
    
    return T
###

###
def update_AGNTYPE_WHAN(T, OPT_UV_TYPE, snr=3, mask=None):
    
    from AGNdiagnosticsFunctionsDESI import WHAN
    '''
    --WHAN masks--

    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    OPT_UV_TYPE - bitmask structure from yaml 
    
    outputs:
    T - table with new column 'OPT_UV_TYPE'
    '''    
    whan, whan_sf, whan_sagn, whan_wagn, whan_retired, whan_passive = WHAN(T, snr=snr, mask=mask)
    agn_mask = np.zeros(len(T))    
    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    agn_mask = whan * OPT_UV_TYPE.WHAN                ## WHAN is available (Halpha and [NII])
    agn_mask |= whan_sf * OPT_UV_TYPE.WHAN_SF         ## WHAN Star-forming
    agn_mask |= whan_sagn * OPT_UV_TYPE.WHAN_SAGN     ## WHAN Strong AGN
    agn_mask |= whan_wagn * OPT_UV_TYPE.WHAN_WAGN          ## WHAN Weak AGN
    agn_mask |= whan_retired * OPT_UV_TYPE.WHAN_RET        ## WHAN Retired
    agn_mask |= whan_passive * OPT_UV_TYPE.WHAN_PASS       ## WHAN Passive
    #  
    agnmask_column = Column(agn_mask, name = 'OPT_UV_TYPE')
    if 'OPT_UV_TYPE' in T.columns:
        T['OPT_UV_TYPE']|=agn_mask
    else:
        T.add_column(agnmask_column)
    
    return T
###

###
def update_AGNTYPE_BLUE(T, OPT_UV_TYPE, snr=3, snrOII=1, mask=None):
    
    from AGNdiagnosticsFunctionsDESI import BLUE
    '''
    --BLUE masks--

    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    OPT_UV_TYPE - bitmask structure from yaml
        
    outputs:
    T - table with new column 'OPT_UV_TYPE'
    '''    
    blue, agn_blue, sflin_blue, liner_blue, sf_blue, sfagn_blue = BLUE(T, snr=snr, snrOII=snrOII, mask=mask)
    agn_mask = np.zeros(len(T))    
    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    agn_mask = blue * OPT_UV_TYPE.BLUE
    agn_mask |= agn_blue * OPT_UV_TYPE.BLUE_AGN
    agn_mask |= sflin_blue * OPT_UV_TYPE.BLUE_SLC
    agn_mask |= liner_blue * OPT_UV_TYPE.BLUE_LINER
    agn_mask |= sf_blue * OPT_UV_TYPE.BLUE_SF
    agn_mask |= sfagn_blue * OPT_UV_TYPE.BLUE_SFAGN
    #  
    agnmask_column = Column(agn_mask, name = 'OPT_UV_TYPE')
    if 'OPT_UV_TYPE' in T.columns:
        T['OPT_UV_TYPE']|=agn_mask
    else:
        T.add_column(agnmask_column)
    
    return T
###


### Mass-Excitation (MEx)
###
def update_AGNTYPE_MEX(T, OPT_UV_TYPE, snr=3, mask=None):
    
    from AGNdiagnosticsFunctionsDESI import MEX

    mex, mex_agn, mex_sf, mex_interm = MEX(T, snr=snr, mask=mask)

    agn_mask = np.zeros(len(T))    
    agn_mask = mex * OPT_UV_TYPE.MEX
    agn_mask |= mex_agn * OPT_UV_TYPE.MEX_AGN
    agn_mask |= mex_sf * OPT_UV_TYPE.MEX_SF
    agn_mask |= mex_interm * OPT_UV_TYPE.MEX_INTERM    

    agnmask_column = Column(agn_mask, name = 'OPT_UV_TYPE')
    if 'OPT_UV_TYPE' in T.columns:
        T['OPT_UV_TYPE']|=agn_mask
    else:
        T.add_column(agnmask_column)
    
    return T
###

### Kinematics-Excitation (KEx)
###
def update_AGNTYPE_KEX(T, OPT_UV_TYPE, snr=3, mask=None):
    
    from AGNdiagnosticsFunctionsDESI import KEX

    kex, kex_agn, kex_sf, kex_interm = KEX(T, snr=snr, mask=mask)

    agn_mask = np.zeros(len(T))    
    agn_mask = kex * OPT_UV_TYPE.KEX
    agn_mask |= kex_agn * OPT_UV_TYPE.KEX_AGN
    agn_mask |= kex_sf * OPT_UV_TYPE.KEX_SF
    agn_mask |= kex_interm * OPT_UV_TYPE.KEX_INTERM    

    agnmask_column = Column(agn_mask, name = 'OPT_UV_TYPE')
    if 'OPT_UV_TYPE' in T.columns:
        T['OPT_UV_TYPE']|=agn_mask
    else:
        T.add_column(agnmask_column)
    
    return T
###

###
def update_AGNTYPE_HeII(T, OPT_UV_TYPE, snr=3, mask=None):
    
    from AGNdiagnosticsFunctionsDESI import HeII_BPT
    '''
    --He II masks--

    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    OPT_UV_TYPE - bitmask structure from yaml
       
    outputs:
    T - table with new column 'OPT_UV_TYPE'
    '''    
    heii_bpt, agn_heii, sf_heii = HeII_BPT(T, snr=snr, mask=None)
    agn_mask = np.zeros(len(T))    
    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    agn_mask = heii_bpt * OPT_UV_TYPE.HEII_BPT
    agn_mask |= agn_heii * OPT_UV_TYPE.HEII_AGN
    agn_mask |= sf_heii * OPT_UV_TYPE.HEII_SF
    #  
    agnmask_column = Column(agn_mask, name = 'OPT_UV_TYPE')
    if 'OPT_UV_TYPE' in T.columns:
        T['OPT_UV_TYPE']|=agn_mask
    else:
        T.add_column(agnmask_column)
    
    return T
###

###
def update_AGNTYPE_NeV(T, OPT_UV_TYPE, snr=2.5, mask=None):
    
    from AGNdiagnosticsFunctionsDESI import NeV
    '''
    --NeV masks--

    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    OPT_UV_TYPE - bitmask structure from yaml
    
    outputs:
    T - table with new column 'OPT_UV_TYPE'
    '''    
    nev, agn_nev, sf_nev = NeV(T, snr=snr, mask=None)
    agn_mask = np.zeros(len(T))    
    # If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    agn_mask = nev * OPT_UV_TYPE.NEV
    agn_mask |= agn_nev * OPT_UV_TYPE.NEV_AGN
    agn_mask |= sf_nev * OPT_UV_TYPE.NEV_SF
    #  
    agnmask_column = Column(agn_mask, name = 'OPT_UV_TYPE')
    if 'OPT_UV_TYPE' in T.columns:
        T['OPT_UV_TYPE']|=agn_mask
    else:
        T.add_column(agnmask_column)
    
    return T
###

###
def update_AGNTYPE_WISE_colors(T, IR_TYPE, snr=3, mask=None):
    
    from AGNdiagnosticsFunctionsDESI import WISE_colors
    '''
    --WISE colour masks--

    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    IR_TYPE - bitmask structure from yaml
    
    outputs:
    T - table with new column 'IR_TYPE'
    '''    

    agn_mask = np.zeros(len(T))
        
    # 'Jarrett11'
    ## using this example to save the wise_w123 info (where W1, W2, W3 are all above S/N cut)
    wise_w123, agn_ir, sf_ir = WISE_colors(T, snr=snr, mask=None, diag='Jarrett11')
    agn_mask = agn_ir * IR_TYPE.WISE_AGN_J11
    agn_mask |= sf_ir * IR_TYPE.WISE_SF_J11

    # 'Stern12'
    ## using this example to save the wise_w12 info (where W1, W2 are both above S/N cut)
    wise_w12, agn_ir, sf_ir = WISE_colors(T, snr=snr, mask=None, diag='Stern12')
    agn_mask |= agn_ir * IR_TYPE.WISE_AGN_S12
    agn_mask |= sf_ir * IR_TYPE.WISE_SF_S12

    # 'Mateos12'
    wise_av, agn_ir, sf_ir = WISE_colors(T, snr=snr, mask=None, diag='Mateos12')
    agn_mask |= agn_ir * IR_TYPE.WISE_AGN_M12
    agn_mask |= sf_ir * IR_TYPE.WISE_SF_M12

    # 'Assef18'
    wise_av, agn_ir, sf_ir = WISE_colors(T, snr=snr, mask=None, diag='Assef18')
    agn_mask |= agn_ir * IR_TYPE.WISE_AGN_A18
    agn_mask |= sf_ir * IR_TYPE.WISE_SF_A18

    # 'Yao20'
    wise_av, agn_ir, sf_ir = WISE_colors(T, snr=snr, mask=None, diag='Yao20')
    agn_mask |= agn_ir * IR_TYPE.WISE_AGN_Y20
    agn_mask |= sf_ir * IR_TYPE.WISE_SF_Y20

    # 'Hviding22'
    wise_av, agn_ir, sf_ir = WISE_colors(T, snr=snr, mask=None, diag='Hviding22')
    agn_mask |= agn_ir * IR_TYPE.WISE_AGN_H22
    agn_mask |= sf_ir * IR_TYPE.WISE_SF_H22
        
    ## SJ: not sure if I can automate in a loop
#    diags = ['Jarrett11', 'Mateos12', 'Stern12', 'Assef18', 'Yao20', 'Hviding22']
#    for diag in diags:
#        wise_avail, agn_ir, sf_ir = WISE_colors(T, snr=snr, mask=None, diag=diag)
#        if diag=='Stern12':
#            wise_w12 = wise_avail
#        if diag=='Jarrett11':
#            wise_w123 = wise_avail
        

    # If W1, W2 fluxes are above threshold snr (required for Stern+ and Assef+)
    agn_mask |= wise_w12 * IR_TYPE.WISE_W12
    # If W1, W2, W3 fluxes are above threshold snr (required for Jarrett+, Mateos+, Yao+, Hviding+)
    agn_mask |= wise_w123 * IR_TYPE.WISE_W123

    # Turn into a table column
    agnmask_column = Column(agn_mask, name = 'IR_TYPE')
    if 'IR_TYPE' in T.columns:
        T['IR_TYPE']|=agn_mask
    else:
        T.add_column(agnmask_column)
    
    return T
###


###
### THE BELOW ARE  - BC 3 Nov 2024
###
### SJ: are these used? If so, how?
###

###################################################################################################################

def update_AGNTYPE_BPT_all(T, OPT_UV_TYPE):
    
    from AGNdiagnosticsFunctionsDESI import NII_BPT, SII_BPT, OI_BPT, WHAN
    '''
    BC modified to take from AGNDiagnsticFunctions
    BC modified from RP code May 22
    
    Calls emline_classification to make ratios and test against standard lines
    Makes OPT_UV_TYPE
    
    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    OPT_UV_TYPE - bitmask structure from yaml 
    
    outputs:
    T - table with new column 'OPT_UV_TYPE'
    '''    
    nii_bpt, sf_nii, agn_nii, liner_nii, composite_nii, quiescent_nii = NII_BPT(T)
    sii_bpt, sf_sii, agn_sii, liner_sii, quiescent_sii = SII_BPT(T)
    oi_bpt, sf_oi, agn_oi, liner_oi = OI_BPT(T)

    bpt_type = OPT_UV_TYPE
    
    bpt_mask = np.zeros(len(T))
    
    ## If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    ## [NII]-BPT masks
    bpt_mask = nii_bpt * bpt_type.NII_BPT              ## All the emission lines have S/N >= 3
    bpt_mask |= sf_nii * bpt_type.NII_SF               ## [NII] - Star Forming
    bpt_mask |= agn_nii * bpt_type.NII_SY              ## [NII] - Seyfert
    bpt_mask |= liner_nii * bpt_type.NII_LINER         ## [NII] - LINER
    bpt_mask |= composite_nii * bpt_type.NII_COMP      ## [NII] - Composite
    
    ## [SII]-BPT masks
    bpt_mask |= sii_bpt * bpt_type.SII_BPT             ## All the emission lines have S/N >= 3
    bpt_mask |= sf_sii * bpt_type.SII_SF               ## [SII] - Star Forming
    bpt_mask |= agn_sii * bpt_type.SII_SY              ## [SII] - Seyfert
    bpt_mask |= liner_sii * bpt_type.SII_LINER         ## [SII] - LINER

    ## [OI]-BPT masks
    bpt_mask |= oi_bpt * bpt_type.OI_BPT               ## Except [OI] - other emission lines have S/N >= 3
    bpt_mask |= sf_oi * bpt_type.OI_SF                 ## [OI] - Star Forming
    bpt_mask |= agn_oi * bpt_type.OI_SY                ## [OI] - Seyfert
    bpt_mask |= liner_oi * bpt_type.OI_LINER           ## [OI] - LINER
    
    bptmask_column = Column(bpt_mask, name = 'OPT_UV_TYPE')
    if 'OPT_UV_TYPE' in T.columns:
        T['OPT_UV_TYPE']|=bpt_mask
    else:
        T.add_column(bptmask_column)
        
    return T

def create_bpt_mask_all(T, OPT_UV_TYPE):
    
    from AGNdiagnosticsFunctionsDESI import NII_BPT, SII_BPT, OI_BPT, WHAN
    '''
    BC modified to take from AGNDiagnsticFunctions
    BC modified from RP code May 22
    
    Calls emline_classification to make ratios and test against standard lines
    Makes OPT_UV_TYPE
    
    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    OPT_UV_TYPE - bitmask structure from yaml 
    
    outputs:
    T - table with new column 'OPT_UV_TYPE'
    '''    
    nii_bpt, sf_nii, agn_nii, liner_nii, composite_nii, quiescent_nii = NII_BPT(T)
    sii_bpt, sf_sii, agn_sii, liner_sii, quiescent_sii = SII_BPT(T)
    oi_bpt, sf_oi, agn_oi, liner_oi = OI_BPT(T)

    bpt_type = OPT_UV_TYPE
    
    bpt_mask = np.zeros(len(T))
    
    ## If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    ## [NII]-BPT masks
    bpt_mask = nii_bpt * bpt_type.NII_BPT              ## All the emission lines have S/N >= 3
    bpt_mask |= sf_nii * bpt_type.NII_SF               ## [NII] - Star Forming
    bpt_mask |= agn_nii * bpt_type.NII_SY              ## [NII] - Seyfert
    bpt_mask |= liner_nii * bpt_type.NII_LINER         ## [NII] - LINER
    bpt_mask |= composite_nii * bpt_type.NII_COMP      ## [NII] - Composite
   
    ## [SII]-BPT masks
    bpt_mask |= sii_bpt * bpt_type.SII_BPT             ## All the emission lines have S/N >= 3
    bpt_mask |= sf_sii * bpt_type.SII_SF               ## [SII] - Star Forming
    bpt_mask |= agn_sii * bpt_type.SII_SY              ## [SII] - Seyfert
    bpt_mask |= liner_sii * bpt_type.SII_LINER         ## [SII] - LINER

    ## [OI]-BPT masks
    bpt_mask |= oi_bpt * bpt_type.OI_BPT               ## Except [OI] - other emission lines have S/N >= 3
    bpt_mask |= sf_oi * bpt_type.OI_SF                 ## [OI] - Star Forming
    bpt_mask |= agn_oi * bpt_type.OI_SY                ## [OI] - Seyfert
    bpt_mask |= liner_oi * bpt_type.OI_LINER           ## [OI] - LINER
    
    bptmask_column = Column(bpt_mask, name = 'OPT_UV_TYPE')
    if 'OPT_UV_TYPE' in T.columns:
        T['OPT_UV_TYPE']|=bpt_mask
    else:
        T.add_column(bptmask_column)
    
    return T
