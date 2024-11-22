## Raga Pucha: First draft of the function 
## Need to add detailed comments to the script
## Returns a lot of warnings because of division by zero - have to use filterwarnings after checking everything
## Version: 2021 December 21

## Edited: B. Canning 24th April 2022

import numpy as np
from astropy.table import Table, Column
from join_tables import join_zcat_fastspec
from emline_classification import Classify_NII_BPT, Classify_SII_BPT, Classify_OI_BPT
from desiutil.bitmask import BitMask
import yaml

########################################################################################################################################
########################################################################################################################################

def create_bpt_mask(survey, faprgrm, yaml_file, specred = 'fugi'):
    table = join_zcat_fastspec(survey = survey, faprgrm = faprgrm, specred = specred)
    
    nii_bpt, sf_nii, agn_nii, liner_nii, composite_nii, quiescent_nii = Classify_NII_BPT(table)
    sii_bpt, sf_sii, agn_sii, liner_sii, quiescent_sii = Classify_SII_BPT(table)
    oi_bpt, sf_oi, agn_oi, liner_oi = Classify_OI_BPT(table)
    
    file = open(yaml_file, 'r')
    bpt_defs = yaml.safe_load(file)
    bpt_type = BitMask('AGN_TYPE', bpt_defs)
    
    bpt_mask = np.zeros(len(table))
    
    ## If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    ## [NII]-BPT masks
    bpt_mask = nii_bpt.data * bpt_type.NII_BPT_AV           ## All the emission lines have S/N >= 3
    bpt_mask |= sf_nii.data * bpt_type.NII_SF               ## [NII] - Star Forming
    bpt_mask |= agn_nii.data * bpt_type.NII_SY              ## [NII] - Seyfert
    bpt_mask |= liner_nii.data * bpt_type.NII_LINER         ## [NII] - LINER
    bpt_mask |= composite_nii.data * bpt_type.NII_COMP      ## [NII] - Composite
    bpt_mask |= quiescent_nii.data * bpt_type.NII_QUIES     ## [NII] - Quiescent -- S/N one of the lines < 3
    
    ## [SII]-BPT masks
    bpt_mask |= sii_bpt.data * bpt_type.SII_BPT_AV          ## All the emission lines have S/N >= 3
    bpt_mask |= sf_sii.data * bpt_type.SII_SF               ## [SII] - Star Forming
    bpt_mask |= agn_sii.data * bpt_type.SII_SY              ## [SII] - Seyfert
    bpt_mask |= liner_sii.data * bpt_type.SII_LINER         ## [SII] - LINER
    bpt_mask |= quiescent_sii.data * bpt_type.SII_QUIES     ## [SII] - Quiescent -- S/N for one of the lines < 3

    ## [OI]-BPT masks
    bpt_mask |= oi_bpt.data * bpt_type.OI_BPT_AV            ## Except [OI] - other emission lines have S/N >= 3
    bpt_mask |= sf_oi.data * bpt_type.OI_SF                 ## [OI] - Star Forming
    bpt_mask |= agn_oi.data * bpt_type.OI_SY                ## [OI] - Seyfert
    bpt_mask |= liner_oi.data * bpt_type.OI_LINER           ## [OI] - LINER
    
    bptmask_column = Column(bpt_mask, name = 'BPT_MASK')
    
    table.add_column(bptmask_column)
    
    return (table)

def create_bpt_mask_all(tab, AGN_TYPE):
    '''
    BC modified from RP code May 22
    
    Calls emline_classification to make ratios and test against standard lines
    Makes AGN_TYPE
    
    inputs: 
    T - table joined with FastSpecFit (or other emline cat)
    AGN_TYPE - bitmask from yaml loaded with get_qso_maskbits(yaml_file)
    
    outputs:
    T - table with new column
    '''
    T=tab
    
    nii_bpt, sf_nii, agn_nii, liner_nii, composite_nii, quiescent_nii = Classify_NII_BPT(T)
    sii_bpt, sf_sii, agn_sii, liner_sii, quiescent_sii = Classify_SII_BPT(T)
    oi_bpt, sf_oi, agn_oi, liner_oi = Classify_OI_BPT(T)
    
    #file = open(yaml_file, 'r')
    #bpt_defs = yaml.safe_load(file)
    #bpt_type = BitMask('AGN_TYPE', bpt_defs)
    bpt_type = AGN_TYPE
    
    bpt_mask = np.zeros(len(T))
    
    ## If anyone of the emission line fluxes is zero, then there is no bpt_mask (bpt_mask = 0)  
    ## [NII]-BPT masks
    bpt_mask = nii_bpt.data * bpt_type.NII_BPT_AV           ## All the emission lines have S/N >= 3
    bpt_mask |= sf_nii.data * bpt_type.NII_SF               ## [NII] - Star Forming
    bpt_mask |= agn_nii.data * bpt_type.NII_SY              ## [NII] - Seyfert
    bpt_mask |= liner_nii.data * bpt_type.NII_LINER         ## [NII] - LINER
    bpt_mask |= composite_nii.data * bpt_type.NII_COMP      ## [NII] - Composite
    bpt_mask |= quiescent_nii.data * bpt_type.NII_QUIES     ## [NII] - Quiescent -- S/N one of the lines < 3
    
    ## [SII]-BPT masks
    bpt_mask |= sii_bpt.data * bpt_type.SII_BPT_AV          ## All the emission lines have S/N >= 3
    bpt_mask |= sf_sii.data * bpt_type.SII_SF               ## [SII] - Star Forming
    bpt_mask |= agn_sii.data * bpt_type.SII_SY              ## [SII] - Seyfert
    bpt_mask |= liner_sii.data * bpt_type.SII_LINER         ## [SII] - LINER
    bpt_mask |= quiescent_sii.data * bpt_type.SII_QUIES     ## [SII] - Quiescent -- S/N for one of the lines < 3

    ## [OI]-BPT masks
    bpt_mask |= oi_bpt.data * bpt_type.OI_BPT_AV            ## Except [OI] - other emission lines have S/N >= 3
    bpt_mask |= sf_oi.data * bpt_type.OI_SF                 ## [OI] - Star Forming
    bpt_mask |= agn_oi.data * bpt_type.OI_SY                ## [OI] - Seyfert
    bpt_mask |= liner_oi.data * bpt_type.OI_LINER           ## [OI] - LINER
    
    bptmask_column = Column(bpt_mask, name = 'BPT_MASK')
    if 'BPT_MASK' in T.columns:
        T['BPT_MASK']=bpt_mask
    else:
        T.add_column(bptmask_column)
    
    return T

def test_bpt_mask(tab, AGN_TYPE, directory):
    '''
    BC modified from RP code May 22
    
    Plots a test bpt to make sure all is working well
    
    '''
    import matplotlib.pyplot as plt
    #does BPT_MASK exists?
    if 'BPT_MASK' in tab.columns:
        print("Found mask")
        bptmask = tab['BPT_MASK']
    
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

########################################################################################################################################
########################################################################################################################################
