## Raga Pucha: First draft of the function 
## Need to add detailed comments to the script
## Returns a lot of warnings because of division by zero - have to use filterwarnings after checking everything
## Version: 2021 December 21

import numpy as np
from astropy.table import Table, Column
from join_tables import join_zcat_fastspec
from emline_classification import Classify_NII_BPT, Classify_SII_BPT, Classify_OI_BPT
from desiutil.bitmask import BitMask
import yaml

########################################################################################################################################
########################################################################################################################################

def create_bpt_mask(survey, faprgrm, specred = 'everest'):
    table = join_zcat_fastspec(survey = survey, faprgrm = faprgrm, specred = specred)
    
    nii_bpt, sf_nii, agn_nii, liner_nii, composite_nii, quiescent_nii = Classify_NII_BPT(table)
    sii_bpt, sf_sii, agn_sii, liner_sii, quiescent_sii = Classify_SII_BPT(table)
    oi_bpt, sf_oi, agn_oi, liner_oi = Classify_OI_BPT(table)
    
    file = open('../Sandbox/agn_masks.yaml', 'r')
    bpt_defs = yaml.safe_load(file)
    bpt_type = BitMask('bpt_type', bpt_defs)
    
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

########################################################################################################################################
########################################################################################################################################
