## Plotting functions for AGN diagnostic diagrams

import numpy as np

###
### THE BELOW ARE  - BC 3 Nov 2024
###

###
def test_bpt_mask(tab, OPT_UV_TYPE, directory):
    '''
    BC modified from RP code May 22
    Plots a test bpt in NII, SII and OI to make sure all is working well
    This code require AGNTYPE maskbits to be present in the data

    inputs: 
    tab - table joined with FastSpecFit (or other emline cat)
    OPT_UV_TYPE - bitmask structure from yaml 
    
    outputs:
    image in directory/test_bpt.jpg
    '''
    import matplotlib.pyplot as plt
    if 'OPT_UV_TYPE' in tab.columns:
        print("Found mask")
        bptmask = tab['OPT_UV_TYPE']
    
        bpt_type = OPT_UV_TYPE

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
