#ipython
import pandas as pd
import numpy as np
from AGNdiagnosticFunctions import NII_BPT, SII_BPT, OI_BPT, WHAN

dir = '/Users/mmezcua/Documents/Postdoc/ICE/DESI/inputs/'
input=pd.read_csv(dir+'input_stripe82.txt',header=0, delim_whitespace=True)
output=input[['#id']]

## NII BPT ##
#nii_bpt, sf_nii, agn_nii, liner_nii, composite_nii, quiescent_nii = NII_BPT(input)
output_nii = NII_BPT(input)
conditions_niibpt = [output_nii[0]==True,output_nii[2]==True,output_nii[1]==True,output_nii[3]==True,output_nii[4]==True,output_nii[5]==True]
choices_niibpt = ['NII_BPT_AV','NII_SY','NII_SF','NII_LINER','NII_COMP','NII_QUIES']
output['NIIBPT'] = np.select(conditions_niibpt, choices_niibpt, default='--')

## SII BPT ##
#sii_bpt, sf_sii, agn_sii, liner_sii, quiescent_sii = SII_BPT(input, Kewley01=True)
output_sii = SII_BPT(input, Kewley01=True)
conditions_siibpt = [output_sii[0]==True,output_sii[2]==True,output_sii[1],output_sii[3]==True,output_sii[4]==True]
choices_siibpt = ['SII_BPT_AV','SII_SY','SII_SF','SII_LINER','SII_QUIES']
output['SIIBPT'] = np.select(conditions_siibpt, choices_siibpt, default='--')

## OI BPT ##
#oi_bpt, sf_oi, agn_oi, liner_oi = OI_BPT(input, Kewley01=True)
output_oi = OI_BPT(input, Kewley01=True)
conditions_oibpt = [output_oi[0]==True,output_oi[2]==True,output_oi[1]==True,output_oi[3]==True]
choices_oibpt = ['OI_BPT_AV','OI_SY','OI_SF','OI_LINER']
output['OIBPT'] = np.select(conditions_oibpt, choices_oibpt, default='--')

## WHAN ##
#whan, whan_sf, whan_sagn, whan_wagn, whan_retired, whan_passive =  WHAN(input)
output_whan = WHAN(input)
conditions_whan = [output_whan[0]==True, output_whan[1]==True, output_whan[2]==True, output_whan[3]==True, output_whan[4]==True, output_whan[5]==True]
choices_whan = ['WHAN_AV','WHAN_SF','WHAN_SAGN','WHAN_WAGN','WHAN_RET','WHAN_PASS']
output['WHAN'] = np.select(conditions_whan, choices_whan, default='--')

#Save output table as .csv or .ascii table
dirout = '/Users/mmezcua/Documents/Postdoc/ICE/DESI/outputs/'
output.to_csv(dirout+'output.csv', header= True, index=None, sep='\t')
output.to_csv(dirout+'output.ascii', header= True, index=None, sep='\t')

