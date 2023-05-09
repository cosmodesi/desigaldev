
from astropy.table import Table
import pandas as pd
import numpy as np

#Convert the table from join_zcat_fastspec to pandas
inputTable = table.to_pandas()

#Select only those columns we are interested in
inputTable=inputTable[['#id','HALPHA_FLUX','HBETA_FLUX','OIII_5007_FLUX','NII_6584_FLUX','SII_6716_FLUX','SII_6731_FLUX','OI_6300_FLUX','OII_3726_FLUX','HALPHA_EW','HALPHA_FLUX_IVAR','HBETA_FLUX_IVAR','OIII_5007_FLUX_IVAR','NII_6584_FLUX_IVAR','SII_6716_FLUX_IVAR','SII_6731_FLUX_IVAR','OI_6300_FLUX_IVAR','OII_3726_EW','OII_3726_FLUX_IVAR']]

#Rename flux column names to those used in AGNdiagnosticFunctions.py
inputTable=inputTable.rename(columns={'HALPHA_FLUX':'flux_ha_6562','HBETA_FLUX':'flux_hb_4861','OIII_5007_FLUX':'flux_oiii_5006','NII_6584_FLUX':'flux_nii_6583','SII_6716_FLUX':'flux_sii_6716','SII_6731_FLUX':'flux_sii_6730','OI_6300_FLUX':'flux_oi_6300','HALPHA_EW':'ew_ha_6562','OII_3726_FLUX':'flux_oii_3727','OII_3726_EW':'ew_oii_3727'})

#If ivar=0 set it to NaN to avoid infinites when computing the error:
inputTable['HALPHA_FLUX_IVAR']=np.where(inputTable['HALPHA_FLUX_IVAR']==0,np.nan,inputTable['HALPHA_FLUX_IVAR'])
inputTable['HBETA_FLUX_IVAR']=np.where(inputTable['HBETA_FLUX_IVAR']==0,np.nan,inputTable['HBETA_FLUX_IVAR'])
inputTable['OIII_5007_FLUX_IVAR']=np.where(inputTable['OIII_5007_FLUX_IVAR']==0,np.nan,inputTable['OIII_5007_FLUX_IVAR'])
inputTable['NII_6584_FLUX_IVAR']=np.where(inputTable['NII_6584_FLUX_IVAR']==0,np.nan,inputTable['NII_6584_FLUX_IVAR'])
inputTable['SII_6716_FLUX_IVAR']=np.where(inputTable['SII_6716_FLUX_IVAR']==0,np.nan,inputTable['SII_6716_FLUX_IVAR'])
inputTable['SII_6731_FLUX_IVAR']=np.where(inputTable['SII_6731_FLUX_IVAR']==0,np.nan,inputTable['SII_6731_FLUX_IVAR'])
inputTable['OI_6300_FLUX_IVAR']=np.where(inputTable['OI_6300_FLUX_IVAR']==0,np.nan,inputTable['OI_6300_FLUX_IVAR'])
inputTable['OII_3726_FLUX_IVAR']=np.where(inputTable['OII_3726_FLUX_IVAR']==0,np.nan,inputTable['OII_3726_FLUX_IVAR'])

#Compute flux error from ivar
inputTable['flux_ha_6562_err']=1/np.sqrt(inputTable['HALPHA_FLUX_IVAR'])
inputTable['flux_hb_4861_err']=1/np.sqrt(inputTable['HBETA_FLUX_IVAR'])
inputTable['flux_oiii_5006_err']=1/np.sqrt(inputTable['OIII_5007_FLUX_IVAR'])
inputTable['flux_nii_6583_err']=1/np.sqrt(inputTable['NII_6584_FLUX_IVAR'])
inputTable['flux_sii_6716_err']=1/np.sqrt(inputTable['SII_6716_FLUX_IVAR'])
inputTable['flux_sii_6730_err']=1/np.sqrt(inputTable['SII_6731_FLUX_IVAR'])
inputTable['flux_oi_6300_err']==1/np.sqrt(inputTable['OI_6300_FLUX_IVAR'])
inputTable['flux_oii_3727_err']==1/np.sqrt(inputTable['OII_3726_FLUX_IVAR'])

#Drop the ivar columns
inputTable=inputTable.drop('HALPHA_FLUX_IVAR',axis=1)
inputTable=inputTable.drop('HBETA_FLUX_IVAR',axis=1)
inputTable=inputTable.drop('OIII_5007_FLUX_IVAR',axis=1)
inputTable=inputTable.drop('NII_6584_FLUX_IVAR',axis=1)
inputTable=inputTable.drop('SII_6716_FLUX_IVAR',axis=1)
inputTable=inputTable.drop('SII_6731_FLUX_IVAR',axis=1)
inputTable=inputTable.drop('OI_6300_FLUX_IVAR',axis=1)
inputTable=inputTable.drop('OII_3726_FLUX_IVAR',axis=1)

inputTable.to_csv('inputTable.txt', header= True, index=None, sep='\t')
