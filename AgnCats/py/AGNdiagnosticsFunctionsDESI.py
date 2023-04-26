import pandas as pd
import numpy as np

##########################################################################################################
##########################################################################################################
def NII_BPT(input):
	## BPT DIAGRAM: NII Kewley ##
	#Kewley et al. 2001: starburst vs AGN classification. Solid lines in BPT
	#log10(flux_oiii_5006/flux_hbeta)=0.61/(log10(flux_nii_6583/flux_halpha)-0.47)+1.19
	#Kauffmann et al. 2003: starburst vs composites. Dashed line in BPT
	#log10(flux_oiii_5006/flux_hbeta)=0.61/(log10(flux_nii_6583/flux_halpha)-0.05)+1.3
	#Schawinsky et al. 2007: Seyferts vs LINERS
	#log10(flux_oiii_5006/flux_hbeta)=1.05*log10(flux_nii_6583/flux_halpha)+0.45
	#Law et al. 2021 proposed revised lines based on MaNGA observation (not implemented b/c similar to Ka03):
    #log10(flux_oiii_5006/flux_hbeta)=0.438/(log10(flux_nii_6583/flux_halpha)+0.023)+1.222
	log_nii_ha=np.log10(input['NII_6584_FLUX']/input['HALPHA_FLUX'])
	log_oiii_hb=np.log10(input['OIII_5007_FLUX']/input['HBETA_FLUX'])
	Kew01_nii=0.61/(log_nii_ha-0.47)+1.19
	Scha07=1.05*log_nii_ha+0.45
	Ka03=0.61/(log_nii_ha-0.05)+1.3

	snr=3
	#If ivar=0 set it to NaN to avoid infinites when computing the error:
	input['HALPHA_FLUX_IVAR']=np.where(input['HALPHA_FLUX_IVAR']==0,np.nan,input['HALPHA_FLUX_IVAR'])
	input['HBETA_FLUX_IVAR']=np.where(input['HBETA_FLUX_IVAR']==0,np.nan,input['HBETA_FLUX_IVAR'])
	input['OIII_5007_FLUX_IVAR']=np.where(input['OIII_5007_FLUX_IVAR']==0,np.nan,input['OIII_5007_FLUX_IVAR'])
	input['NII_6584_FLUX_IVAR']=np.where(input['NII_6584_FLUX_IVAR']==0,np.nan,input['NII_6584_FLUX_IVAR'])

	SNR_Ha=input['HALPHA_FLUX']*np.sqrt(input['HALPHA_FLUX_IVAR'])
	SNR_Hb=input['HBETA_FLUX']*np.sqrt(input['HBETA_FLUX_IVAR'])
	SNR_OIII=input['OIII_5007_FLUX']*np.sqrt(input['OIII_5007_FLUX_IVAR'])
	SNR_NII=input['NII_6584_FLUX']*np.sqrt(input['NII_6584_FLUX_IVAR'])
	zero_flux_nii= (input['HALPHA_FLUX']==0) | (input['HBETA_FLUX']==0) | (input['OIII_5007_FLUX']==0) | (input['NII_6584_FLUX']==0)

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

def SII_BPT(input, Kewley01=False):
	## VO87 DIAGRAM: SII ##
    #Kewley et al. 2001: starburst vs AGN classification. Solid lines in BPT
    #log10(flux_oiii_5006/flux_hbeta)=0.72/(log10(flux_sii_6716,6731/flux_halpha)-0.32)+1.30
    #Kewley et al. 2006: Seyferts vs LINERS
    #log10(flux_oiii_5006/flux_hbeta)=1.89*log10(flux_sii_6716,6731/flux_halpha)+0.76
    #Law et al. 2021 proposed revised lines based on MaNGA observation:
    #log10(flux_oiii_5006/flux_hbeta)=0.648/(log10(flux_sii_6716,6731/flux_halpha)-0.324)+1.43  #modified (+1.349 was original)

    #By default, use the Law+21 line for SF/AGN separation and the Kewley+06 line for LINER/Seyfert separation on the AGN side.
    #Optionally, can set Kewley01=True to use the Kewley+01 line instead of Law+21
	log_sii_ha=np.log10((input['SII_6716_FLUX']+input['SII_6731_FLUX'])/input['HALPHA_FLUX'])
	log_oiii_hb=np.log10(input['OIII_5007_FLUX']/input['HBETA_FLUX'])
	Kew01_sii=0.72/(log_sii_ha-0.32)+1.30
	Kew06_sii=1.89*log_sii_ha+0.76
	Law21_sii=0.648/(log_sii_ha-0.324)+1.43 #modified (+1.349 was original)
    
	if Kewley01=='True':
		line_sii = Kew01_sii
	else:
		line_sii = Law21_sii

	snr=3
	#If ivar=0 set it to NaN to avoid infinites when computing the error:
	input['HALPHA_FLUX_IVAR']=np.where(input['HALPHA_FLUX_IVAR']==0,np.nan,input['HALPHA_FLUX_IVAR'])
	input['HBETA_FLUX_IVAR']=np.where(input['HBETA_FLUX_IVAR']==0,np.nan,input['HBETA_FLUX_IVAR'])
	input['OIII_5007_FLUX_IVAR']=np.where(input['OIII_5007_FLUX_IVAR']==0,np.nan,input['OIII_5007_FLUX_IVAR'])
	input['SII_6716_FLUX_IVAR']=np.where(input['SII_6716_FLUX_IVAR']==0,np.nan,input['SII_6716_FLUX_IVAR'])
	input['SII_6731_FLUX_IVAR']=np.where(input['SII_6731_FLUX_IVAR']==0,np.nan,input['SII_6731_FLUX_IVAR'])

	SNR_Ha=input['HALPHA_FLUX']*np.sqrt(input['HALPHA_FLUX_IVAR'])
	SNR_Hb=input['HBETA_FLUX']*np.sqrt(input['HBETA_FLUX_IVAR'])
	SNR_OIII=input['OIII_5007_FLUX']*np.sqrt(input['OIII_5007_FLUX_IVAR'])
	SNR_SII=(input['SII_6716_FLUX']+input['SII_6731_FLUX'])/(1/np.sqrt(input['SII_6716_FLUX_IVAR'])+1/np.sqrt(input['SII_6731_FLUX_IVAR']))
	zero_flux_sii= (input['HALPHA_FLUX']==0) | (input['HBETA_FLUX']==0) | (input['OIII_5007_FLUX']==0) | ((input['SII_6716_FLUX']+input['SII_6731_FLUX'])==0)

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

def OI_BPT(input, Kewley01=False):
	## VO87 DIAGRAM: OI ##
    #Kewley et al. 2001: starburst vs AGN classification. Solid lines in BPT
    #log10(flux_oiii_5006/flux_hbeta)=0.73/(log10(flux_oi_6300/flux_halpha)+0.59)+1.33
    #Kewley et al. 2006: Seyferts vs LINERS
    #log10(flux_oiii_5006/flux_hbeta)=1.18*log10(flux_oi_6300/flux_halpha)+1.30
	#Law et al. 2021: 
    #log10(flux_oiii_5006/flux_hbeta)=0.884/(log10(flux_oi_6300/flux_halpha)+0.124)+1.4 #modified (original was +1.291)
 
    #By default, use the Law+21 line for SF/AGN separation and the Kewley+06 line for LINER/Seyfert separation on the AGN side.
    #Optionally, can set Kewley01=True to use the Kewley+01 line instead of Law+21
	log_oi_ha=np.log10(input['OI_6300_FLUX']/input['HALPHA_FLUX'])
	log_oiii_hb=np.log10(input['OIII_5007_FLUX']/input['HBETA_FLUX'])
	Kew01_oi=0.73/(log_oi_ha+0.59)+1.33
	Kew06_oi=1.18*log_oi_ha+1.30
	Law21_oi=0.884/(log_oi_ha+0.124)+1.4   #modified (original was +1.291)

	if Kewley01=='True':
		line_oi = Kew01_oi
	else:
		line_oi = Law21_oi

	#Keep the S/N=3 threshold for the Ha, Hb, OIII, OI emission lines and S/N=1 for OI:
	snr=3
	#If ivar=0 set it to NaN to avoid infinites when computing the error:
	input['HALPHA_FLUX_IVAR']=np.where(input['HALPHA_FLUX_IVAR']==0,np.nan,input['HALPHA_FLUX_IVAR'])
	input['HBETA_FLUX_IVAR']=np.where(input['HBETA_FLUX_IVAR']==0,np.nan,input['HBETA_FLUX_IVAR'])
	input['OIII_5007_FLUX_IVAR']=np.where(input['OIII_5007_FLUX_IVAR']==0,np.nan,input['OIII_5007_FLUX_IVAR'])
	input['OI_6300_FLUX_IVAR']=np.where(input['OI_6300_FLUX_IVAR']==0,np.nan,input['OI_6300_FLUX_IVAR'])

	SNR_Ha=input['HALPHA_FLUX']*np.sqrt(input['HALPHA_FLUX_IVAR'])
	SNR_Hb=input['HBETA_FLUX']*np.sqrt(input['HBETA_FLUX_IVAR'])
	SNR_OIII=input['OIII_5007_FLUX']*np.sqrt(input['OIII_5007_FLUX_IVAR'])
	snrOI=1
	SNR_OI=input['OI_6300_FLUX']*np.sqrt(input['OI_6300_FLUX_IVAR'])
	zero_flux_oi= (input['HALPHA_FLUX']==0) | (input['HBETA_FLUX']==0) | (input['OIII_5007_FLUX']==0) | (input['OI_6300_FLUX']==0)

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

def WHAN(input):
	## WHAN diagram (Cid Fernandes et al. 2011) ##
	#pure star-forming galaxies: log10(flux_nii_6583/flux_halpha) < -0.4 and ew_ha_6562 > 3A
	#strong AGN (e.g. Seyferts): log10(flux_nii_6583/flux_halpha) > -0.4 and ew_ha_6562 > 6A
	#weak AGN: log10(flux_nii_6583/flux_halpha) > -0.4 and 3A < ew_ha_6562 < 6A
	#retired galaxies (fake AGN, i.e. galaxies that have stopped forming stars and are ionized by their hot low-mass evolved stars): 0.5 A < ew_ha_6562 < 3A
	#passive: ew_ha_6562 < 0.5A
	ew_ha_6562=input['HALPHA_EW']
	log_nii_ha=np.log10(input['NII_6584_FLUX']/input['HALPHA_FLUX'])

	snr=3
	#If ivar=0 set it to NaN to avoid infinites when computing the error:
	input['HALPHA_FLUX_IVAR']=np.where(input['HALPHA_FLUX_IVAR']==0,np.nan,input['HALPHA_FLUX_IVAR'])
	input['NII_6584_FLUX_IVAR']=np.where(input['NII_6584_FLUX_IVAR']==0,np.nan,input['NII_6584_FLUX_IVAR'])

	SNR_Ha=input['HALPHA_FLUX']*np.sqrt(input['HALPHA_FLUX_IVAR'])
	SNR_NII=input['NII_6584_FLUX']*np.sqrt(input['NII_6584_FLUX_IVAR'])
	zero_flux_whan= (input['HALPHA_FLUX']==0) | (input['NII_6584_FLUX']==0)

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

def BLUE(input):
	## BLUE diagram from Lamareille et al (2004) & Lamareille (2010)
	#Main division between SF/AGN (eq. 1 of Lamareille 2010): log10(flux_oiii_5006/flux_hbeta) = 0.11/(log10(ew_oii_3727/ew_hb_4861) - 0.92) + 0.85
	#Division between SF and "mixed" SF/Sy2 (eq. 2 of Lamareille 2010): log10(flux_oiii_5006/flux_hbeta) > 0.3
	#Divisions for the SF-LIN/Comp overlap region (eq. 3 of Lamareille 2010): 
    #log10(flux_oiii_5006/flux_hbeta) = -(log10(ew_oii_3727/ew_hb_4861)-1.0)**2 - 0.1*log10(ew_oii_3727/ew_hb_4861) + 0.25
    #log10(flux_oiii_5006/flux_hbeta) = (log10(ew_oii_3727/ew_hb_4861)-0.2)**2 - 0.6
   	#Division between Sy2/LINER (eq. 4 of Lamareille 2010): log10(flux_oiii_5006/flux_hbeta) = 0.95*log10(ew_oii_3727/ew_hb_4861) - 0.4
	log_ewoii_ewhb = np.log10(input['OII_3726_EW']/input['HBETA_EW'])
	log_oiii_hb=np.log10(input['OIII_5007_FLUX']/input['HBETA_FLUX'])
	main_blue = 0.11/(log_ewoii_ewhb-0.92)+0.85
	eq3_blue1 = -(log_ewoii_ewhb-1.0)**2-0.1*log_ewoii_ewhb+0.25
	eq3_blue2 = (log_ewoii_ewhb-0.2)**2-0.6
	eq4_blue = 0.95*log_ewoii_ewhb - 0.4

	snr=3
	#If ivar=0 set it to NaN to avoid infinites when computing the error:
	input['HBETA_FLUX_IVAR']=np.where(input['HBETA_FLUX_IVAR']==0,np.nan,input['HBETA_FLUX_IVAR'])
	input['OIII_5007_FLUX_IVAR']=np.where(input['OIII_5007_FLUX_IVAR']==0,np.nan,input['OIII_5007_FLUX_IVAR'])

	SNR_Hb=input['HBETA_FLUX']*np.sqrt(input['HBETA_FLUX_IVAR'])
	SNR_OIII=input['OIII_5007_FLUX']*np.sqrt(input['OIII_5007_FLUX_IVAR'])
	snrOII=1
	SNR_OII=input['OII_3726_FLUX']*np.sqrt(input['OII_3726_FLUX_IVAR'])
	zero_flux_blue= (input['HBETA_FLUX']==0) | (input['OIII_5007_FLUX']==0) | (input['OII_3726_FLUX']==0)

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

def HeII_BPT(input):
	## BPT DIAGRAM Shirazi & Brinchmann 2012 ##
	#log10(flux_heii_4685/flux_hbeta)=-1.22+1/(8.92*log10(flux_nii_6583/flux_halpha)+1.32)
	log_nii_ha=np.log10(input['NII_6584_FLUX']/input['HALPHA_FLUX'])
	log_heii_hb=np.log10(input['HEII_4686_FLUX']/input['HBETA_FLUX'])
	Shir12=-1.22+1/(8.92*log_nii_ha+1.32)
	
	snr=3
	#If ivar=0 set it to NaN to avoid infinites when computing the error:
	input['HALPHA_FLUX_IVAR']=np.where(input['HALPHA_FLUX_IVAR']==0,np.nan,input['HALPHA_FLUX_IVAR'])
	input['HBETA_FLUX_IVAR']=np.where(input['HBETA_FLUX_IVAR']==0,np.nan,input['HBETA_FLUX_IVAR'])
	input['HEII_4686_FLUX_IVAR']=np.where(input['HEII_4686_FLUX_IVAR']==0,np.nan,input['HEII_4686_FLUX_IVAR'])
	input['NII_6584_FLUX_IVAR']=np.where(input['NII_6584_FLUX_IVAR']==0,np.nan,input['NII_6584_FLUX_IVAR'])

	SNR_Ha=input['HALPHA_FLUX']*np.sqrt(input['HALPHA_FLUX_IVAR'])
	SNR_Hb=input['HBETA_FLUX']*np.sqrt(input['HBETA_FLUX_IVAR'])
	SNR_HeII=input['HEII_4686_FLUX']*np.sqrt(input['HEII_4686_FLUX_IVAR'])
	SNR_NII=input['NII_6584_FLUX']*np.sqrt(input['NII_6584_FLUX_IVAR'])
	zero_flux_heii= (input['HALPHA_FLUX']==0) | (input['HBETA_FLUX']==0) | (input['HEII_4686_FLUX']==0) | (input['NII_6584_FLUX']==0)

	## HeII-BPT is available (All lines SNR >= 3)
	heii_bpt = (SNR_Ha >= snr) & (SNR_Hb >= snr) & (SNR_HeII >= snr) & (SNR_NII >= snr) & (~zero_flux_heii)

	## HeII-AGN, SF
	agn_heii=(heii_bpt) & log_heii_hb>=Shir12
	sf_heii=(heii_bpt) & ~agn_heii

	return (heii_bpt, agn_heii, sf_nii)
##########################################################################################################
##########################################################################################################

def NeV(input):
	## NeV diagnostic ##
	#Detection of [NeV]λ3426 implies hard radiation with photon energies above 96.6 eV, indicating AGN	
	snr=2.5
	#If ivar=0 set it to NaN to avoid infinites when computing the error:
	input['NEV_3426_FLUX']=np.where(input['NEV_3426_FLUX_IVAR']==0,np.nan,input['NEV_3426_FLUX_IVAR'])

	SNR_NeV=input['NEV_3426_FLUX']*np.sqrt(input['NEV_3426_FLUX_IVAR'])
	zero_flux_nev= input['NEV_3426_FLUX']==0

	## NeV diagnostic is available if flux is not zero
	nev = (~zero_flux_nev)

	## NeV-AGN, SF
	agn_nev= (nev) & (SNR_NeV >= snr)
	sf_nev= (nev) & ~agn_nev

	return (agn_nev, sf_nev)
##########################################################################################################
##########################################################################################################

