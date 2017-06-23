# Roadmap for what we want to do

## Motivation
Spectral fitting of the DESI spectra will be performed in order to make measurements in addition to (just) redshift. In order we wanting to iteratively fit the stellar contimuum component and emssion line components. Note, there is almost certainly going to be a high S/N case, as well as a low S/N case.


## Inputs

**Data**:: lambda, flux, covariance

**Models**:: many options, but default to SSPs

mag but default to SSP

3800Ang - 10,000 um observed, (i.e. ~800Ang to ~2,200Ang at z=3.5, and stick together template (LRGs, ELGs, AGN/QSOs)
Construct an AGN template for mocks. 

ELGs to z=1.6, but could have selected QSOs scatted to lower-z. 

Generally will treat v. high z QSOs separately. 



## Outputs
  * Potentialy working in two regimes, high S/N outputs, Low S/N  outputs 
  * Emission line flux, FWHM, redshift
  * Reddening
  * dust
  * Stellar masses
  * SFH
  * Stellar absorption
  * AV_star
  * AV_gas
  * E(B-V)



## Iteration Algorithm

Step 1:: Continuum only, with certain lambda mask, corresponding to strong ELs. 
_Make a note of complete weirdos, e.g. any blazars!!_ ;-)

Step 2:: If continuum detection; subtract continuum.

Step 3:: Then EL fitting. These ELs would all be at a fixed redshift. ( Some emission lines can be fixed. e.g. OIII doublet, OI lines.)

_Check if ELs are dectected or not._

Step 4:: If so, subtract ELs. 

Step 5:: Fit again the contiuum, with a finer grid of models (e.g. age, _Z_) but this time without the lambda mask.

Step 6:: Assess of signifcance of ELs and of contiuum fit. 

Step 7:: If ELs well (where well is to be quantified) then width of Balmer lines and width of forbidden lines. 

Step 8:: Substract new EL fit. 

Step 9:: If broad Balmer lines fits e.g. QSO 
Fit continuum again, without masking (again) - do we include instellar absorption lines. 

Step 10:: If broad Balmer lines then do e.g. broad and narrow line fits. With multiple componments, for all lines. Really try to allow multiple kinematics. 

After `exiting' loop, re-fit simultaneously continuum and ELs, using latest fit results as best guess. At this point, this is going to be a data structure that would be an input to some galaxy properties table, 
matched to each object. 



## To Be Decided
Do we tie the EL redshifts to each other? This, probably, could be an option.

Absorption.

Do we fit selected SFHs for selected for populations. e.g. LRGs being older Stellar Pops. 

Do we fit the 3 spec-arms individually or jointly? 


## Work Still To Do
Adding Lower-luminosity AGN

Adding Type 2 QSOs



