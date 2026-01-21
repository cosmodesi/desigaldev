## This script takes the redshift catalog for Loa and splits it into
## subfiles for survey, program, and (where necessary) healpix
## for consistency with fastspecfit results
## (/global/cfs/cdirs/desi/vac/dr2/fastspecfit/loa/v1.0/catalogs).

## Note that although the separate zpix-survey-program files are also created,
## we split the summary catalog in order to keep only the primary result
## for any target observed in multiple surveys/programs.

## It will overwite previously created split files.

## Authors:
# Claire Greenwell (Durham), Stephanie Juneau (NOIRLab)

###################

# module imports
import numpy as np

import time

from astropy.table import Table

from fitsio import FITS

###################
# Start timer

t0 = time.time()

# in testing took ~8-9 minutes

###################
# Define filepaths

# data team generates zcatalog for all targets at:
path_zcat = '/global/cfs/cdirs/desi/spectro/redux/loa/zcatalog/v1/'
fname_zcat_all = f'{path_zcat}/zall-pix-loa.fits'

# output files from this split for use with AGN/Gal Classification VAC
# are saved at:
path_gqp = '/global/cfs/cdirs/desi/science/gqp/agncatalog/zpix_nside1/loa/v1/'
# path_gqp = '/pscratch/sd/c/clg/agngal_testdata/' # for testing

###################
# List surveys and programs to process

survey_list = ['main', 'special', 'cmx', 'sv1', 'sv2', 'sv3']
program_list = ['bright', 'dark', 'backup', 'other']

# make into single lists to cover all combinations
survey_list_long = np.sort(survey_list*len(program_list))
program_list_long = program_list*len(survey_list)

###################
# Settings for healpix splitting

# survey/program combinations that will be split further by healpix
survey_programs_to_split_hpix = ['main-bright', 'main-dark']

# batch the healpixels
batch_size = 4096

# NSIDE=1 healpixels
pixs = np.arange(0,12)

###################
# Columns to retrieve from file

zcat_cols=[
        'TARGETID', 'HEALPIX', 'Z', 'ZERR', 'ZWARN', 'SPECTYPE',
        'COADD_FIBERSTATUS','TARGET_RA','TARGET_DEC', 'OBJTYPE',
        'MIN_MJD','MEAN_MJD','MAX_MJD','COADD_NUMEXP', 'COADD_EXPTIME',
        'TSNR2_LRG', 'DESI_TARGET', 'SCND_TARGET', 'BGS_TARGET',
        'CMX_TARGET', 'SURVEY', 'PROGRAM', 'HEALPIX', 'ZCAT_NSPEC',
        'SV_NSPEC', 'MAIN_NSPEC', 'SV1_DESI_TARGET', 'SV2_DESI_TARGET',
        'SV3_DESI_TARGET', 'SV1_BGS_TARGET', 'SV2_BGS_TARGET', 'SV3_BGS_TARGET',
        'SV1_SCND_TARGET', 'SV2_SCND_TARGET', 'SV3_SCND_TARGET',
        'MAIN_PRIMARY', 'SV_PRIMARY', 'ZCAT_PRIMARY', 'RELEASE',
        'BRICKNAME', 'BRICKID', 'BRICK_OBJID'
          ]

###################
# Import full zcatalog, buffered in groups of 1000 rows to save memory
# Iterate through survey-program combinations, saving the subsets

with FITS(fname_zcat_all, iter_row_buffer=1000) as fits: # actually maybe this isn't necessary??
# with FITS(fname_zcat_all) as fits:
    
    # zcatalog file
    # t_zcat = Table(
    #     fits['ZCATALOG'].read(rows=range(20000000, 25000000), columns=zcat_cols)
    #               ) # for testing: use subset of rows only
    t_zcat = Table(fits['ZCATALOG'].read(columns=zcat_cols)) # real option
    
    # number of rows in raw file
    N_init = len(t_zcat)
    print(f'number in zcatalog: N={N_init}')
    
    # keep only targets: OBJTYPE='TGT'
    t_zcat = t_zcat[t_zcat['OBJTYPE']=='TGT']
    N_cut = len(t_zcat)

    # report on stats from cut
    print(
        f'...after cutting on OBTYPE=TGT: N={N_cut} (fraction: {N_cut/N_init:.2e})'
         )

    N_saved = 0

    # iterate through survey-programs
    for survey, program in zip(survey_list_long, program_list_long):

        # get subset of rows within current survey-program
        flag_survey_program = np.logical_and(
            t_zcat['SURVEY']==survey, t_zcat['PROGRAM']==program
                                            )
        t_survey_program = t_zcat[flag_survey_program]

        if len(t_survey_program)>0:
            if f'{survey}-{program}' not in survey_programs_to_split_hpix:
                
                # if targets in survey-program are found and it 
                # does *not* need to be split by healpix, save subset
                print(
                    '\n', survey, program, 'totals', len(t_survey_program),
                    'targets'
                     )
                
                fname_out_survey_program = f'zpix-{survey}-{program}.fits'
                t_survey_program.write(f'{path_gqp}{fname_out_survey_program}', overwrite=True)
                N_survey_program = len(t_survey_program)
                N_saved += N_survey_program
                
                print(
                    fname_out_survey_program, f'saved ({N_survey_program} targets)'
                     )
                
            else:
                
                # if targets in survey-program are found and it
                # *does* need to be split by healpix, save subset
                print(
                    '\n', survey, program, 'totals', len(t_survey_program),
                    'targets: split by healpix needed!'
                     )

                # iterate through healpix subsets
                for pix in pixs:

                    # translate healpix subset numbers to find 'HEALPIX'
                    # column values that should be selected
                    hpx64_min = pix*batch_size
                    hpx64_max = (pix+1)*batch_size - 1

                    fname_out_survey_program_hpix = f'zpix-{survey}-{program}-nside1-hp{pix:02}.fits'
            
                    flag_isin_pix = np.logical_and(
                        t_survey_program['HEALPIX']>=hpx64_min, t_survey_program['HEALPIX']<=hpx64_max
                                                  )

                    # get targets in healpix subset and save (if any)
                    N_in_pix = len(t_survey_program[flag_isin_pix])
                    if N_in_pix>=1:
                        t_survey_program[flag_isin_pix].write(f'{path_gqp}{fname_out_survey_program_hpix}', overwrite=True)
                        N_survey_program = len(t_survey_program[flag_isin_pix])
                        print(
                            fname_out_survey_program_hpix, f'saved ({N_survey_program} targets)'
                             )
                        
                        N_saved += N_survey_program


t1 = time.time()
print(f'\nsplit completed (N saved = {N_saved}) in {(t1-t0)/60.:.2f} minutes!')