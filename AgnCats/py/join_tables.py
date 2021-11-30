"""
This module has functions that join information from different tables
"""

import numpy
from astropy.io import fits
from astropy.table import Table, Column, vstack, join

from desimodel.footprint import radec2pix

#######################################################################################################
#######################################################################################################

def join_zcat_fastspec(survey, faprgrm, specred = 'everest'):
    """
    Join the zpix* redshift catalog with the fastspec catalog for a given 'survey' and 'faprgrm'
    
    Parameters 
    ----------
    survey : str
        sv1|sv2|sv3|main
        
    faprgrm : str
        Fiber Assignment Program: dark|bright
    
    specred : str
        DESI Spectral Reduction Pipeline. Default = 'everest'
    """
    
    # Directory Paths for zcatalogs and fastspec in NERSC
    zcat_dir = f'/global/cfs/cdirs/desi/spectro/redux/{specred}/zcatalog'
    fast_dir = f'/global/cfs/cdirs/desi/spectro/fastspecfit/{specred}/catalogs'
    
    # Filenames for zcatalog and fastspec for the given survey and faprgrm
    zcat_file = f'{zcat_dir}/zpix-{survey}-{faprgrm}.fits'
    fastspec_file = f'{fast_dir}/fastspec-{specred}-{survey}-{faprgrm}.fits'
    
    # Reading in the fits files
    hzcat = fits.open(zcat_file)
    hspec = fits.open(fastspec_file)
    
    # Reading the tables from the hdus
    zcat = Table(hzcat['ZCATALOG'].data)
    tspec = Table(hspec['FASTSPEC'].data)
    mspec = Table(hspec['METADATA'].data)
    
    # zcat does not have HPXPIXEL column in everest - it will be added in fuji.
    # Adding the HPXPIXEL column here for consistency
    hpx = radec2pix(nside = 64, ra = zcat['TARGET_RA'].data, dec = zcat['TARGET_DEC'])
    hpx_col = Column(hpx, name = 'HPXPIXEL')
    zcat.add_column(hpx_col, 1)
    
    # Selecting only some columns from zcatalog
    zcat = zcat['TARGETID', 'HPXPIXEL', 'Z', 'ZERR', 'ZWARN', 'SPECTYPE', \
                'COADD_FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC']
    
    # Selecting important columns from the metadeta 
    mspec_sel = mspec['TARGETID', 'SURVEY', 'FAPRGRM', 'HPXPIXEL']
    # Joining the two fastspec tables to create a single table
    spec = join(mspec_sel, tspec, keys = 'TARGETID')
    
    # Left-joining the zcatalog and fastspec table
    res_table = join(zcat, spec, keys = ['TARGETID', 'HPXPIXEL'], join_type = 'left')  
    # Should add 'SURVEY' and 'FAPRGM' if we start from summary catalogs
    
    return (res_table)

#######################################################################################################
#######################################################################################################
