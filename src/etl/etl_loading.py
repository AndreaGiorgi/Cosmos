from astropy.table import Table
import numpy as np

class InvalidLightcurveData(Exception):
  """Indicates light curve data has invalid data, empty columns or different data sizes."""
  pass


def load_lightcurve_data(filename, flux_type = 'KSPSAP_FLUX'):
    """Reads time and flux measurements for a Kepler target star.

    Args:
      filename: str name of fits file containing light curve.
      flux_type: str name of flux type to fetch from fits file [SAP/KSPSAP_FLUX].
                 default is KSPSAP_FLUX.
                 " Among other information, we provide the un-detrended flux from the optimal aperture in the FITS files under the keyword SAP_FLUX, and detrended flux under KSPSAP_FLUX. 
                 We determine the optimal aperture for each target based on its TESS magnitude. The flux measurements from relatively bigger/smaller 
                 apertures are under the keywords KSPSAP_FLUX_LAG and KSPSAP_FLUX_SML, respectively. "
                 Cit: https://iopscience.iop.org/article/10.3847/2515-5172/ac2ef0

    Returns:
      time: Numpy array; the time values of the light curve.
      flux: Numpy array of flux values corresponding to the time array.
    """
    try:
      lightcurve_fits = Table.read(filename, format = 'fits')
      time = np.array(lightcurve_fits["TIME"])
      flux = np.array(lightcurve_fits[flux_type])
      quality = np.array(lightcurve_fits['QUALITY'])

      if len(time) == len(quality) == len(flux):
        pass
      else:
        raise InvalidLightcurveData

      quality_flag = np.where(np.array(lightcurve_fits['QUALITY']) == 0)
      time = time[quality_flag]
      flux = flux[quality_flag]
    except IOError:
      print("Invalid FITS data")

    return time, flux


def load_new_lightcurve_data(astroTable, flux_type = 'KSPSAP_FLUX'):
    """Reads time and flux measurements for a Kepler target star.

    Args:
      astroTable: astropy Table containing light curve.
      flux_type: str name of flux type to fetch from fits file [SAP/KSPSAP_FLUX].
                 default is KSPSAP_FLUX.
                 " Among other information, we provide the un-detrended flux from the optimal aperture in the FITS files under the keyword SAP_FLUX, and detrended flux under KSPSAP_FLUX. 
                 We determine the optimal aperture for each target based on its TESS magnitude. The flux measurements from relatively bigger/smaller 
                 apertures are under the keywords KSPSAP_FLUX_LAG and KSPSAP_FLUX_SML, respectively. "
                 Cit: https://iopscience.iop.org/article/10.3847/2515-5172/ac2ef0

    Returns:
      time: Numpy array; the time values of the light curve.
      flux: Numpy array of flux values corresponding to the time array.
    """
    try:
      time = np.array(astroTable["TIME"])
      flux = np.array(astroTable[flux_type])
      quality = np.array(astroTable['QUALITY'])

      if len(time) == len(quality) == len(flux):
        pass
      else:
        raise InvalidLightcurveData

      quality_flag = np.where(np.array(astroTable['QUALITY']) == 0)
      time = time[quality_flag]
      flux = flux[quality_flag]
    except IOError:
      print("Invalid FITS data")

    return time, flux


if __name__ == '__main__':
    print(load_lightcurve_data('ingested_data\\sector_1\\hlsp_qlp_tess_ffi_s0001-0000000114990015_tess_v01_llc.fits'))