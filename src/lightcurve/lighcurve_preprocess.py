# TODO: Spiegazione del file

import numpy as np
from coordinator import etl_coordinator
from lightcurve import lightcurve_utilities
from statsmodels.robust import scale

class EmptyLightCurveError(Exception):
    """Indicates light curve with no points in chosen time range."""
    pass

def generate_view(time, flux, num_bins, bin_width, t_min, t_max, normalize=True):
  """Generates a view of a phase-folded light curve using a median filter.

  Args:
    time: 1D array of time values, phase folded and sorted in ascending order.
    flux: 1D array of flux values.
    num_bins: The number of intervals to divide the time axis into.
    bin_width: The width of each bin on the time axis.
    t_min: The inclusive leftmost value to consider on the time axis.
    t_max: The exclusive rightmost value to consider on the time axis.
    normalize: Whether to center the median at 0 and minimum value at -1.

  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  view = lightcurve_utilities.median_filter(time, flux, num_bins, bin_width, t_min, t_max)
  if normalize:
    view -= np.median(view)
    view /= np.abs(np.min(view))  # In pathological cases, min(view) is zero...

  return view


def global_view(time, flux, period, num_bins=201, bin_width_factor=1.2/201):
    """
    Args:
        time: 1D array of time values, sorted in ascending order.
        flux: 1D array of flux values.
        period: The period of the event (in days).
        num_bins: The number of intervals to divide the time axis into.
        bin_width_factor: Width of the bins, as a fraction of period.

    Returns:
        1D NumPy array of size num_bins containing the median flux values of
        uniformly spaced bins on the phase-folded time axis.
  """
    return generate_view(
      time,
      flux,
      num_bins=num_bins,
      bin_width=period * bin_width_factor,
      t_min=-period / 2,
      t_max=period / 2)


def local_view(time, flux, period, duration, num_bins=61, bin_width_factor=0.16, num_durations=2):
  """Generates a 'local view' of a phase folded light curve.
  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta
  Args:
    time: 1D array of time values, sorted in ascending order.
    flux: 1D array of flux values.
    period: The period of the event (in days).
    duration: The duration of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    bin_width_factor: Width of the bins, as a fraction of duration.
    num_durations: The number of durations to consider on either side of 0 (the
        event is assumed to be centered at 0).
  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  return generate_view(
      time,
      flux,
      num_bins=num_bins,
      bin_width=duration * bin_width_factor,
      t_min=max(-period / 2, -duration * num_durations),
      t_max=min(period / 2, duration * num_durations))


def lc_processing(lc_time, lc_flux):

    """Processes a single lightcurve info
    Args:
        lc_time: time data read from fits file
        lc_flux: flux magnitude data read from fits file

    Returns:
        all_time: processed time data
        all_flux: processed flux data
    """
    mad = scale.mad(lc_flux)
    indices = np.where(lc_flux > np.median(lc_flux) - 5*mad)
    lc_flux = lc_flux[indices]
    all_time = lc_time[indices]
    all_flux = 10.**(-(lc_flux - np.median(lc_flux))/2.5)

    return all_time, all_flux


def load_lightcurve(tic, sector):
    """Load the lightcurve from a single tic identifier and its respective sector

    Args:
        tic: TESS Identifier
        sector: observed sector

    Returns:
        all_time: processed time data
        all_flux: processed flux data
    """
    lightcurve_fits = etl_coordinator.start_ingestion_phase(tic, sector)
    if not lightcurve_fits:
      print("failed to load fits file")
      raise IOError
    lc_time, lc_flux = etl_coordinator.start_loading_phase(lightcurve_fits)

    return lc_processing(lc_time, lc_flux)


def load_new_data_lightcurve(tic, sector):
    """Load the lightcurve from a single tic identifier and its respective sector

    Args:
        tic: TESS Identifier
        sector: observed sector

    Returns:
        all_time: processed time data
        all_flux: processed flux data
    """
    lightcurve_astroTable = etl_coordinator.start_new_data_ingestion_phase(tic, sector)
    if not lightcurve_astroTable:
      print("failed to download {tic} fits file".format(tic))
      raise IOError
    lc_time, lc_flux = etl_coordinator.start_new_data_loading_phase(lightcurve_astroTable)

    return lc_processing(lc_time, lc_flux)


def phase_fold(time, flux, period, t0):

    time = lightcurve_utilities.phase_fold_algorithm(time, period, t0)

    # Sort by ascending time
    sorted_element = np.argsort(time)
    time = time[sorted_element]
    flux = flux[sorted_element]

    return time, flux
