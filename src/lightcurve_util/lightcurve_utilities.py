import numpy as np
from six import moves

class SparseLightCurveError(Exception):
    """Indicates light curve with too few points in chosen time range."""
    pass


def median_filter(x, y, num_bins, bin_width=None, x_min=None, x_max=None):
  """Computes the median y-value in uniform intervals (bins) along the x-axis.

  The interval [x_min, x_max) is divided into num_bins uniformly spaced
  intervals of width bin_width. The value computed for each bin is the median
  of all y-values whose corresponding x-value is in the interval. Bins are overlapping if bin_width > bin_spacing.

  NOTE: x must be sorted in ascending order or the results will be incorrect.

  Args:
    x: 1D array of x-coordinates sorted in ascending order. Must have at least 2
        elements, and all elements cannot be the same value.
    y: 1D array of y-coordinates with the same size as x.
    num_bins: The number of intervals to divide the x-axis into. Must be at
        least 2.
    bin_width: The width of each bin on the x-axis. Must be positive, and less
        than x_max - x_min. Defaults to (x_max - x_min) / num_bins.
    x_min: The inclusive leftmost value to consider on the x-axis. Must be less
        than or equal to the largest value of x. Defaults to min(x).
    x_max: The exclusive rightmost value to consider on the x-axis. Must be
        greater than x_min. Defaults to max(x).

  Returns:
    1D NumPy array of size num_bins containing the median y-values of uniformly
    spaced bins on the x-axis.

  Raises:
    ValueError: If an argument has an inappropriate value.
    SparseLightCurveError: If light curve has too few points within given window.
  """
  if num_bins < 2:
    raise ValueError("num_bins must be at least 2. Got: %d" % num_bins)

  # Validate the lengths of x and y.
  x_len = len(x)
  if x_len < 2:
    raise SparseLightCurveError("len(x) must be at least 2. Got: %s" % x_len)
  if x_len != len(y):
    raise ValueError("len(x) (got: %d) must equal len(y) (got: %d)" % (x_len,
                                                                       len(y)))

  # Validate x_min and x_max.
  x_min = x_min if x_min is not None else x[0]
  x_max = x_max if x_max is not None else x[-1]
  if x_min >= x_max:
    raise ValueError("x_min (got: %d) must be less than x_max (got: %d)" %
                     (x_min, x_max))

  # This is unhelpful for sparse light curves. Use more specific error below
  # if x_min > x[-1]:
  #   raise ValueError(
  #       "x_min (got: %d) must be less than or equal to the largest value of x "
  #       "(got: %d)" % (x_min, x[-1]))

  # Drop light curves with no/few points in time range considered, or too little coverage in time
  in_range = np.where((x >= x_min) & (x <= x_max))[0]
  if (len(in_range) < 5) or (x[-1] - x[0] < (x_max - x_min) / 2 ):
    raise SparseLightCurveError('Too few points near transit')

  # Validate bin_width.
  bin_width = bin_width if bin_width is not None else (x_max - x_min) / num_bins
  if bin_width <= 0:
    raise ValueError("bin_width must be positive. Got: %d" % bin_width)
  if bin_width >= x_max - x_min:
    raise ValueError(
        "bin_width (got: %d) must be less than x_max - x_min (got: %d)" %
        (bin_width, x_max - x_min))

  bin_spacing = (x_max - x_min - bin_width) / (num_bins - 1)

  # # Bins with no y-values will fall back to the global median. - Don't do this for sparse light curves
  # result = np.repeat(np.median(y), num_bins)
  result = np.repeat(np.nan, num_bins)
  # For sparse light curves, fill empty bins with NaN to be interpolated over later.

  # Find the first element of x >= x_min. This loop is guaranteed to produce
  # a valid index because we know that x_min <= x[-1].
  x_start = 0
  while x[x_start] < x_min:
    x_start += 1

  # The bin at index i is the median of all elements y[j] such that
  # bin_min <= x[j] < bin_max, where bin_min and bin_max are the endpoints of
  # bin i.
  bin_min = x_min  # Left endpoint of the current bin.
  bin_max = x_min + bin_width  # Right endpoint of the current bin.
  j_start = x_start  # Inclusive left index of the current bin.
  j_end = x_start  # Exclusive end index of the current bin.

  for i in moves.range(num_bins):
    # Move j_start to the first index of x >= bin_min.
    while j_start < x_len and x[j_start] < bin_min:
      j_start += 1

    # Move j_end to the first index of x >= bin_max (exclusive end index).
    while j_end < x_len and x[j_end] < bin_max:
      j_end += 1

    if j_end > j_start:
      # Compute and insert the median bin value.
      result[i] = np.median(y[j_start:j_end])

    # Advance the bin.
    bin_min += bin_spacing
    bin_max += bin_spacing

  result = fill_empty_bin(result)
  return result


def fill_empty_bin(y):
  """Fill empty bins by interpolating between adjacent bins.

  :param y: 1D array of y-coordinates with the same size as x. Empty bins should have NaN values.
  :return: same as y, but with NaNs replaced with interpolated values.
  """

  i = 0
  while i < len(y):
    if np.isnan(y[i]):
      left = i-1
      right = i+1
      # Find nearest non-NaN values on both sides
      while left >= 0 and np.isnan(y[left]):
        left -= 1
      while right < len(y) and np.isnan(y[right]):
        right += 1
      if left >= 0 and right < len(y):
        slope = (y[right] - y[left]) / (right - left)
        for j in moves.range(left + 1, right):
          y[j] = y[left] + slope*(j - left)
      elif left < 0 and right < len(y):
        y[:right] = y[right]
      elif left >= 0 and right == len(y):
        y[left+1:] = y[left]
      else:
        raise ValueError('Light curve consists only of invalid values')
    i += 1
  return y

def phase_fold_algorithm(time, period, t0):
  """Creates a phase-folded time vector.

  result[i] is the unique number in [-period / 2, period / 2)
  such that result[i] = time[i] - t0 + k_i * period, for some integer k_i.

  Args:
    time: 1D numpy array of time values.
    period: A positive real scalar; the period to fold over.
    t0: The center of the resulting folded vector; this value is mapped to 0.

  Returns:
    A 1D numpy array.
  """
  half_period = period / 2
  result = np.mod(time + (half_period - t0), period)
  result -= half_period
  return result

def split(all_time, all_flux, gap_width=0.75):
  """Splits a light curve on discontinuities (gaps).

  This function accepts a light curve that is either a single segment, or is
  piecewise defined (e.g. split by quarter breaks or gaps in the in the data).

  Args:
    all_time: Numpy array or sequence of numpy arrays; each is a sequence of
        time values.
    all_flux: Numpy array or sequence of numpy arrays; each is a sequence of
        flux values of the corresponding time array.
    gap_width: Minimum gap size (in time units) for a split.

  Returns:
    out_time: List of numpy arrays; the split time arrays.
    out_flux: List of numpy arrays; the split flux arrays.
  """
  # Handle single-segment inputs.
  if isinstance(all_time, np.ndarray) and all_time.ndim == 1:
    all_time = [all_time]
    all_flux = [all_flux]

  out_time = []
  out_flux = []
  for time, flux in zip(all_time, all_flux):
    start = 0
    for end in moves.range(1, len(time) + 1):
      # Choose the largest endpoint such that time[start:end] has no gaps.
      if end == len(time) or time[end] - time[end - 1] > gap_width:
        out_time.append(time[start:end])
        out_flux.append(flux[start:end])
        start = end

  return out_time, out_flux


def remove_events(all_time, all_flux, events, width_factor=1.0):
  """Removes events from a light curve.

  This function accepts either a single-segment or piecewise-defined light
  curve (e.g. one that is split by quarter breaks or gaps in the in the data).

  Args:
    all_time: Numpy array or sequence of numpy arrays; each is a sequence of
        time values.
    all_flux: Numpy array or sequence of numpy arrays; each is a sequence of
        flux values of the corresponding time array.
    events: List of Event objects to remove.
    width_factor: Fractional multiplier of the duration of each event to remove.

  Returns:
    output_time: Numpy array or list of numpy arrays; the time arrays with
        events removed.
    output_flux: Numpy array or list of numpy arrays; the flux arrays with
        events removed.
  """
  # Handle single-segment inputs.
  if isinstance(all_time, np.ndarray) and all_time.ndim == 1:
    all_time = [all_time]
    all_flux = [all_flux]
    single_segment = True
  else:
    single_segment = False

  output_time = []
  output_flux = []
  for time, flux in zip(all_time, all_flux):
    mask = np.ones_like(time, dtype=np.bool)
    for event in events:
      transit_dist = np.abs(phase_fold_algorithm(time, event.period, event.t0))
      mask = np.logical_and(mask,
                            transit_dist > 0.5 * width_factor * event.duration)

    if single_segment:
      output_time = time[mask]
      output_flux = flux[mask]
    else:
      output_time.append(time[mask])
      output_flux.append(flux[mask])

  return output_time, output_flux


def interpolate_masked_spline(all_time, all_masked_time, all_masked_spline):
  """Linearly interpolates spline values across masked points.

  Args:
    all_time: List of numpy arrays; each is a sequence of time values.
    all_masked_time: List of numpy arrays; each is a sequence of time values
        with some values missing (masked).
    all_masked_spline: List of numpy arrays; the masked spline values
        corresponding to all_masked_time.

  Returns:
    interp_spline: List of numpy arrays; each is the masked spline with missing
        points linearly interpolated.
  """
  interp_spline = []
  for time, masked_time, masked_spline in zip(
      all_time, all_masked_time, all_masked_spline):
    if masked_time.size:
      interp_spline.append(np.interp(time, masked_time, masked_spline))
    else:
      interp_spline.append(np.array([np.nan] * len(time)))
  return interp_spline


def count_transit_points(time, event):
  """Computes the number of points in each transit of a given event.

  Args:
    time: Sorted numpy array of time values.
    event: An Event object.

  Returns:
    A numpy array containing the number of time points "in transit" for each
    transit occurring between the first and last time values.

  Raises:
    ValueError: If there are more than 10**6 transits.
  """
  t_min = np.min(time)
  t_max = np.max(time)

  # Tiny periods or erroneous time values could make this loop take forever.
  if (t_max - t_min) / event.period > 10**6:
    raise ValueError(
        "Too many transits! Time range is [%.2f, %.2f] and period is %.2e." %
        (t_min, t_max, event.period))

  # Make sure t0 is in [t_min, t_min + period).
  t0 = np.mod(event.t0 - t_min, event.period) + t_min

  # Prepare loop variables.
  points_in_transit = []
  i, j = 0, 0

  for transit_midpoint in np.arange(t0, t_max, event.period):
    transit_begin = transit_midpoint - event.duration / 2
    transit_end = transit_midpoint + event.duration / 2

    # Move time[i] to the first point >= transit_begin.
    while time[i] < transit_begin:
      # transit_begin is guaranteed to be < np.max(t) (provided duration >= 0).
      # Therefore, i cannot go out of range.
      i += 1

    # Move time[j] to the first point > transit_end.
    while time[j] <= transit_end:
      j += 1
      # j went out of range. We're finished.
      if j >= len(time):
        break

    # The points in the current transit duration are precisely time[i:j].
    # Since j is an exclusive index, there are exactly j-i points in transit.
    points_in_transit.append(j - i)

  return np.array(points_in_transit)