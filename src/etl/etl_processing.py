## ETL: Trasform
#* Lettura della lightcurve e processamento in NumPy Array 1D
import numpy as np
import tensorflow as tf
from lightcurve import lighcurve_preprocess

class EmptyLightCurveError(Exception):
    """Indicates light curve with no points in chosen time range."""
    pass

class SparseLightCurveError(Exception):
    """Indicates light curve with too few points in chosen time range."""
    pass


def _set_float_feature(ex, name, value):
  """Sets the value of a float feature in a tensorflow.train.Example proto."""
  assert name not in ex.features.feature, "Duplicate feature: %s" % name
  ex.features.feature[name].float_list.value.extend([float(v) for v in value])


def _set_bytes_feature(ex, name, value):
  """Sets the value of a bytes feature in a tensorflow.train.Example proto."""
  assert name not in ex.features.feature, "Duplicate feature: %s" % name
  ex.features.feature[name].bytes_list.value.extend([
      str(v).encode("latin-1") for v in value])


def _set_int64_feature(ex, name, value):
  """Sets the value of an int64 feature in a tensorflow.train.Example proto."""
  assert name not in ex.features.feature, "Duplicate feature: %s" % name
  ex.features.feature[name].int64_list.value.extend([int(v) for v in value])


def process_lightcurve(tce, only_local_flag):
    try:
        time, flux = lighcurve_preprocess.load_lightcurve(tce.tic_id, sector=tce.Sectors)
    except (RuntimeWarning, Exception) as e:
        print('Too many invalid values in TIC %s', tce.tic_id)
        print('Loading error occured: ', e)

    try:
        time, flux = lighcurve_preprocess.phase_fold(time, flux, tce.Period, tce.Epoc)
    except Exception as e:
        print('Phase Folding error occured: ', e)

    try:
        global_view = None
        if only_local_flag is False:
            global_view = lighcurve_preprocess.global_view(time, flux, tce.Period)
        else:
            local_view = lighcurve_preprocess.local_view(time, flux, tce.Period, tce.Duration)
    except Exception as e:
        print('Global or Local view generation error: ', e)

    example = tf.train.Example()
    if only_local_flag is False:
        _set_float_feature(example, "global_view", global_view)
    else:
        _set_float_feature(example, "local_view", local_view)

    for col_name, value in tce.items():
        if np.issubdtype(type(value), np.int64):
            _set_int64_feature(example, col_name, [value])
        else:
            try:
                _set_float_feature(example, col_name, [float(value)])
            except ValueError:
                _set_bytes_feature(example, col_name, [value])

    return example