## ETL: Trasform
#* Lettura della lightcurve e processamento in NumPy Array 1D 
import numpy as np
import tensorflow as tf
from lightcurve_util import lighcurve_preprocess

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
            time, flux = lighcurve_preprocess.phase_fold(time, flux, tce.Period, tce.Epoc)
            if only_local_flag is False:
                global_view = lighcurve_preprocess.global_view(time. flux, tce.Period)
            local_view = lighcurve_preprocess.local_view(time, flux, tce.Period, tce.Duration)
        
    except RuntimeWarning:
        print('Too many invalid values in TIC %s', tce.tic_id)
        raise SparseLightCurveError
    
    example = tf.train.Example() #! output prototype see documentation for this shit
    
    if only_local_flag is False:
        _set_float_feature(example, "global_view", global_view)
    _set_float_feature(example, "local_view", local_view)
    
      # Set other columns.
    for col_name, value in tce.items():
        if np.issubdtype(type(value), np.integer):
            _set_int64_feature(example, col_name, [value])
        else:
            try:
                _set_float_feature(example, col_name, [float(value)])
            except ValueError:
                _set_bytes_feature(example, col_name, [value])
    
    return example
 