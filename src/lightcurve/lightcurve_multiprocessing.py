import tensorflow as tf
from coordinator import etl_coordinator

class EmptyLightCurveError(Exception):
    """Indicates light curve with no points in chosen time range."""


class SparseLightCurveError(Exception):
    """Indicates light curve with too few points in chosen time range."""

def _process_tce(tce, only_local):
    """Processes the light curve for a Kepler TCE and returns an Example proto.

  Args:
    tce: Row of the input TCE table.
    only_local: Boolean for switching to AstroNet Implement and Cosmos' one

  Returns:
    A tensorflow.train.Example proto containing TCE features.

  Raises:
    IOError or general Exceptionx: If the light curve files for this Kepler ID cannot be found.
  """
    try:
        processed_tce = None
        processed_tce = etl_coordinator.start_processing_phase(tce, only_local)
    except (Exception, IOError) as e:
        print("Exception occurred: ", e)
    return processed_tce


def process_file_shard(tce_table, file_name, only_local = False):
    """Processes a single file shard.

  Args:
    tce_table: A Pandas DateFrame containing the TCEs in the shard.
    file_name: The output TFRecord file.
  """
    with tf.io.TFRecordWriter(file_name) as writer:
        for _, tce in tce_table.iterrows():
            try:
                tce_to_write = _process_tce(tce, only_local)
            except(IOError, EmptyLightCurveError, SparseLightCurveError):
                continue
            if tce_to_write is not None:
                writer.write(tce_to_write.SerializeToString())
