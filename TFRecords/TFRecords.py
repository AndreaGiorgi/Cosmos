import tensorflow as tf 

def create_tfrecords_index(dataset):
    index = None
    #TODO
    return index


def open_tfrecord():
    raw_dataset = tf.data.TFRecordDataset("TFRecords\\test-00000-of-00001")
    for raw_record in raw_dataset.take(2):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)
    
if __name__ == '__main__':
    open_tfrecord()