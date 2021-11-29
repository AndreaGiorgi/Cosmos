import tensorflow as tf 

def main():
    raw_dataset = tf.data.TFRecordDataset("TFRecords\\training_set-00000-of-00010")
    for raw_record in raw_dataset.take(2):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)
    
if __name__ == '__main__':
    main()