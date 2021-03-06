import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import tensorflow as tf
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd


def hybrid_dataset_augmentation(df):

    hybrid_y = df['targets_left']
    hybrid_x_cnn = df['inputs/global_view']
    hybrid_x_snn = df.drop(labels=['inputs/tic_id', 'inputs/global_view','targets_left', 'targets_right'], axis=1)

    hybrid_x_cnn = np.array(list(x for x in hybrid_x_cnn))
    hybrid_y = hybrid_y.to_numpy()
    hybrid_x_snn = hybrid_x_snn.to_numpy()

    hybrid_y = tf.keras.utils.to_categorical(hybrid_y)

    return hybrid_x_cnn, hybrid_x_snn, hybrid_y

def hybrid_dataset_formatter(type, dataset):

    df = tfds.as_dataframe(dataset)

    if type == 'cnn':
        df['inputs/tic_id'] = df['inputs/tic_id'].str.get(0)
        df['inputs/global_view'] = df['inputs/global_view'].to_numpy()
        df['targets'] = df['targets'].str.get(0)
    else:
        df['inputs/tic_id'] = df['inputs/tic_id'].str.get(0)
        df['inputs/Duration'] = df['inputs/Duration'].str.get(0)
        df['inputs/Epoc'] = df['inputs/Epoc'].str.get(0)
        df['inputs/Tmag'] = df['inputs/Tmag'].str.get(0)
        df['inputs/Period'] = df['inputs/Period'].str.get(0)
        df['inputs/Transit_Depth'] = df['inputs/Transit_Depth'].str.get(0)
        df['targets'] = df['targets'].str.get(0)

    return df


def _lc_dataset_formatter(dataset, train = True):

    df = tfds.as_dataframe(dataset)
    df['targets'] = df['targets'].str.get(0)
    y = df['targets']
    data = df['inputs/global_view'].to_numpy()
    x = np.array(list(x for x in data))
    if train:
        over = ADASYN()
        under = RandomUnderSampler()
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        x, y = pipeline.fit_resample(x, y)
    else:
        over = ADASYN(n_neighbors = 2)
        steps = [('o', over)]
        pipeline = Pipeline(steps=steps)
        x, y = pipeline.fit_resample(x, y)

    y = tf.keras.utils.to_categorical(y)

    return x, y

def _aux_dataset_formatter(dataset, train = True):

    df = tfds.as_dataframe(dataset)
    df['inputs/Duration'] = df['inputs/Duration'].str.get(0)
    df['inputs/Epoc'] = df['inputs/Epoc'].str.get(0)
    df['inputs/Tmag'] = df['inputs/Tmag'].str.get(0)
    df['inputs/Period'] = df['inputs/Period'].str.get(0)
    df['inputs/Transit_Depth'] = df['inputs/Transit_Depth'].str.get(0)
    df['targets'] = df['targets'].str.get(0)

    print(df['targets'].value_counts())

    y = df['targets']
    df = df.drop(labels=['inputs/tic_id', 'targets'], axis=1)
    x = df

    if train:
        over = ADASYN()
        under = RandomUnderSampler()
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        x, y = pipeline.fit_resample(x, y)
    else:
        over = ADASYN(n_neighbors = 2)
        steps = [('o', over)]
        pipeline = Pipeline(steps=steps)
        x, y = pipeline.fit_resample(x, y)

    y = tf.keras.utils.to_categorical(y)

    return x, y