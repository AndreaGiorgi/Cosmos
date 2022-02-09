import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
import numpy as np

def _lc_dataset_formatter(dataset, train = True):
    df = tfds.as_dataframe(dataset)
    y = df['targets'] = df['targets'].str.get(0)
    data = df['inputs/global_view'].to_numpy()
    x = np.array(list(x for x in data))
    if train:
        over = ADASYN(random_state=42, n_jobs=-1)
        under = RandomUnderSampler(random_state=42)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        x, y = pipeline.fit_resample(x, y)

    y = to_categorical(y)

    return x, y

def _aux_dataset_formatter(dataset, train = True):

    df = tfds.as_dataframe(dataset)
    df['inputs/Duration'] = df['inputs/Duration'].str.get(0)
    df['inputs/Epoc'] = df['inputs/Epoc'].str.get(0)
    df['inputs/Tmag'] = df['inputs/Tmag'].str.get(0)
    df['inputs/Period'] = df['inputs/Period'].str.get(0)
    df['inputs/Transit_Depth'] = df['inputs/Transit_Depth'].str.get(0)
    df['targets'] = df['targets'].str.get(0)

    y = df['targets']
    x = df.drop(labels=['targets'], axis=1)

    if train:
        over = ADASYN(random_state=42, n_jobs=-1)
        under = RandomUnderSampler(random_state=42)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        x, y = pipeline.fit_resample(x, y)

    y = to_categorical(y)
    return x, y