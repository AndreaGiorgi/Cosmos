from keras import backend as K
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
import numpy as np
import sklearn

from neural_network_util import dataset_postprocess

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def results_visualization(acc_per_fold):

    acc_df = pd.DataFrame(acc_per_fold, columns=[str(i) for i in range(1,11)])
    sns.set_theme(style="whitegrid")
    ax = sns.boxplot(data=acc_df)
    ax.set(ylabel='accuracy')
    pyplot.show()

def combined_model_evaluation(model, cnn_dataset, cnn_val_dataset, cnn_test_dataset, fnn_dataset, fnn_val_dataset, fnn_test_dataset,config):

    cnn_x_train, cnn_y_train = dataset_postprocess._lc_dataset_formatter(cnn_dataset)
    #cnn_x_val, cnn_y_val = dataset_postprocess._lc_dataset_formatter(cnn_val_dataset, train = False)
    cnn_x_test, cnn_y_test = dataset_postprocess._lc_dataset_formatter(cnn_test_dataset, train=False)
    ann_x_train, ann_y_train = dataset_postprocess._aux_dataset_formatter(fnn_dataset)
    #ann_x_val, ann_y_val = dataset_postprocess._aux_dataset_formatter(fnn_val_dataset)
    ann_x_test, ann_y_test = dataset_postprocess._aux_dataset_formatter(fnn_test_dataset, False)

    #cv_inputs = np.concatenate((cnn_x_train, cnn_x_test, cnn_x_val), axis = 0)
    #cv_targets = np.concatenate((cnn_y_train, cnn_y_test, cnn_y_val), axis = 0)

    #ann_inputs = np.concatenate((ann_x_train, ann_x_test, ann_x_val), axis = 0)
    #ann_targets = np.concatenate((ann_y_train, ann_y_test, ann_y_val), axis = 0)

    _ = model.fit([cnn_x_train, ann_x_train], cnn_y_train, batch_size=config.batch_size, epochs=10, use_multiprocessing=True)

    model.save('model.h5')
    loss, accuracy = model.evaluate([cnn_x_test,ann_x_test], cnn_y_test, verbose=0)
    print('------------------------------------------------------------------------')
    print('Combined Cosmos Scores:')
    print(f'> Accuracy: {accuracy * 100}')
    print(f'> Loss: {loss}')
    print('------------------------------------------------------------------------')

def model_kfold_evaluation(model_type, model, dataset, val_dataset, test_dataset, config):

    if model_type == 'cnn':
        x_train, y_train = dataset_postprocess._lc_dataset_formatter(dataset)
        x_val, y_val = dataset_postprocess._lc_dataset_formatter(val_dataset, train = False)
        x_test, y_test = dataset_postprocess._lc_dataset_formatter(test_dataset, train=False)
    else:
        x_train, y_train = dataset_postprocess._aux_dataset_formatter(dataset)
        x_val, y_val = dataset_postprocess._aux_dataset_formatter(val_dataset)
        x_test, y_test = dataset_postprocess._aux_dataset_formatter(test_dataset, False)

    cv_inputs = np.concatenate((x_train, x_test, x_val), axis = 0)
    cv_targets = np.concatenate((y_train, y_test, y_val), axis = 0)

    acc_per_fold = []
    loss_per_fold = []
    original_model = model
    kfold =  sklearn.model_selection.KFold(n_splits = 10, shuffle = True)
    for train, test in kfold.split(cv_inputs, cv_targets):
        fold_model = original_model
        _ = fold_model.fit(cv_inputs[train], cv_targets[train],
              batch_size=config.batch_size,
              epochs=25,
              use_multiprocessing=True)

        scores = fold_model.evaluate(cv_inputs[test], cv_targets[test], verbose=0)
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')