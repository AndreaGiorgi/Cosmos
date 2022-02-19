from sklearn import neural_network
import  tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np
import sklearn
import datetime
from scipy.stats import sem
from matplotlib import pyplot
from keras import backend as K
from coordinator.neural_network_coordinator import hybrid_dataset_augmentation_init, start_dataset_postprocessing, hybrid_dataset_formatter_init


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

def hybrid_kfold_evaluation(model, cnn_train_dataset, cnn_val_dataset, cnn_test_dataset, fnn_train_dataset, fnn_val_dataset, fnn_test_dataset,config):
    #merge su tic_id tra cnn_Dataset e fnn_dataset (merge prima dataset totale cnn e dataset totale fnn)
    #shuffle
    #x_cnn = simil lc_format
    #x_fnn = simil aux_format
    #y = simil lc_format


    cnn_train_dataset = hybrid_dataset_formatter_init('cnn',cnn_train_dataset)
    cnn_val_dataset = hybrid_dataset_formatter_init('cnn', cnn_val_dataset)
    cnn_test_dataset = hybrid_dataset_formatter_init('cnn', cnn_test_dataset)
    fnn_train_dataset = hybrid_dataset_formatter_init('fnn',fnn_train_dataset)
    fnn_val_dataset = hybrid_dataset_formatter_init('fnn', fnn_val_dataset)
    fnn_test_dataset = hybrid_dataset_formatter_init('fnn', fnn_test_dataset)

    cnn_dataset = pd.concat([cnn_train_dataset, cnn_val_dataset, cnn_test_dataset], ignore_index=True)
    fnn_dataset = pd.concat([fnn_train_dataset, fnn_val_dataset, fnn_test_dataset], ignore_index=True)

    cnn_dataset = cnn_dataset.set_index(['inputs/tic_id'])
    fnn_dataset = fnn_dataset.set_index(['inputs/tic_id'])
    hybrid_dataset = cnn_dataset.join(fnn_dataset, on="inputs/tic_id", how = 'inner')
    print(hybrid_dataset.head())
    print(hybrid_dataset.info())
    hybrid_x_cnn, hybrid_x_fnn, hybrid_y = hybrid_dataset_augmentation_init(hybrid_dataset)

    KFold_y = []
    acc_per_fold = []
    loss_per_fold = []

    model_dir = 'F:\Cosmos\Cosmos\src\model_checkpoint\model_hybrid' + '.h5'
    model.save(model_dir)

    def kfold_fnn(Kfold_cnn_train, KFold_cnn_test, KFold_y, KFold_y_test):
        kfold =  sklearn.model_selection.RepeatedKFold(n_splits = 5, n_repeats=10, random_state = 17)
        for train, _ in kfold.split(hybrid_x_fnn, hybrid_y):
            fold_model = tf.keras.models.load_model(model_dir)
            _ = fold_model.fit([Kfold_cnn_train, hybrid_x_fnn[train]], KFold_y,
                batch_size=config.batch_size,
                epochs=25,
                use_multiprocessing=True)

            scores = fold_model.evaluate([KFold_cnn_test, hybrid_x_fnn[test]], KFold_y_test, batch_size = 64, verbose=1, use_multiprocessing = True)
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])

    kfold =  sklearn.model_selection.RepeatedKFold(n_splits = 5, n_repeats=10, random_state = 17)
    for train, test in kfold.split(hybrid_x_cnn, hybrid_y):
        Kfold_cnn_train = hybrid_x_cnn[train]
        KFold_y = hybrid_y[train]
        KFold_cnn_test = hybrid_x_cnn[test]
        KFold_y_test= hybrid_y[test]
        kfold_fnn(Kfold_cnn_train, KFold_cnn_test, KFold_y, KFold_y_test)

    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {sem(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')


def combined_model_evaluation(model, cnn_dataset, cnn_val_dataset, cnn_test_dataset, fnn_dataset, fnn_val_dataset, fnn_test_dataset,config):

    cnn_x_train, cnn_y_train = start_dataset_postprocessing('cnn', cnn_dataset, train = True)
    cnn_x_test, cnn_y_test = start_dataset_postprocessing('cnn', cnn_test_dataset, train = False)
    ann_x_train, ann_y_train = start_dataset_postprocessing('fnn', fnn_dataset, train = True)
    ann_x_test, ann_y_test = start_dataset_postprocessing('fnn', fnn_test_dataset, train = False)

    _ = model.fit([cnn_x_train, ann_x_train], cnn_y_train, batch_size=config.batch_size, epochs=10, use_multiprocessing=True)
    ct = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model.save('F:\Cosmos\Cosmos\src\model_checkpoint\model_hybrid'+ str(ct) + '.h5')
    loss, accuracy = model.evaluate([cnn_x_test, ann_x_test], cnn_y_test, verbose=0)
    print('------------------------------------------------------------------------')
    print('Combined Cosmos Scores:')
    print(f'> Accuracy: {accuracy * 100}')
    print(f'> Loss: {loss}')
    print('------------------------------------------------------------------------')

def model_kfold_evaluation(model_type, model, dataset, val_dataset, test_dataset, config):

    x_train, y_train = start_dataset_postprocessing(model_type, dataset, train = True)
    x_val, y_val = start_dataset_postprocessing(model_type, val_dataset, train = False)
    x_test, y_test = start_dataset_postprocessing(model_type, test_dataset, train = False)

    kf_inputs = np.concatenate((x_train, x_test, x_val), axis = 0)
    kf_targets = np.concatenate((y_train, y_test, y_val), axis = 0)
    ct = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = 'F:\Cosmos\Cosmos\src\model_checkpoint\model_'+ str(model_type) +'_' + str(ct) + '.h5'
    model.save(model_dir)

    acc_per_fold = []
    loss_per_fold = []
    kfold =  sklearn.model_selection.RepeatedKFold(n_splits = 5, n_repeats=10, random_state = 17)
    for train, test in kfold.split(kf_inputs, kf_targets):
        fold_model = tf.keras.models.load_model(model_dir)
        _ = fold_model.fit(kf_inputs[train], kf_targets[train],
              batch_size=config.batch_size,
              epochs=25,
              use_multiprocessing=True)

        scores = fold_model.evaluate(kf_inputs[test], kf_targets[test], batch_size = 64, verbose=1, use_multiprocessing = True)
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {sem(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')