from sklearn import neural_network
from sklearn.metrics import rand_score
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

def hybrid_kfold_evaluation(model, cnn_train_dataset, cnn_val_dataset, cnn_test_dataset, snn_train_dataset, snn_val_dataset, snn_test_dataset,config):

    cnn_train_dataset = hybrid_dataset_formatter_init('cnn',cnn_train_dataset)
    cnn_val_dataset = hybrid_dataset_formatter_init('cnn', cnn_val_dataset)
    cnn_test_dataset = hybrid_dataset_formatter_init('cnn', cnn_test_dataset)
    snn_train_dataset = hybrid_dataset_formatter_init('snn',snn_train_dataset)
    snn_val_dataset = hybrid_dataset_formatter_init('snn', snn_val_dataset)
    snn_test_dataset = hybrid_dataset_formatter_init('snn', snn_test_dataset)

    cnn_dataset = pd.concat([cnn_train_dataset, cnn_val_dataset, cnn_test_dataset], ignore_index=True)
    snn_dataset = pd.concat([snn_train_dataset, snn_val_dataset, snn_test_dataset], ignore_index=True)

    print(cnn_dataset.info())
    print(snn_dataset.info())

    hybrid_dataset = cnn_dataset.merge(snn_dataset, left_on='inputs/tic_id', right_on='inputs/tic_id', suffixes=('_left', '_right'))
    hybrid_x_cnn, hybrid_x_snn, hybrid_y = hybrid_dataset_augmentation_init(hybrid_dataset)
    hybrid_x_cnn = np.stack(hybrid_x_cnn,0)
    hybrid_x_snn = np.stack(hybrid_x_snn,0)
    hybrid_y = np.stack(hybrid_y,0)

    KFold_y = []
    KFold_y_test = []
    KFold_cnn_train = []
    KFold_snn_train = []
    KFold_cnn_test = []
    KFold_snn_test = []
    acc_per_fold = []
    loss_per_fold = []

    model_dir = 'F:\Cosmos\Cosmos\src\model_checkpoint\model_hybrid' + '.h5'
    model.save(model_dir)
    kfold =  sklearn.model_selection.RepeatedKFold(n_splits = 5, n_repeats=1, random_state = 42)



            #scores = fold_model.evaluate([KFold_cnn_test, hybrid_x_snn[test]], KFold_y_test, batch_size = 64, verbose=1, use_multiprocessing = True)
            #acc_per_fold.append(scores[1] * 100)
            #loss_per_fold.append(scores[0])

    for train, test in kfold.split(hybrid_x_cnn, hybrid_y):
        KFold_cnn_train.append(hybrid_x_cnn[train])
        KFold_y.append(hybrid_y[train])
        KFold_cnn_test.append(hybrid_x_cnn[test])
        KFold_y_test.append(hybrid_y[test])


    for train, _ in kfold.split(hybrid_x_snn, hybrid_y):
        KFold_snn_train.append(hybrid_x_cnn[train])
        KFold_snn_test.append(hybrid_x_cnn[test])

    for i in range(len(KFold_cnn_train)):
        fold_model = tf.keras.models.load_model(model_dir)
        _ = fold_model.fit([KFold_cnn_train[i], KFold_snn_train[i]], KFold_y[i], batch_size=config.batch_size,epochs=25,use_multiprocessing=True)

        scores = fold_model.evaluate([KFold_cnn_test[i], KFold_snn_test[i]], KFold_y_test[i], batch_size = 64, verbose=1, use_multiprocessing = True)
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {sem(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')


def combined_model_evaluation(model, cnn_dataset, cnn_val_dataset, cnn_test_dataset, snn_dataset, snn_val_dataset, snn_test_dataset,config):

    cnn_x_train, cnn_y_train = start_dataset_postprocessing('cnn', cnn_dataset, train = True)
    cnn_x_test, cnn_y_test = start_dataset_postprocessing('cnn', cnn_test_dataset, train = False)
    ann_x_train, ann_y_train = start_dataset_postprocessing('snn', snn_dataset, train = True)
    ann_x_test, ann_y_test = start_dataset_postprocessing('snn', snn_test_dataset, train = False)

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
    x_val, y_val = start_dataset_postprocessing(model_type, val_dataset, train = True)
    x_test, y_test = start_dataset_postprocessing(model_type, test_dataset, train = False)

    kf_inputs = np.concatenate((x_train, x_test, x_val), axis = 0)
    kf_targets = np.concatenate((y_train, y_test, y_val), axis = 0)
    ct = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = 'F:\Cosmos\Cosmos\src\model_checkpoint\model_'+ str(model_type) +'_' + str(ct) + '.h5'
    model.save(model_dir)

    acc_per_fold = []
    loss_per_fold = []
    kfold =  sklearn.model_selection.RepeatedKFold(n_splits = 5, n_repeats=5)
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
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')