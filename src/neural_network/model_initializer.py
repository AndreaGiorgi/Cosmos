import tensorflow as tf
import numpy as np

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv1D, MaxPool1D, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import lecun_normal, RandomNormal, Constant
import sklearn
from matplotlib import pyplot

from neural_network_util import dataset_postprocess

def _mlp_test_eval(history, model, x_train, y_train, x_test, y_test, y_pred, y_pred_train):

    #results = list()
    #results.append(scores)
    #pyplot.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
    #pyplot.show()
    return

def _mlp_builder(config):

    initializer = lecun_normal()
    inputs = Input(shape=(int(config.input_dim),), name = 'inputs')
    x = BatchNormalization(momentum=0.95,epsilon=0.005,beta_initializer=RandomNormal(mean=0.0, stddev=0.05),gamma_initializer=Constant(value=0.9))(inputs)
    for i in range(1, config.layers_num):
        x = Dense(config.units, activation = config.activation, name = "dense_" + str(i), kernel_initializer=initializer)(x)
    x = BatchNormalization(momentum=0.95,epsilon=0.005,beta_initializer=RandomNormal(mean=0.0, stddev=0.05),gamma_initializer=Constant(value=0.9))(x)
    outputs = Dense(int(config.output_dim), activation=config.output_act, name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])

    return model

def _dcnn_builder(config):
    #? shape 1
    #? Conv1D dato che i dati sono una time series, Conv2D è ideale per immagini
    #? MaxPool
    #? Repeat conv-pool block for x times
    #? AVG POOL at the end? maybe it depends by AUC results (try max and avg)

    inputs = Input(shape=(int(config.input_dim),), name='inputs')
    net = tf.expand_dims(inputs, -1)
    initializer = lecun_normal()

    x = BatchNormalization(momentum=0.95,epsilon=0.005,beta_initializer=RandomNormal(mean=0.0, stddev=0.05),gamma_initializer=Constant(value=0.9))(net)
    for i in range(config.layers_num):
        num_filters = int(config.num_filters)
        for j in range(3):
            x = Conv1D(filters=num_filters, kernel_size=int(config.kernel), padding=config.padding, activation=str(config.activation), kernel_initializer=initializer)(x)
        x = MaxPool1D(pool_size=int(config.pool), strides=int(config.stride))(x)
        x = BatchNormalization()(x)
        x = Dropout(config.dropout_cnn)(x)
    x = Flatten()(x)
    x = BatchNormalization(momentum=0.95,epsilon=0.005,beta_initializer=RandomNormal(mean=0.0, stddev=0.05),gamma_initializer=Constant(value=0.9))(x)
    for i in range(config.fc_layers_num):
        x = Dense(config.fc_units, activation=config.activation, kernel_initializer=initializer)(x)

    outputs = Dense(int(config.output_dim), activation=config.output_act, name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])

    return model


def _combined_fnn_builder(mlp, dcnn):
    models = mlp



def model_evaluation(model_type, dataset, val_dataset, test_dataset, config):

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
    kfold =  sklearn.model_selection.KFold(n_splits = 10, shuffle = True)
    for train, test in kfold.split(cv_inputs, cv_targets):
        model = None
        if model_type == 'cnn':
            model = _dcnn_builder(config)
        else:
            model = _mlp_builder(config)

        history = model.fit(cv_inputs[train], cv_targets[train],
              batch_size=config.batch_size,
              epochs=100,
              use_multiprocessing=True)

        scores = model.evaluate(cv_inputs[test], cv_targets[test], verbose=0)
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
        print('------------------------------------------------------------------------')

def model_builder():


    '''
    Coordina la costruzione dei quattro modelli che compongono il cuore dell'archiettura di cosmos.
    Rete 1: Cosmos_DCNN
    Rete 2: Cosmos_MLP_FC
    Rete 3: Comsos_Combine_Layer
    Rete 4: Cosmos_Combined_FNN

    ---------------              ----------------
      Cosmos_DCNN                  Cosmos_MLP_FC
    ---------------              ----------------
            |                             |
            |                             |
            |                             |
    ----------------------------------------------
                Cosmos_Combined_Layer
    ----------------------------------------------
                        |
                        |
                        |
    ----------------------------------------------
                Cosmos_Combined_FNN
    ----------------------------------------------
                        |
                        |
                        |
                    ----------
                      Output
                    ----------
    '''

    '''
    Rete 1: Cosmos_DCNN
        Utilizza le global/local view tramite un addestramento convoluzionale.
    '''

    cosmos_dcnn = _dcnn_builder()

    '''
    Rete 2: Cosmos_MLP_FC
        Utilizza e sfrutta i dati secondari presenti all'ìnterno del TCE, ossia quelli categorici/numerici non direttamente connessi alle global\local view.
    '''

    cosmos_ann = _mlp_builder()

    '''
    Rete 3: Cosmos_Combined_Layer
            Combina gli output delle due reti. Dato che i dati sono multi-tipo è la chiave di volta dell'intera architettura. Permette di combinarne gli output
            generando l'input per l'ultimo strato fully connected di classificazione, che utilizzerà i risultati delle reti precedenti
    '''

    '''
    Rete 4: Cosmos_Combined_FNN
            Ultimo modulo della rete complessa di cosmos. Effettua la fase di classificazione finale.
    '''

    return #modello1, modello2, modello3, modello4

def _test_build(local, lc_train_dataset, aux_train_dataset, lc_valid_dataset, aux_valid_dataset, lc_test_dataset, aux_test_dataset, config_mlp, config_cnn):
    mlp = model_evaluation('mlp', aux_train_dataset, aux_valid_dataset, aux_test_dataset, config_mlp)
    dcnn = model_evaluation('cnn', lc_train_dataset, lc_valid_dataset, lc_test_dataset, config_cnn)
    #_combined_fnn_builder(mlp, dcnn)
    return

if __name__ == '__main__':
    _test_build()