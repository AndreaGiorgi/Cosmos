import datetime
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from keras import Model, Input
from keras.layers import Dense, BatchNormalization, Dropout, Conv1D, MaxPooling1D, AveragePooling1D, Flatten
from keras.initializers import RandomNormal, Constant
from keras.layers.merge import concatenate
from neural_network.model_evaluation import combined_model_evaluation, hybrid_kfold_evaluation, model_kfold_evaluation

def _fnn_builder(config):

    initializer = tf.keras.initializers.LecunNormal()
    fnn_inputs = Input(shape=(int(config.input_dim),), name = 'fnn_inputs')
    x = BatchNormalization(momentum=0.95,epsilon=0.005,beta_initializer=RandomNormal(mean=0.0, stddev=0.05),gamma_initializer=Constant(value=0.9))(fnn_inputs)
    for i in range(1, config.layers_num):
        x = Dense(config.units, activation = config.activation, name = "fnn_dense_" + str(i), kernel_initializer=initializer)(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization(momentum=0.95,epsilon=0.005,beta_initializer=RandomNormal(mean=0.0, stddev=0.05),gamma_initializer=Constant(value=0.9))(x)
    outputs = Dense(int(config.output_dim), activation=config.output_act, name="predictions")(x)

    model = Model(inputs=fnn_inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])
    ct = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plot_model(model, show_shapes=True, to_file='F:\Cosmos\Cosmos\src\model_checkpoint\cosmos_fnn.png')
    return model, fnn_inputs, x

def _dcnn_builder(config):

    cnn_inputs = Input(shape=(int(config.input_dim),), name='cnn_inputs')
    net = tf.expand_dims(cnn_inputs, -1)
    initializer = tf.keras.initializers.LecunNormal()

    x = BatchNormalization(momentum=0.95,epsilon=0.005,beta_initializer=RandomNormal(mean=0.0, stddev=0.05),gamma_initializer=Constant(value=0.9))(net)
    for i in range(config.layers_num):
        num_filters = int(config.num_filters)
        for _ in range(2):
            x = Conv1D(filters=num_filters, kernel_size=int(config.kernel), padding=config.padding, activation=str(config.activation), kernel_initializer=initializer)(x)
        x = AveragePooling1D(pool_size=int(config.pool),  name = "AvgPool_" + str(i), strides=int(config.stride))(x)
        x = Dropout(config.dropout_cnn)(x)
        x = BatchNormalization(momentum=0.95,epsilon=0.005,beta_initializer=RandomNormal(mean=0.0, stddev=0.05),gamma_initializer=Constant(value=0.9))(x)
    x = Conv1D(filters=num_filters, kernel_size=int(config.kernel), padding=config.padding, activation=str(config.activation), kernel_initializer=initializer)(x)
    x = MaxPooling1D(pool_size=int(config.pool),  name = "MaxoPool_1", strides=int(config.stride))(x)
    x_flatten = Flatten()(x)
    x = BatchNormalization(momentum=0.95,epsilon=0.005,beta_initializer=RandomNormal(mean=0.0, stddev=0.05),gamma_initializer=Constant(value=0.9))(x_flatten)
    for _ in range(config.fc_layers_num - 1):
        x = Dense(config.fc_units, activation=config.activation, kernel_initializer=initializer)(x)
    x = Dense(config.fc_units, activation=config.activation, kernel_initializer=initializer)(x)
    outputs = Dense(int(config.output_dim), activation=config.output_act, name="predictions")(x)

    model = Model(inputs=cnn_inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])
    ct = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plot_model(model, show_shapes=True, to_file='F:\Cosmos\Cosmos\src\model_checkpoint\cosmos_cnn.png')
    return model, cnn_inputs, x_flatten


def _combined_fnn_builder(fnn, dcnn_flatten, fnn_inputs, dcnn_inputs, config):

    initializer = tf.keras.initializers.LecunNormal()
    merged_model = concatenate([fnn, dcnn_flatten])
    x = Dropout(config.dropout_fc)(merged_model)
    x = BatchNormalization(momentum=0.95,epsilon=0.005,beta_initializer=RandomNormal(mean=0.0, stddev=0.05),gamma_initializer=Constant(value=0.9))(fnn_inputs)
    for _ in range(5):
        x = Dense(config.units, activation = config.activation, kernel_initializer=initializer)(x)
    outputs = Dense(int(config.output_dim), activation=config.output_act, name="Output")(x)
    model = Model(inputs=[dcnn_inputs, fnn_inputs], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])
    ct = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plot_model(model, show_shapes=True, to_file='F:\Cosmos\Cosmos\src\model_checkpoint\cosmos_hybrid.png')

    return model


def _model_builder(config_ann, config_cnn):


    '''
    Coordina la costruzione dei quattro modelli che compongono il cuore dell'archiettura di cosmos.
    Rete 1: Cosmos_DCNN
    Rete 2: Cosmos_fnn_FC
    Rete 3: Comsos_Combine_Layer
    Rete 4: Cosmos_Combined_FNN

    ---------------              ----------------
      Cosmos_DCNN                  Cosmos_fnn_FC
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

    cosmos_dcnn = _dcnn_builder(config_cnn)

    '''
    Rete 2: Cosmos_fnn_FC
        Utilizza e sfrutta i dati secondari presenti all'ìnterno del TCE, ossia quelli categorici/numerici non direttamente connessi alle global\local view.
    '''

    cosmos_ann = _fnn_builder(config_ann)

    '''
    Rete 3: Cosmos_Combined_Layer
            Combina gli output delle due reti. Dato che i dati sono multi-tipo è la chiave di volta dell'intera architettura. Permette di combinarne gli output
            generando l'input per l'ultimo strato fully connected di classificazione, che utilizzerà i risultati delle reti precedenti
    '''

    '''
    Rete 4: Cosmos_Combined_FNN
            Ultimo modulo della rete complessa di cosmos. Effettua la fase di classificazione finale.
    '''
    return cosmos_dcnn, cosmos_ann


def _test_build(local, lc_train_dataset, aux_train_dataset, lc_valid_dataset, aux_valid_dataset, lc_test_dataset, aux_test_dataset, config_fnn, config_cnn):
    model_fnn, fnn_inputs, fnn_last_layer = _fnn_builder(config_fnn)
    model_cnn, cnn_inputs, cnn_last_layer = _dcnn_builder(config_cnn)
    combined_model = _combined_fnn_builder(fnn_last_layer, cnn_last_layer, fnn_inputs, cnn_inputs, config_fnn)
    #model_kfold_evaluation('fnn', model_fnn, aux_train_dataset, aux_valid_dataset, aux_test_dataset, config_fnn)
    #model_kfold_evaluation('cnn', model_cnn, lc_train_dataset, lc_valid_dataset, lc_test_dataset, config_cnn)
    hybrid_kfold_evaluation(combined_model, lc_train_dataset, lc_valid_dataset, lc_test_dataset, aux_train_dataset, aux_valid_dataset, aux_test_dataset, config_cnn)

    return True

if __name__ == '__main__':
    _test_build()