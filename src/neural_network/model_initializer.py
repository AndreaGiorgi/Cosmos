import tensorflow as tf
from keras.utils.vis_utils import plot_model
from keras import Model, Input
from keras.layers import Dense, BatchNormalization, Conv1D, MaxPooling1D, AveragePooling1D, Flatten
from keras.initializers import RandomNormal, Constant
from keras.layers.merge import concatenate
from neural_network.model_evaluation import hybrid_kfold_evaluation, model_kfold_evaluation


def _snn_builder(config):

    initializer = tf.keras.initializers.LecunNormal()
    snn_inputs = Input(shape=(int(config.input_dim),), name = 'snn_inputs')
    x = BatchNormalization(momentum=0.95,epsilon=0.005, beta_initializer=RandomNormal(mean=0.0, stddev=0.05),gamma_initializer = Constant(value=0.9))(snn_inputs)
    for i in range(0, config.layers_num):
        x = Dense(config.units, activation = config.activation, name = "snn_dense_" + str(i), kernel_initializer=initializer)(x)
        x = tf.keras.layers.AlphaDropout(config.dropout_fc)(x)
    x = Dense(config.units, activation = config.activation, name = "snn_dense", kernel_initializer=initializer)(x)
    outputs = Dense(int(config.output_dim), activation=config.output_act, name="predictions")(x)

    model = Model(inputs=snn_inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate = config.learning_rate, epsilon = 1e-07), metrics=[tf.keras.metrics.AUC(curve = 'PR'), 'accuracy'])
    plot_model(model, show_shapes=True, to_file='F:\Cosmos\Cosmos\src\model_checkpoint\cosmos_snn.png')

    return model, snn_inputs, x

def _dcnn_builder(config):

    cnn_inputs = Input(shape=(int(config.input_dim),), name='cnn_inputs')
    net = tf.expand_dims(cnn_inputs, -1)
    initializer = tf.keras.initializers.LecunNormal()
    initializer_snn = tf.keras.initializers.LecunNormal()

    x = BatchNormalization(momentum=0.95,epsilon=0.005,beta_initializer=RandomNormal(mean=0.0, stddev=0.05),gamma_initializer=Constant(value=0.9))(net)
    for i in range(0, config.layers_num):
        num_filters = int(config.num_filters)
        for _ in range(2):
            x = Conv1D(filters=num_filters, kernel_size=int(config.kernel), padding=config.padding, activation=str(config.activation), kernel_initializer=initializer)(x)
        x = AveragePooling1D(pool_size=int(config.pool),  name = "AvgPool_" + str(i), strides=int(config.stride))(x)
        x = tf.keras.layers.SpatialDropout1D(0.25)(x)
    x = MaxPooling1D(pool_size=int(config.pool),  name = "MaxPool", strides=int(config.stride))(x)
    x_flatten = Flatten()(x)
    x = tf.keras.layers.AlphaDropout(config.dropout_fc)(x_flatten)
    for _ in range(config.fc_layers_num):
        x = Dense(config.fc_units, activation=config.activation, kernel_initializer=initializer_snn)(x)
    outputs = Dense(int(config.output_dim), activation=config.output_act, name="predictions")(x)

    model = Model(inputs=cnn_inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate = config.learning_rate, epsilon = 1e-07, amsgrad=True),  metrics=[tf.keras.metrics.AUC(curve = 'PR'), 'accuracy'])
    plot_model(model, show_shapes=True, to_file='F:\Cosmos\Cosmos\src\model_checkpoint\cosmos_cnn.png')

    return model, cnn_inputs, x_flatten


def _hybrid_builder(snn, dcnn_flatten, snn_inputs, dcnn_inputs, config):

    initializer = tf.keras.initializers.LecunNormal()
    merged_model = concatenate([dcnn_flatten, snn])
    x = tf.keras.layers.AlphaDropout(config.dropout_fc)(merged_model)
    x = BatchNormalization(momentum=0.95,epsilon=0.005, beta_initializer=RandomNormal(mean=0.0, stddev=0.05),gamma_initializer = Constant(value=0.9))(x)
    for _ in range(1, config.layers_num):
        x = Dense(config.units, activation = config.activation, kernel_initializer=initializer)(x)
    outputs = Dense(int(config.output_dim), activation=config.output_act, name="Output")(x)
    model = Model(inputs=[dcnn_inputs, snn_inputs], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(momentum = 0.50, nesterov = True), metrics=[tf.keras.metrics.AUC(curve = 'PR'), 'accuracy'])
    plot_model(model, show_shapes=True, to_file='F:\Cosmos\Cosmos\src\model_checkpoint\cosmos_hybrid.png')

    return model


def _model_builder(config_snn, config_cnn, config_hybrid):


    '''
    Coordina la costruzione dei quattro modelli che compongono il cuore dell'archiettura di cosmos.
    Rete 1: Cosmos_DCNN
    Rete 2: Cosmos_snn_FC
    Rete 4: Cosmos_Hybrid

    ---------------              ----------------
      Cosmos_DCNN                  Cosmos_snn_FC
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
                   Cosmos_Hybrid
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

    model_cnn, cnn_inputs, cnn_last_layer = _dcnn_builder(config_cnn)

    '''
    Rete 2: Cosmos_snn_FC
        Utilizza e sfrutta i dati secondari presenti all'ìnterno del TCE, ossia quelli categorici/numerici non direttamente connessi alle global\local view.
    '''

    model_snn, snn_inputs, snn_last_layer = _snn_builder(config_snn)


    '''
    Rete 3: Cosmos_Hybrid
            Ultimo modulo della rete complessa di cosmos. Effettua la fase di classificazione finale.

    '''
    model_hybrid = _hybrid_builder(snn_last_layer, cnn_last_layer, snn_inputs, cnn_inputs, config_hybrid)

    return model_snn, model_cnn, model_hybrid


def _test_build(local, lc_train_dataset, aux_train_dataset, lc_valid_dataset, aux_valid_dataset, lc_test_dataset, aux_test_dataset, config_snn, config_cnn, config_hybrid):
    model_snn, model_cnn, model_hybrid = _model_builder(config_snn, config_cnn, config_hybrid)

    #model_kfold_evaluation('snn', model_snn, aux_train_dataset, aux_valid_dataset, aux_test_dataset, config_snn)
    #model_kfold_evaluation('cnn', model_cnn, lc_train_dataset, lc_valid_dataset, lc_test_dataset, config_cnn)
    hybrid_kfold_evaluation(model_hybrid, lc_train_dataset, lc_valid_dataset, lc_test_dataset, aux_train_dataset, aux_valid_dataset, aux_test_dataset, config_hybrid)

    return True


if __name__ == '__main__':
    _test_build()