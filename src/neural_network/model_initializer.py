import tensorflow as tf
import numpy as np
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv1D, MaxPool1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import lecun_normal, RandomNormal, Constant
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot
from sklearn.metrics import classification_report
import tensorflow_datasets as tfds

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

def training_pipeline():
    '''
    Step 1: Addestra il modello Cosmos_DNN
    Step 2: Addestra il modello Cosmos_MLP_FC
    Step 3: "Addestra" il modello Cosmos_Combined_Layer
    Step 4: Addestra il modello Cosmos_Combined_FNN e ritorna
    '''
    return


def _lc_dataset_formatter(dataset, local = False, train = True):
    df = tfds.as_dataframe(dataset)
    df['targets'] = df['targets'].str.get(0)
    y = df['targets']
    if local:
        data = df['inputs/local_view'].to_numpy()
    else:
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


def _mlp_test_eval(history, model, x_train, y_train, x_test, y_test, y_pred, y_pred_train):

    print(model.summary())
    # evaluate the model
    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print('Loss [Train: %.3f, Test: %.3f]' % (train_loss, test_loss))
    print('Accuracy [Train: %.3f, Test: %.3f]' % (train_acc, test_acc))
    print('Recall [Train: %.3f, Test: %.3f]' % (recall_m(y_train, y_pred_train),recall_m(y_test, y_pred)))
    print('Precision [Train: %.3f, Test: %.3f]' % (precision_m(y_train, y_pred_train),precision_m(y_test, y_pred)))
    print('F1 Score [Train: %.3f, Test: %.3f]' % (f1_m(y_train, y_pred_train),f1_m(y_test, y_pred)))

def _mlp_builder(dataset, val_dataset, test_set, config):

    x_train, y_train = _aux_dataset_formatter(dataset)
    x_val, y_val = _aux_dataset_formatter(val_dataset)
    x_test, y_test = _aux_dataset_formatter(test_set, False)

    initializer = lecun_normal()
    inputs = Input(shape=(int(config.input_dim),), name = 'inputs')
    x = BatchNormalization(momentum=0.95,epsilon=0.005,beta_initializer=RandomNormal(mean=0.0, stddev=0.05),gamma_initializer=Constant(value=0.9))(inputs)
    for i in range(1, config.layers_num):
        x = Dense(config.units, activation = config.activation, name = "dense_" + str(i), kernel_initializer=initializer)(x)
    x = BatchNormalization(momentum=0.95,epsilon=0.005,beta_initializer=RandomNormal(mean=0.0, stddev=0.05),gamma_initializer=Constant(value=0.9))(x)
    outputs = Dense(int(config.output_dim), activation=config.output_act, name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer=Adam(config.learning_rate, amsgrad=True), metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=100, batch_size=config.batch_size, validation_data=(x_val, y_val), validation_freq = 2, use_multiprocessing=True)
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    _mlp_test_eval(history, model, x_train, y_train, x_test, y_test, y_pred, y_pred_train)

    return outputs

def _lc_test_eval(history, model, x_train, y_train, x_test, y_test, y_pred, y_pred_train):


    # evaluate the model
    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print('Loss [Train: %.3f, Test: %.3f]' % (train_loss, test_loss))
    print('Accuracy [Train: %.3f, Test: %.3f]' % (train_acc, test_acc))
    print('Recall [Train: %.3f, Test: %.3f]' % (recall_m(y_train, y_pred_train),recall_m(y_test, y_pred)))
    print('Precision [Train: %.3f, Test: %.3f]' % (precision_m(y_train, y_pred_train),precision_m(y_test, y_pred)))
    print('F1 Score [Train: %.3f, Test: %.3f]' % (f1_m(y_train, y_pred_train),f1_m(y_test, y_pred)))

def _dcnn_builder(dataset, val_dataset, test_dataset, config, local = False):
    #? shape 1
    #? Conv1D dato che i dati sono una time series, Conv2D è ideale per immagini
    #? MaxPool
    #? Repeat conv-pool block for x times
    #? AVG POOL at the end? maybe it depends by AUC results (try max and avg)

    x_train, y_train = _lc_dataset_formatter(dataset, local)
    x_val, y_val = _lc_dataset_formatter(val_dataset, local, train = False)
    x_test, y_test = _lc_dataset_formatter(test_dataset, local, train=False)


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
    model.compile(loss='categorical_crossentropy', optimizer=Adam(config.learning_rate, amsgrad=True), metrics=['accuracy'])
    print(model.summary())
    history = model.fit(x_train, y_train, batch_size = 64, epochs=50, validation_data = (x_val, y_val), use_multiprocessing=True)
    y_pred_train = model.predict(x_train)
    y_pred = model.predict(x_test)

    _lc_test_eval(history, model, x_train, y_train, x_test, y_test, y_pred, y_pred_train)
    return outputs


def _combined_fnn_builder(mlp, dcnn):
    models = mlp


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

    '''
    Rete 2: Cosmos_MLP_FC
        Utilizza e sfrutta i dati secondari presenti all'ìnterno del TCE, ossia quelli categorici/numerici non direttamente connessi alle global\local view.
    '''

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
   # mlp = _mlp_builder(aux_train_dataset, aux_valid_dataset, aux_test_dataset, config_mlp)
    dcnn = _dcnn_builder(lc_train_dataset, lc_valid_dataset, lc_test_dataset, config_cnn, local)
    #_combined_fnn_builder(mlp, dcnn)
    return

if __name__ == '__main__':
    _test_build()