import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv1D, MaxPool1D
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.initializers import lecun_normal
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot
import tensorflow_datasets as tfds


def callbacks_builder():
    #checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model_{type}.h5".format(type = data_type), save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience=20, restore_best_weights=True)

    return early_stopping_cb


def training_pipeline():
    '''
    Step 1: Addestra il modello Cosmos_DNN
    Step 2: Addestra il modello Cosmos_MLP_FC
    Step 3: "Addestra" il modello Cosmos_Combined_Layer
    Step 4: Addestra il modello Cosmos_Combined_FNN e ritorna
    '''
    return

def _aux_dataset_formatter(dataset):
    return


def _aux_dataset_formatter(dataset):

    df = tfds.as_dataframe(dataset)
    df['inputs/Duration'] = df['inputs/Duration'].str.get(0)
    df['inputs/Epoc'] = df['inputs/Epoc'].str.get(0)
    df['inputs/Tmag'] = df['inputs/Tmag'].str.get(0)
    df['inputs/Period'] = df['inputs/Period'].str.get(0)
    df['inputs/Transit_Depth'] = df['inputs/Transit_Depth'].str.get(0)
    df['targets'] = df['targets'].str.get(0)

    y = df['targets']
    x = df.drop(labels=['targets'], axis=1)
    y = to_categorical(y)
    return x, y

def _mlp_builder(dataset, val_dataset, test_set):

    x_train, y_train = _aux_dataset_formatter(dataset)
    x_val, y_val = _aux_dataset_formatter(val_dataset)
    x_test, y_test = _aux_dataset_formatter(test_set)

    initializer = lecun_normal()
    inputs = Input(shape=(5,), name = 'inputs')
    x = BatchNormalization()(inputs)
    x = Dense(512, activation='selu', name="dense_1", kernel_initializer=initializer)(x)
    x = Dense(512, activation='selu', name="dense_2", kernel_initializer=initializer)(x)
    x = Dense(512, activation='selu', name="dense_3", kernel_initializer=initializer)(x)
    x = Dense(512, activation='selu', name="dense_4", kernel_initializer=initializer)(x)
    x = Dense(512, activation='selu', name="dense_5", kernel_initializer=initializer)(x)
    outputs = Dense(2, activation='softmax',name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(amsgrad=True), metrics=['accuracy'])

    print(model.summary())
    early = callbacks_builder()
    history = model.fit(x_train, y_train, epochs=100, batch_size=128, callbacks = early, validation_data=(x_val, y_val), validation_freq = 2, use_multiprocessing=True)

    # evaluate the model
    _, train_acc = model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # plot accuracy during training
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()

    return model


def _dcnn_builder():
    #? shape 1
    #? Conv1D dato che i dati sono una time series, Conv2D è ideale per immagini
    #? MaxPool
    #? Repeat conv-pool block for x times
    #? AVG POOL at the end? maybe it depends by AUC results (try max and avg)

    #x_train, y_train = _lc_dataset_formatter(dataset)
    #x_val, y_val = _lc_dataset_formatter(val_dataset)

    inputs = Input(shape=(1,), name='inputs')
    x = Conv1D(filters = 64, kernel_size = 50, activation='relu')(inputs)
    x = MaxPool1D(pool_size= 32, strides= 32)(x)
    x = Conv1D(filters = 64, kernel_size = 50, activation='relu')(x)
    x = MaxPool1D(pool_size= 32, strides= 32)(x)
    x = Conv1D(filters = 64, kernel_size = 50, activation='relu')(x)
    x = MaxPool1D(pool_size= 32, strides= 32)(x)
    x = Conv1D(filters = 64, kernel_size = 50, activation='relu')(x)
    x = MaxPool1D(pool_size= 32, strides= 32)(x)

    return 


def _combined_fnn_builder():
    model = Sequential()
    model.add(Input(shape=(6,)))
    #? Dropout 0.10/0.25
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    return


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

def _test_build(lc_train_dataset, aux_train_dataset, lc_valid_dataset, aux_valid_dataset, lc_test_dataset, aux_test_dataset, data_type):
    _mlp_builder(aux_train_dataset,aux_valid_dataset, aux_test_dataset)
    return

if __name__ == '__main__':
    _test_build()