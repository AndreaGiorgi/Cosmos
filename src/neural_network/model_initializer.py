import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv1D, MaxPool1D
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.metrics import AUC
import tensorflow_datasets as tfds
import sklearn

def callbacks_builder():
    initial_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True)
    #checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model_{type}.h5".format(type = data_type), save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience=20, restore_best_weights=True)

    return lr_schedule, early_stopping_cb


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

    return x, y

def _mlp_builder(dataset, val_dataset, test_set):

    x_train, y_train = _aux_dataset_formatter(dataset)
    x_val, y_val = _aux_dataset_formatter(val_dataset)
    x_test, y_test = _aux_dataset_formatter(test_set)

    inputs = Input(shape=(5,), name = 'inputs')
    x = BatchNormalization()(inputs)
    x = Dense(512, activation='relu', name="dense_1")(x)
    x = Dropout(0.25)(x)
    x = Dense(512, activation='relu', name="dense_2")(x)
    x = Dense(512, activation='relu', name="dense_3")(x)
    x = Dense(512, activation='relu', name="dense_4")(x)
    x = Dense(512, activation='relu', name="dense_5")(x)
    outputs = Dense(3, activation='softmax',name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    print(model.summary())
    _, early = callbacks_builder()
    model.fit(x_train, y_train, epochs=50, batch_size=128, callbacks = [early], validation_data=(x_val, y_val), validation_freq = 2, use_multiprocessing=True)
    y_pred = model.predict(x_test, batch_size = 128, use_multiprocessing=True)
    print(sklearn.metrics.accuracy_score(y_test, y_pred))

    return


def _dcnn_builder():
    #? shape 1
    #? Conv1D dato che i dati sono una time series, Conv2D è ideale per immagini
    #? MaxPool
    #? Repeat conv-pool block for x times
    #? AVG POOL at the end? maybe it depends by AUC results (try max and avg)

    #x_train, y_train = _lc_dataset_formatter(dataset)
    #x_val, y_val = _lc_dataset_formatter(val_dataset)

    inputs = Input(shape=(1,), name='inputs')
    

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