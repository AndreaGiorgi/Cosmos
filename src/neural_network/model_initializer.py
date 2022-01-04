import tensorflow as tf
from tensorflow import keras

def callbacks_builder(data_type):
    initial_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model_{type}.h5".format(type = data_type), save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)

    return lr_schedule, checkpoint_cb, early_stopping_cb


def training_pipeline():
    '''
    Step 1: Addestra il modello Cosmos_DNN
    Step 2: Addestra il modello Cosmos_MLP_FC
    Step 3: "Addestra" il modello Cosmos_Combined_Layer
    Step 4: Addestra il modello Cosmos_Combined_FNN e ritorna
    '''
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

def _test_build(train_dataset, valid_dataset, data_type):
    return

if __name__ == '__main__':
    _test_build()