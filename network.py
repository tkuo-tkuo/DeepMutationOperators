import tensorflow as tf
import keras 

class SimplyNetwork():

    def __init__(self):
        self.number_of_train_data = 5000
        self.number_of_test_data = 1000
        self.resize_width = 28
        self.resize_height = 28


    def get_number_of_train_data(self):
        return self.number_of_train_data


    def load_data(self):
        (train_datas, train_labels), (test_datas, test_labels) = keras.datasets.mnist.load_data()

        train_labels = train_labels[:self.number_of_train_data]
        test_labels = test_labels[:self.number_of_test_data]
        train_datas = train_datas[:self.number_of_train_data].reshape(-1, self.resize_width * self.resize_height) / 255.0
        test_datas = test_datas[:self.number_of_test_data].reshape(-1, self.resize_width * self.resize_height) / 255.0
        return (train_datas, train_labels), (test_datas, test_labels)


    def create_model(self):
        model = keras.models.Sequential([
            keras.layers.Dense(64, activation=tf.nn.relu,
                                  input_shape=(self.resize_width * self.resize_height, )),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        model = self.compile_model(model)
        return model


    def create_debug_model(self):
        model = keras.models.Sequential([
            keras.layers.Dense(64, activation=tf.nn.relu,
                                  input_shape=(self.resize_width * self.resize_height, )),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation=tf.nn.relu),
            keras.layers.Dense(16, activation=tf.nn.relu),
            keras.layers.Dense(16, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        model = self.compile_model(model)
        return model


    def compile_model(self, model):
        model.compile(optimizer='adam',
                      loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])
        return model

    def train_model(self, model, train_datas, train_labels, name_of_file, epochs=20, batch_size=None, with_checkpoint=False):
        if with_checkpoint:
            prefix = ''
            filepath = prefix + name_of_file + '-{epoch:02d}-{loss:.4f}.h5'
            checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=5, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]
            model.fit(train_datas, train_labels, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, verbose=False)
        else:
            model.fit(train_datas, train_labels, epochs=epochs, batch_size=batch_size, callbacks=None, verbose=False)
        return model
        
    def load_model(self, name_of_file):
        file_name = name_of_file + '.h5'
        return keras.models.load_model(file_name)

        # return keras.models.load_model(file_name)

    def save_model(self, model, name_of_file, mode='normal'):
        prefix = ''
        file_name = prefix + name_of_file + '.h5'
        model.save(file_name)
        if mode == 'normal':
            print('Normal model is successfully trained and saved at', file_name)
        else: 
            print('Mutated model by ' + mode + ' is successfully trained and saved at', file_name)
        print('')

    def evaluate_model(self, model, test_datas, test_labels, mode='normal'):
        loss, acc = model.evaluate(test_datas, test_labels)
        if mode == 'normal':
            print('model accurancy: {:5.2f}%'.format(100*acc))
            print('')
        else:
            print(mode, 'mutation operator executed')
            print('Mutated model, accurancy: {:5.2f}%'.format(100*acc))
            print('')

    def train_and_save_normal_model(self, name_of_file, verbose=False, with_checkpoint=False):
        (train_datas, train_labels), (test_datas, test_labels) = self.load_data()
        model = self.create_model()
        model = self.train_model(model, train_datas, train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            print('Current tensorflow version:', tf.__version__)
            print('')

            print('train dataset shape:', train_datas.shape)
            print('test dataset shape:', test_datas.shape)
            print('network architecture:')
            model.summary()
            print('')

            self.evaluate_model(model, test_datas, test_labels)

        self.save_model(model, 'normal_model')

    def train_and_save_debug_model(self, name_of_file, verbose=False, with_checkpoint=False):
        (train_datas, train_labels), (test_datas, test_labels) = self.load_data()
        model = self.create_debug_model()
        model = self.train_model(model, train_datas, train_labels, name_of_file, with_checkpoint=with_checkpoint)

        if verbose:
            print('Current tensorflow version:', tf.__version__)
            print('')

            print('train dataset shape:', train_datas.shape)
            print('test dataset shape:', test_datas.shape)
            print('network architecture:')
            model.summary()
            print('')

            self.evaluate_model(model, test_datas, test_labels)

        self.save_model(model, 'debug_model')
