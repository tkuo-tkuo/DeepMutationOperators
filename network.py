import tensorflow as tf

class SimplyNetwork():

    def __init__(self):
        self.number_of_train_data = 5000
        self.number_of_test_data = 1000
        self.resize_width = 28
        self.resize_height = 28

    def get_number_of_train_data(self):
        return self.number_of_train_data

    def load_data(self):
        (train_images, train_labels), (test_images,
                                       test_labels) = tf.keras.datasets.mnist.load_data()

        train_labels = train_labels[:self.number_of_train_data]
        test_labels = test_labels[:self.number_of_test_data]

        train_images = train_images[:self.number_of_train_data].reshape(
            -1, self.resize_width * self.resize_height) / 255.0
        test_images = test_images[:self.number_of_test_data].reshape(
            -1, self.resize_width * self.resize_height) / 255.0

        return (train_images, train_labels), (test_images, test_labels)

    def create_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu,
                                  input_shape=(self.resize_width * self.resize_height, )),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        model = self.compile_model(model)

        return model

    def create_debug_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu,
                                  input_shape=(self.resize_width * self.resize_height, )),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation=tf.nn.relu),
            tf.keras.layers.Dense(16, activation=tf.nn.relu),
            tf.keras.layers.Dense(16, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        model = self.compile_model(model)

        return model

    def compile_model(self, model):
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])

        return model

    def load_model(self, name_of_file):
        file_name = name_of_file + '.h5'
        return tf.keras.models.load_model(file_name)

    def train_and_save_normal_model(self, name_of_file, verbose=False):
        (train_images, train_labels), (test_images, test_labels) = self.load_data()
        model = self.create_model()
        model.fit(train_images, train_labels, epochs=20, verbose=False)

        if verbose:
            print('Current tensorflow version:', tf.__version__)
            print('')

            print('train dataset shape:', train_images.shape)
            print('test dataset shape:', test_images.shape)
            print('network architecture:')
            model.summary()
            print('')

            loss, acc = model.evaluate(test_images, test_labels)
            print('Trained model, accurancy: {:5.2f}%'.format(100*acc))
            print('')

        file_name = name_of_file + '.h5'
        model.save(file_name)

        print('New model is successfully created and saved as', file_name)
