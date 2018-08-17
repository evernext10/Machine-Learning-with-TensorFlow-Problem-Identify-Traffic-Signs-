from Course_OpenWebinars.UsefulTools.TensorFlowUtils import *
import tensorflow as tf

pt("Version de Tensorflow", tf.__version__)
import cv2
import numpy as np
import random
import time
import matplotlib.pyplot as plt




class Modelo():

    def __init__(self, input, test, input_labels, test_labels, numero_de_clase, options_problem="None"):
        self.input = input
        self.test = test
        self.input_labels = input_labels
        self.test_labels = test_labels
        self.numero_clases = numero_de_clase
        self.input_batch = None
        self.label_batch = None

        self.show_advance_info = False
        self.show_images = False

        self.shuffle_date = True

        #Variables
        self.input_rows_numer = 60
        self.input_colums_number = 60
        self.kernel_size = [7,7]
        self.epoch_number = 100
        self.batch_size = 16
        self.input_size = len(input)
        self.test_size = len(test)
        self.train_droput = 0.5

        #Primera Capa de Neuronas
        self.first_label_neurons = 16
        self.second_label_neurons = 32
        self.third_label_neurons = 64

        #Radio de aprendizaje
        self.learning_rate = 1e-3
        self.number_epoch_to_change_learning_rate = 2
        self.trains = int(self.input_size / self.batch_size) + 1

        #Variables de informacion
        self.index_buffer_data = 0
        self.num_trains_count = 1
        self.train_accuracy = None
        self.test_accuracy = None
        self.num_epochs_count = 1

        self.options = [options_problem, cv2.IMREAD_GRAYSCALE, self.input_rows_numer, self.input_colums_number]


    def convolucion_imagenes(self):
        self.print_actual_configuracion()
        x_input, y_label, keep_probably = self.placeholders(args=None, kwargs=None)
        x_reshape = tf.reshape(x_input,[-1,self.input_rows_numer, self.input_colums_number, 1])


        y_prediction = self.network_structure(x_reshape, args=None, keep_probably=keep_probably)
        cross_entropy, train_step, correct_prediction, accuracy = self.model_evaluation(y_label=y_label, y_prediction=y_prediction)


        sess = initialize_session()
        self.train_model(args=None, kwargs=locals())


    def print_actual_configuracion(self):
        pt("Numero de neuronas en primera capa", self.first_label_neurons)
        pt("Numero de neuronas en segunda capa", self.second_label_neurons)
        pt("Numero de neuronas en tercera capa", self.third_label_neurons)
        pt("Tamaño de nuestro input", self.input_size)
        pt("Tamaño del batch", self.batch_size)

    def placeholders(self, *args, **kwargs):
        x = tf.placeholder(tf.float32, shape=[None, self.input_rows_numer * self.input_colums_number])
        y_ = tf.placeholder(tf.float32, shape=[None, self.numero_clases])
        dropout = tf.placeholder(tf.float32)

        return x, y_, dropout

    def network_structure(self, input, *args, **kwargs):

        keep_dropout = kwargs["keep_probably"]

        #Primera capa convolucional
        convolution_1 = tf.layers.conv2d(inputs=input,filters=self.first_label_neurons, kernel_size=self.kernel_size,padding="same")

        #Max pool 1
        pool1 = tf.layers.max_pooling2d(inputs=convolution_1, pool_size=[2,2], strides=2)

        # Segunda capa convolucional
        convolution_2 = tf.layers.conv2d(inputs=pool1, filters=self.second_label_neurons, kernel_size=[4,4], padding="same")

        # Max pool 2
        pool2 = tf.layers.max_pooling2d(inputs=convolution_2, pool_size=[2,2], strides=2)

        dropout1 = tf.nn.dropout(pool2, keep_dropout)

        #Dense Layer
        pool2_flat = tf.reshape(dropout1, [-1, int(self.input_rows_numer/4)* int(self.input_colums_number / 4) * self.second_label_neurons])

        dense = tf.layers.dense(inputs=pool2_flat, units=self.third_label_neurons)
        dropout2 = tf.nn.dropout(dense, keep_dropout)

        #Readout Layer

        w_fc2 = weight_variable([self.third_label_neurons, self.numero_clases])
        b_fc2 = bias_variable([self.numero_clases])

        y_convolucion = tf.matmul(dropout2, w_fc2) + b_fc2
        return y_convolucion


    def model_evaluation(self,y_labels, y_prediction):
        #Cross Entropy
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_labels, logits=y_prediction))

        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y_prediction, axis=1), tf.argmax(y_labels,axis=1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return cross_entropy, train_step, correct_prediction, accuracy



    def train_model(self, *args, **kwargs):

        x = kwargs['kwargs']['x_input']
        y_labels = kwargs['kwargs']['y_labels']
        keep_probably = kwargs['kwargs']['keep_probably']
        accuracy = kwargs['kwargs']['accuracy']
        train_step = kwargs['kwargs']['train_step']
        cross_entropy = kwargs['kwargs']['cross_entropy']
        y_prediction = kwargs['kwargs']['y_prediction']

        #Actualizar los batches
        self.update_batch(is_test = False)
        x_test_feed, y_test_feed = self.uptade_batch(is_test = True)

        #Empieza a contar el tiempo
        start_time = time.time()

        accuracies_train, accuracies_test, loss_train, loss_test = [],[],[],[]

        feed_dict_test_100 = {x:x_test_feed, y_labels:y_test_feed, keep_probably:1}

        num_train_start = int(self.num_trains_count % self.trains)
        if num_train_start == self.trains:
            num_train_start = 0

        #Por si paramos el entrenamiento
        parar_entrenamiento = False

        #Entrenamiento

        for epoch in range(self.num_epochs_count, self.epoch_number):
            if parar_entrenamiento:
                break
            for num_train in range(num_train_start, self.trains):
                #Actualizamos alimentadores de entrenamiento
                feed_dict_train_100 = {x:self.input_batch, y_labels:self.label_batch, keep_probably:1}
                feed_dict_train_dropout = {x:self.input_batch, y_labels:self.label_batch, keep_probably:self.train_droput}

                self.train_accuracy = accuracy.eval(feed_dict_train_100) * 100
                train_step.run(feed_dict_train_dropout)
                self.train_accuracy = accuracy.eval(feed_dict_test_100) * 100

                cross_entropy_train = cross_entropy.eval(feed_dict_train_100)
                cross_entropy_test = cross_entropy.eval(feed_dict_test_100)

                #Para generar estadisticas
                accuracies_train.append(self.train_accuracy)
                accuracies_test.append(self.test_accuracy)
                loss_train.append(cross_entropy_train)
                loss_test.append(cross_entropy_test)

                if num_train % 10 == 0:
                    percent_advance = str(num_train * 100 / self.trains)
                    pt("Tiempo", str(time.strftime("%Hh%Mm%Ss", time.gmtime((time.time() - start_time)))))

                    pt("Numero de entrenamiento " + str(self.num_trains_count) + " | Porcentaje del epoch " + str(epoch) + percent_advance + "%")
                    pt("train_accuracy", self.train_accuracy)
                    pt("cross_entropy_train", cross_entropy_train)
                    pt("test_accuracy", self.test_accuracy)
                    pt("index_buffer_date", self.index_buffer_data)

                if epoch % self.number_epoch_to_change_learning_rate == 0 and num_train == 9 and epoch != 0:
                    self.learning_rate = float(self.learning_rate / 2.)

                if self.show_advance_info:
                    self.show_advance_information(y_labels=y_labels, y_prediction: y_prediction, feed_dist = feed_dict_train_100)


                #Actualizamos el numero de entrenamiento
                self.num_trains_count += 1

                #Para parar el entrenamiento
                    if epoch % 2 == 0 and num_train == 9 and epoch != 0:
                        respuesta = str(input("¿Paramos el entranamiento?: Pulsa 'S' para si y 'N' para no")).upper()
                        if respuesta == "S":
                            parar_entrenamiento = True
                        elif respuesta == "N":
                            pass

                    self.update_batch(is_test = False)

        pt("FIN DEL ENTRENAMIENTO")
        self.show_save_statistics(accuracies_train=accuracies_train, accuracies_test=accuracies_test, loss_train=loss_train, loss_test=loss_test)


    def update_batch(self,is_test=False):
        if not is_test:
            self.input_batch, self.label_batch = self.data_buffer_generic_class(inputs = self.input, input_labels = self.input_labels, shuffle_data = self.shuffle_date, batch_size = self.batch_size, is_test = False, options = self.options)
        elif is_test:
            x_test_feed, y_test_feed = self.label_batch = self.data_buffer_generic_class(inputs = self.test, input_labels = self.test_labels, shuffle_data = self.shuffle_date, batch_size = None, is_test = True, options = self.options)
            return x_test_feed, y_test_feed




    def show_save_statistics(self, accuracies_train, accuracies_test, loss_train, loss_test):
        pass



def weight_variable(shape):
    peso_inicial = tf.truncated_normal(shape=shape, stddev=0.01)
    return tf.Variable(peso_inicial)


def bias_variable(shape):
    valor_incial = tf.constant(0.01,shape=shape)
    return tf.Variable(valor_incial)
