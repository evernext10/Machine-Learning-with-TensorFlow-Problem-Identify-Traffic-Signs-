import tensorflow as tf
from Course_OpenWebinars.UsefulTools.TensorFlowUtils import *
import matplotlib.pyplot as plt
import time

#Placeholders
x_placeholder = tf.placeholder(tf.float32, shape=[4,2]) #Cuatro entradas y dos salidas de la red neuronal
y_placeholder = tf.placeholder(tf.float32, shape=[4,1]) #Cuatro entradas y una salida de la red neuronal

#Pesos
w1 = tf.Variable(tf.random_uniform(shape=[2,2], minval=-0.5, maxval=0.5))
w2 = tf.Variable(tf.random_uniform(shape=[2,1], minval=-0.5, maxval=0.5))

#Bias
b1 = tf.Variable(tf.zeros([2]))
b2 = tf.Variable(tf.zeros([1]))

y1 = tf.sigmoid(tf.matmul(x_placeholder,w1)+b1)
y2 = tf.sigmoid(tf.matmul(y1,w2)+b2)

error = tf.reduce_mean(((y_placeholder * tf.log(y2))+ ((1-y_placeholder)* tf.log(1.0-y2)))* -1)

train_step = tf.train.AdamOptimizer(0.002).minimize(error)

XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

sess = initialize_session()

t_start = time.clock()
y0_ = []
y1_ = []
y2_ = []
y3_ = []
errores = []
alimentador = {x_placeholder:XOR_X, y_placeholder:XOR_Y}

#Entrenamiento

for train in range (100000):
    sess.run(train_step, feed_dict=alimentador)
    y_predicha = y2.eval(alimentador)
    error1 = error.eval(alimentador)

    if train % 2 == 0:
        y0_.append(y_predicha[0])
        y1_.append(y_predicha[1])
        y2_.append(y_predicha[2])
        y3_.append(y_predicha[3])
        errores.append(error1)

    if train % 1000:
        pt("Entrenamiento", train+1)
        pt("W1", w1.eval())
        pt("W2", w2.eval())
        pt("Bias1", b1.eval())
        pt("Bias2", b2.eval())
        pt("Y_Salida", y_predicha)
        pt("Error", error1)

    if error1 < 0.04:
        break

pt("FIN DE ENTRENAMIENTO")

t_end = time.clock()
pt("Ha tardado", t_end - t_start)

plt.plot(y0_, label="y0")
plt.plot(y1_, label="y1")
plt.plot(y2_, label="y2")
plt.plot(y3_, label="y3")
plt.legend()
plt.show()

plt.plot(errores, label="Errores")
plt.legend()
plt.show()