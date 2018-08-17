import tensorflow as tf
import matplotlib.pyplot as plt
from Course_OpenWebinars.UsefulTools.TensorFlowUtils import *

learning_rate = 0.001

training_epoch = 600

display_step = 50

#Placeholders

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

x = 5.0
y = 10.0

#El peso

W = tf.Variable(0.0)

#Biases

#Tipo 1

#b = tf.constant(0.0)

#Tipo 2

b = tf.Variable(0.0)

y_salida = tf.add(tf.multiply(W,X),b)

#Error = Resta absoluta entre lo que debe salir (10.0) y lo que ha salid
error = tf.abs(tf.subtract(Y, y_salida))

#Optimizador
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(error)

session = initialize_session()

W_s = []
b_s = []


#Feeder
alimentador = {X:x, Y:y}

#Entrenamiento
for epoch in range(training_epoch):
    session.run(optimizer,feed_dict=alimentador)
    w = W.eval()
    b_ = b.eval()

    W_s.append(w)
    b_s.append(b_)

    #Ver resultados

    if(epoch + 1) % display_step==0:
        c = session.run(error, feed_dict=alimentador)
        pt("Epoch", epoch + 1)
        pt("W",w)
        pt("b", b_)


pt("Fin del entrameinto")

error_final = session.run(error, feed_dict=alimentador)

pt("Error del enternamiento", error_final)
y_predicha = y_salida.eval(feed_dict=alimentador)

pt("Y de salida", y_predicha)


plt.plot(W_s,label="Pesos")
plt.plot(b_s, label="Biases")
plt.legend()
plt.show()
