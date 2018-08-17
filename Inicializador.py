import tensorflow as tf
import Buscador
from Course_OpenWebinars.UsefulTools.TensorFlowUtils import *

conjunto_de_entrenamiento = "D:\\MACHINE_LEARNING\\DITS-classification\\classification_train\\"
conjunto_de_test = "D:\\MACHINE_LEARNING\\DITS-classification\\classification_test\\"

conjuntos = [conjunto_de_entrenamiento,conjunto_de_test]
numero_clases = 59
buscador = Buscador.Buscador(numero_clases=numero_clases, paths=conjuntos)
buscador.encuentra_imagenes_y_labels()

conjunto_entrenamiento = buscador.x_train
conjunto_entrenamiento_labels = buscador.y_train


pt("1",conjunto_entrenamiento[100])
pt("2",conjunto_entrenamiento_labels[100])