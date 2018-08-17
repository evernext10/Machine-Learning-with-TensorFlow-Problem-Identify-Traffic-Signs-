import os
import numpy as np

class Buscador():

    def __init__(self,paths,numero_clases):
        self.paths = paths
        self.numero_clases = numero_clases
        self.x_train = []
        self.y_train = []
        self.x_test =  []
        self.y_test = []

    def encuentra_imagenes_y_labels(self):
        for path in self.paths:
            for root, dirs, files in os.walk(path):
                for file_name in files:
                    if (file_name.endswith(".png")):
                        full_path = os.path.join(root,file_name)
                        self.obtiene_labels_desde_imagen(full_path)

    def obtiene_labels_desde_imagen(self, full_path):
        labels = np.zeros(self.numero_clases, dtype=np.float32)
        if 'train' in full_path:
            y_label_dir = os.path.dirname(os.path.dirname(full_path))
            y_label = os.path.basename(y_label_dir)
            labels[int(y_label)] = 1.0
            self.y_train.append(list(labels))
            self.x_train.append(full_path)
        if 'test' in full_path:
            y_label_dir = os.path.dirname(full_path)
            y_label = os.path.basename(y_label_dir)
            labels[int(y_label)] = 1.0
            self.y_test.append(list(labels))
            self.x_test.append(full_path)