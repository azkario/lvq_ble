import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spatial


class LVQ():
    def __init__(self, dataset):
        """
        Construtor da classe
        :param nome_arquivo: nome do arquivo csv que contem os dados
        """
        self.dados = dataset
        self.dataset = dataset
        self.qtd_caracteristicas = 0
        self.amplitudes = []
        self.qtd_caracteristicas = len(self.dados[0]) - 1

    def nancosinedist(self,a,b):
        b = b.T  # transpose
        a = np.array(list(a))

        a_sq = np.square(a)

        if a.ndim > 1:
            a_nansum = np.nansum(a_sq, axis=1)  # sum over rows
        else:
            a_nansum = a_sq
        a_nansum = np.transpose(a_nansum)
        normA = np.sqrt(a_nansum)
        # force normA to be a column vector
        normA = normA[:, np.newaxis]

        b_sq = np.square(b)
        b_nansum = np.nansum(b_sq, axis=0)  # cols
        normB = np.sqrt(b_nansum)

        a = np.nan_to_num(a)
        b = np.nan_to_num(b)

        cos_dist = 1 - ((np.dot(a, b)) / normA) / normB

        return cos_dist
        # print(C)

    def normalizar(self):
        """
        Normalized all the characteristics for a range of 0 - 1, for all have the same weight in the
        """
        lista = [] * (len(self.dados[0]) - 1)
        self.amplitudes = []
        for caracteristica in range(len(self.dados[0]) - 1):
            lista = [elemento[caracteristica] for elemento in self.dados]
            self.amplitudes += [[max(lista), min(lista)]]
            for elemento in self.dados:
                elemento[caracteristica] = (elemento[caracteristica] - min(lista)) / (max(lista) + min(lista))

    def triagem(self, split: float = 0.65):
        """
        Divide randomly the elements of the data set into two subsets: test and training
         : param split: from 0 to 1 -> 'percentage' of the elements that will be in the training set
        """
        self.treino, self.teste = [], []
        for elemento in self.dados:
            if random.random() < split:
                self.treino += [elemento]
            else:
                self.teste += [elemento]

    def resumir(self, n: float = 10, e: float = 10, t: float = 0.4):
        """
        Returns the data codebook, that is, the elements that best represent the whole
         : param t: initial learning rate
         : param e: numero de epocas / number of iteration
         : param n: number of coodbook elements
        """
        # Geracacao aleatorio dos elementos iniciais do codebook
        # Random generation of the initial elements of the codebook
        self.codebook = [[]] * n
        for i in range(n):
            self.codebook[i] = random.choice(self.dados)
            # self.codebook[i] = [0] * (self.qtd_caracteristicas + 1)
            # for caracteristica in range(self.qtd_caracteristicas + 1):
            #     self.codebook[i][caracteristica] = random.choice(self.dados)[caracteristica]


        for epoca in range(e):
            taxa = t * (1.0 - (epoca / float(e))) #rate
            for elemento in self.treino:
                representante = self.encontrar_mais_proximo(elemento, self.codebook) #return new representation (from elemento training data) that closer than all current-codebook
                o = -1
                if representante[-1] == elemento[-1]: # compare class-label. If the same, proceed!
                    o = 1
                for caracteristica in range(self.qtd_caracteristicas): #proceed to update representante
                    erro = (elemento[caracteristica] - representante[caracteristica])
                    representante[caracteristica] += (erro * taxa * o)

        #     print("end of elemento iteration")
        # print("end of epoca iteration")

    def testar(self):
        """
        Performs the classification for each element of the test set and returns the accuracy of the algorithm
        """
        qtd_teste = len(self.teste)
        precisao = 100.0
        for elemento in self.teste:
            bmu = self.encontrar_mais_proximo(elemento, self.codebook)
            if bmu[-1] != elemento[-1]:
                precisao -= (1 / qtd_teste) * 100

        return precisao

#elemento = list element data (tiap kolom). From training data
#lista = list of list data (8rows * 5 cols). From current codebook
    #resposta => result
    def encontrar_mais_proximo(self, elemento, lista):
        """
        Performs the classification for each element of the test set and returns the accuracy of the algorithm
         : param elemento: vector to which the nearest vector of a given list is due
         : param lista: list of vectors
        """
        sz = len(elemento)-1; #size_of_element (without label)
        dstnce = [self.nancosinedist(np.array(elemento[0:-1]).reshape(-1,sz), np.array(lista[0][0:-1]).reshape(-1,sz))]

        resposta = [lista[0], dstnce]
        for i in lista: #looping for each list of current codebook
            distancia = self.nancosinedist(np.array(elemento[0:-1]).reshape(-1,sz), np.array(i[0:-1]).reshape(-1,sz))
            if distancia < resposta[1]: #  compare distancia (of training data)_ to each data in lista codebook
                resposta = [i, distancia]
        return resposta[0]

    @property
    def representantes(self):
        """
        Returns the "original" codebook, with the characteristics in their original ranges. That is,
         returns the denormalized codebook, if it has been normalized
        """
        representantes_desnormalizados = [[]] * len(self.codebook)
        if self.amplitudes:
            for index, representante in enumerate(self.codebook):
                representante_desnormalizado = []
                for caracteristica in range(self.qtd_caracteristicas):
                    aux = ((self.amplitudes[caracteristica][0] + self.amplitudes[caracteristica][1]) \
                           * representante[caracteristica]) + self.amplitudes[caracteristica][1]
                    representante_desnormalizado += [aux]
                representante_desnormalizado += [representante[-1]]
                representantes_desnormalizados[index] = representante_desnormalizado
        else:
            return self.codebook

        return representantes_desnormalizados

    @property
    def classes(self):
        """
        Returns the classes in the dataset
        """
        classes = []
        for elemento in self.dados:
            if elemento[-1] not in classes:
                classes.append(elemento[-1])

        return classes

#===================

import random


def importar_dataset(arquivo_csv: str = None):
    """
    Loads the initial data of the class through a csv file. Expect a file to have rows
         with n columns, so that n-ezima represents the class of the element and the previous ones represent,
         each one a different characteristic.
         : stop csv file: csv file name
    """
    dados = []
    with open(arquivo_csv, 'r') as arquivo_csv:
        arquivo = csv.reader(arquivo_csv)
        for index, linha in enumerate(arquivo):
            if linha:
                dados += [list(map(float, linha[0:-1]))]
                dados[index] += [linha[-1]]
    return dados


def random_cores(qtd: int = 3):
    """
    Randomly return colors in hexadecimal format according to the requested quantity
    """
    lista = [(210, 180, 140), (139, 69, 19), (244, 164, 96), (85, 107, 47), (0, 255, 0), (102, 205, 170),
             (127, 255, 212),
             (72, 209, 204), (0, 255, 255), (176, 196, 222), (30, 144, 255), (0, 0, 255), (220, 20, 60),
             (255, 105, 180),
             (255, 0, 255), (139, 0, 139), (255, 192, 203), (255, 0, 0), (250, 128, 114), (255, 165, 0), (255, 255, 0)]

    random.shuffle(lista)
    cores = lista[0:qtd]
    resposta = []
    for cor in cores:
        resposta += ['#%02x%02x%02x' % cor]
    return resposta

# ====================

# import matplotlib.pyplot as plt
dataset = importar_dataset("/Users/azkario/Google Drive/Doctorate (S3)/00023_lvq_python/lvq/datas/ble_ref_mini1.csv")
# dataset = importar_dataset("/Users/azkario/Google Drive/Doctorate (S3)/00023_lvq_python/lvq/datas/IRIS.csv")
#

# print("Algorithm with normalized data between 0 - 1")
# iris_norm = LVQ(dataset)
#
# iris_norm.triagem(0.75) #screening
# iris_norm.normalizar()
# iris_norm.resumir(n=15, e=26, t=0.5)#(n=2, e=2, t=0.5)
# for reprs in iris_norm.representantes:
#     print(reprs)
# print("Precisão: ", iris_norm.testar(), "% \n")

# classes = iris_norm.classes
#
# classes_cor = {}
# cores = random_cores(len(classes))
# for index, classe in enumerate(classes):
#     classes_cor[classe] = cores[index]
#
# for elemento in iris_norm.dataset:
#     plt.plot(elemento[0], elemento[1], 'o', color=classes_cor[elemento[-1]])
#
# for representante in iris_norm.codebook:
#     plt.plot(representante[0], representante[1], 'D', ms=10, mfc='none', color=classes_cor[representante[-1]])
#
# plt.show()

# Sem normalização
print("Algorithm with non-normalized data")
iris = LVQ(dataset)
iris.triagem(0.75)
iris.resumir(n=8, e=13, t=0.5)
print("Precisão: ", iris.testar(),"% \n")
for representante in iris.representantes:
  print(representante)