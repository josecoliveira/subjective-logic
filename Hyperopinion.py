from typing import Union

import numpy as np
from numpy.random import default_rng


class Hyperopinion:
    W = 2

    @staticmethod
    def set_to_index(setx):
        raise NotImplementedError("Must implement conversion of set to value.")

    @staticmethod
    def index_to_set(index) -> set:
        raise NotImplementedError("Must implement conversion of value to set.")

    @staticmethod
    def generate_random_belief(k, a=None):
        rng = default_rng()
        l = np.array([rng.random() for i in range(2 ** k - 1)])
        l = l / np.sum(l)
        return Hyperopinion(k, l[0:-1], a)

    def __calculate_a(self, a_size_k):
        self.a = np.zeros(self.kappa)
        for x in range(self.kappa):
            setx = self.index_to_set(x)
            for index_from_set in setx:
                self.a[x] += a_size_k[index_from_set]

    def relative_a(self, x: Union[int, set], xi: Union[int, set]):
        if isinstance(x, int):
            x = self.index_to_set(x)
        if isinstance(xi, int):
            xi = self.index_to_set(xi)
        return self.a[self.set_to_index(x.intersection(xi))] / self.a[self.set_to_index(xi)]

    def __calculate_P(self):
        self.P = np.zeros(self.k)
        for x in range(self.k):
            P_aux = 0
            for xi in range(self.kappa):
                P_aux += self.relative_a(x, xi) * self.b[xi]
            P_aux += self.a[x]
            self.P[x] = P_aux

    def __calculate_alpha(self):
        if self.u != 0:
            r = self.W * self.b / self.u
        else:
            r = self.b * np.inf
        self.alpha = r + self.a * self.W

    def __init__(self, k, b, a=None):
        self.k = k
        self.kappa = 2 ** self.k - 2
        self.b = b
        if self.kappa != len(self.b):
            raise Exception('Size of belief mass distribution and size of domain (k) are not compatible.')
        if sum(self.b) >= 1:
            raise Exception('Sum of belief mass distribution is greater than 1. It should be equal or less than 1.')
        self.u = 1 - np.sum(self.b)

        if a is None:
            a = np.array([1 / self.k for _ in range(self.k)])
        else:
            a = np.array(a)
        if np.sum(a) != 1:
            raise Exception('Sum of base rate distribution is not 1. It should be equal to 1.')
        self.__calculate_a(a)

        self.__calculate_P()
        self.__calculate_alpha()

