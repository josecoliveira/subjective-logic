import numpy as np


class Opinion:
    W = 2

    def __init__(self, b, a):
        if len(b) != len(a):
            raise Exception("Size of belief mass distribution and base rate distribution are not equal.")
        elif sum(a) != 1:
            raise Exception("Sum of base rate distribution is not 1. It should be equal to 1.")
        elif sum(b) >= 1:
            raise Exception("Sum of belief mass distribution is greater than 1. It should be equal or less than 1.")
        else:
            self.k = len(b)
            self.b = np.array(b)
            self.a = np.array(a)
            self.u = 1 - sum(b)
            self.P = self.b + self.a * self.u
            self.Var = self.P * (1 - self.P) * self.u / (self.W + self.u)

            if self.u != 0:
                self.r = self.W * self.b / self.u
            else:
                self.r = self.b * np.Inf

            self.alpha = self.r + self.a * self.W
