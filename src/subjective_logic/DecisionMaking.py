import numpy as np

from .HyperopinionDecorator import HyperopinionDecorator
from .HyperopinionInterface import HyperopinionInterface


class DecisionMaking(HyperopinionDecorator):
    def __calculate_sharp_b(self):
        sharp_b = np.zeros(self.kappa)
        for x in range(self.kappa):
            for xi in range(self.kappa):
                if self.index_to_set(xi) <= self.index_to_set(x):
                    sharp_b[x] += self.b[xi]
        return sharp_b

    def __calculate_vague_b(self):
        vague_m = np.zeros(self.kappa)
        for x in range(self.kappa):
            for xi in range(self.k + 1, self.kappa):
                if not (self.index_to_set(xi) <= self.index_to_set(x)):
                    vague_m[x] += self.relative_a(x, xi) * self.b[xi]
        return vague_m

    def __init__(self, hyperopinion: HyperopinionInterface, utility):
        super().__init__(hyperopinion)

        if len(utility) != self.k:
            raise Exception(f"Size of utility vector must be equal to {self.k}.")

        self.sharp_b = self.__calculate_sharp_b()
        self.total_sharp_b = np.sum(self.b[0 : self.k])
        self.vague_b = self.__calculate_vague_b()
        self.total_vague_b = np.sum(self.b[self.k + 1 : self.kappa])
        self.focal_u = self.a * self.u

        self.mass_sum = np.column_stack((self.sharp_b, self.vague_b, self.focal_u))
        self.total_mass_sum = (self.total_sharp_b, self.total_vague_b, self.focal_u)

        self.utility = np.array(utility)
        self.L = self.utility * self.P[0 : self.k]
        self.total_L = np.sum(self.L)
        self.utility_max_abs = np.amax(np.abs(self.utility))
        self.normalized_P = self.L / self.utility_max_abs
        self.normalized_sharp_b = (
            self.utility * self.sharp_b[0 : self.k] / self.utility_max_abs
        )
        self.normalized_vague_b = (
            self.utility * self.sharp_b[0 : self.k] / self.utility_max_abs
        )
        self.normalized_focal_u = (
            self.utility * self.focal_u[0 : self.k] / self.utility_max_abs
        )
        self.normalized_mass_sum = np.column_stack(
            (self.normalized_sharp_b, self.normalized_vague_b, self.normalized_focal_u)
        )

    def decide(self, index, other: "DecisionMaking", index_other):
        if isinstance(index, set):
            index = self.set_to_index(index)
        if isinstance(index_other, set):
            index_other = self.set_to_index(index_other)
        if self.normalized_P[index] != other.normalized_P[index_other]:
            return self.normalized_P[index] > other.normalized_P[index_other]
        elif self.sharp_b[index] != other.sharp_b[index_other]:
            return self.sharp_b[index] > other.sharp_b[index_other]
        elif self.focal_u[index] != other.focal_u[index_other]:
            return self.focal_u[index] < other.focal_u[index_other]
        else:
            return "Difficult problem"
