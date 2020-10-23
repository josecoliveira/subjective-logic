import numpy as np

from .HyperopinionDecorator import HyperopinionDecorator
from .HyperopinionInterface import HyperopinionInterface
from .Hyperopinion import Hyperopinion


def cumulative_fusion(ha: HyperopinionInterface, hb: HyperopinionInterface):
    if ha.k != hb.k:
        raise Exception("The hyperopinions have different cardinalities.")

    haa = ha.a[0:ha.k]
    hba = hb.a[0:hb.k]

    k = ha.k
    if  ha.u != 0 or hb.u != 0:
        b = (ha.b * hb.u + hb.b * ha.u) / (ha.u + hb.u - ha.u * hb.u)
        if ha.u != 1 or hb.u != 1:
            a = (haa * hb.u + hba * ha.u - (haa + hba) * ha.u * hb.u) / (ha.u + hb.u - 2 * ha.u * hb.u)
        elif ha.u == 1 and hb.u == 1:
            a = (haa + hba) / 2
    elif ha.u == 0 * hb.u == 0:
        b = 0.5 * ha.b + 0.5 * hb.b
        a = 0.5 * haa + 0.5 * hba

    return Hyperopinion(k, b, a)

# class BeliefFusion(HyperopinionDecorator):

#     @staticmethod
#     def cumulative_fusion(ha: HyperopinionInterface, hb: HyperopinionInterface):
#         if ha.k != hb.k:
#             raise Exception("The hyperopinions have different cardinalities.")

#         haa = ha.a[0:ha.k]
#         hba = hb.a[0:hb.k]

#         k = ha.k
#         if  ha.u != 0 or hb.u != 0:
#             b = (ha.b * hb.u + hb.b * ha.u) / (ha.u + hb.u - ha.u * hb.u)
#             if ha.u != 1 or hb.u != 1:
#                 a = (haa * hb.u + hba * ha.u - (haa + hba) * ha.u * hb.u) / (ha.u + hb.u - 2 * ha.u * hb.u)
#             elif ha.u == 1 and hb.u == 1:
#                 a = (haa + hba) / 2
#         elif ha.u == 0 * hb.u == 0:
#             b = 0.5 * ha.b + 0.5 * hb.b
#             a = 0.5 * haa + 0.5 * hba

#         return Hyperopinion(k, b, a)


#     def __init__(self, hyperopinion: HyperopinionInterface):
#         super().__init__(hyperopinion)