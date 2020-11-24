import numpy as np

from .HyperopinionDecorator import HyperopinionDecorator
from .HyperopinionInterface import HyperopinionInterface
from .Hyperopinion import Hyperopinion


def cumulative_fusion(ha=None, hb=None, opinions_list=None):
    if opinions_list != None:
        if len(opinions_list) < 2:
            raise Exception("List of opinions must be greater than 1")
        else:
            fusion = opinions_list[0]
            for i in range(1, len(opinions_list)):
                fusion = cumulative_fusion(fusion, opinions_list[i])
            return fusion

    if ha.k != hb.k:
        raise Exception("The hyperopinions have different cardinalities.")

    haa = ha.a[0:ha.k]
    hba = hb.a[0:hb.k]

    k = ha.k
    if ha.u != 0 or hb.u != 0:
        b = (ha.b * hb.u + hb.b * ha.u) / (ha.u + hb.u - ha.u * hb.u)
        if ha.u != 1 or hb.u != 1:
            a = (haa * hb.u + hba * ha.u - (haa + hba) *
                 ha.u * hb.u) / (ha.u + hb.u - 2 * ha.u * hb.u)
        elif ha.u == 1 and hb.u == 1:
            a = (haa + hba) / 2
    elif ha.u == 0 and hb.u == 0:
        b = 0.5 * ha.b + 0.5 * hb.b
        a = 0.5 * haa + 0.5 * hba

    return Hyperopinion(k, b, a)


def averaging_fusion(ha: HyperopinionInterface, hb: HyperopinionInterface):
    if ha.k != hb.k:
        raise Exception("The hyperopinions have different cardinalities.")

    haa = ha.a[0:ha.k]
    hba = hb.a[0:hb.k]

    k = ha.k
    if ha.u != 0 or hb.u != 0:
        b = (ha.b * hb.u + hb.b * ha.u) / (ha.u + hb.u)
        u = (2 * ha.u * hb.u) / (ha.u + hb.u)
        a = (ha.a + hb.a) / 2
    elif ha.u == 0 and hb.u == 0:
        b = 0.5 * ha.b + 0.5 * hb.b
        a = 0.5 * haa + 0.5 * hba

    return Hyperopinion(k, b, a)

