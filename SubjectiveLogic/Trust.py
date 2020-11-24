import numpy as np

from .HyperopinionDecorator import HyperopinionDecorator
from .HyperopinionInterface import HyperopinionInterface
from .Hyperopinion import Hyperopinion


def trust_discount_2e(hab: HyperopinionInterface, hbx: HyperopinionInterface):
    if hab.k != 2:
        raise Exception("Trust opinion must have cardinality 2.")
    k = hbx.k
    b = hab.P[0]* hbx.b
    a = hbx.a
    return Hyperopinion(k, b, a)