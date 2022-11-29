import numpy as np

from .HyperopinionDecorator import HyperopinionDecorator
from .HyperopinionInterface import HyperopinionInterface
from .Hyperopinion import Hyperopinion

epsilon = 0.000001

def cumulative_fusion(CC, epistemic=True) -> HyperopinionInterface:
    if any(C.k != CC[0].k for C in CC[1:]):
        raise Exception("The hyperopinions have different cardinalities.")
    if len(CC) == 0:
        raise Exception("Empty list of hyperopinions")
    if len(CC) == 1:
        return CC[0]

    N = len(CC)
    k = CC[0].k
    kappa = CC[0].kappa

    if all(C.u != 0 for C in CC):
        b = np.array([])
        dem = np.sum([np.prod([Cj.u for Cj in CC if Cj != C]) for C in CC]) - (N - 1) * np.prod([C.u for C in CC])
        for x in range(kappa):
            num = np.sum([C.b[x] * np.prod([Cj.u for Cj in CC if Cj != C]) for C in CC])
            b = np.append(b, num / dem)

        if all(C.u == 1 for C in CC):
            a = np.array([])
            for x in range(k):
                a = np.append(a, np.sum([C.a[x] for C in CC]) / N)
        else:
            a = np.array([])
            for x in range(k):
                anum = np.sum([C.a[x] * np.prod([Cj.u for Cj in CC if Cj != C]) for C in CC]) - np.sum(
                    [C.a[x] for C in CC]) * np.prod([C.u for C in CC])
                adem = np.sum([np.prod([Cj.u for Cj in CC if Cj != C]) for C in CC]) - N * np.prod([C.u for C in CC])
                a = np.append(a, anum / adem)
    else:
        CCdog = [C for C in CC if C.u == 0]
        gamma = len(CCdog)

        b = np.array([])
        for x in range(kappa):
            b = np.append(b, np.sum([C.b[x] for C in CCdog]) / gamma)

        a = np.array([])
        for x in range(k):
            a = np.append(a, np.sum([C.a[x] for C in CCdog]) / gamma)
    
    b = np.round(b, decimals=6)
    a = np.round(a, decimals=6)

    if epistemic:
        return Hyperopinion(k, b, a).maximize_uncertainty()
    else:
        # print("aleatory")
        return Hyperopinion(k, b, a)


def averaging_fusion(CC, epistemic=True):
    if any(C.k != CC[0].k for C in CC[1:]):
        raise Exception("The hyperopinions have different cardinalities.")
    if len(CC) == 0:
        raise Exception("Empty list of hyperopinions")
    if len(CC) == 1:
        return CC[0]

    N = len(CC)
    k = CC[0].k
    kappa = CC[0].kappa

    if all(C.u != 0 for C in CC):
        b = np.array([])
        dem = np.sum([np.prod([Cj.u for Cj in CC if Cj != C]) for C in CC])
        for x in range(kappa):
            num = np.sum([C.b[x] * np.prod([Cj.u for Cj in CC if Cj != C]) for C in CC])
            b = np.append(b, num / dem)

        a = np.array([])
        for x in range(k):
            a = np.append(a, np.sum([C.a[x] for C in CC]) / N)
    else:
        CCdog = [C for C in CC if C.u == 0]
        gamma = len(CCdog)

        b = np.array([])
        for x in range(kappa):
            b = np.append(b, np.sum([C.b[x] for C in CCdog]) / gamma)

        a = np.array([])
        for x in range(k):
            a = np.append(a, np.sum([C.a[x] for C in CCdog]) / gamma)

    b = np.round(b, decimals=6)
    a = np.round(a, decimals=6)

    if epistemic:
        return Hyperopinion(k, b, a).maximize_uncertainty()
    else:
        # print("aleatory")
        return Hyperopinion(k, b, a)


def weighted_fusion(CC, epistemic=True):
    if any(C.k != CC[0].k for C in CC[1:]):
        raise Exception("The hyperopinions have different cardinalities.")
    if len(CC) == 0:
        raise Exception("Empty list of hyperopinions")
    if len(CC) == 1:
        return CC[0]

    N = len(CC)
    k = CC[0].k
    kappa = CC[0].kappa

    if all(C.u != 0 for C in CC) and any(C.u != 1 for C in CC):
        b = np.array([])
        dem = np.sum([np.prod([Ci.u for Ci in CC if Ci != C]) for C in CC]) - N * np.prod([C.u for C in CC])
        for x in range(kappa):
           num = np.sum([C.b[x] * (1 - C.u) * np.prod([Ci.u for Ci in CC if Ci != C]) for C in CC])
           b = np.append(b, num / dem)

        a = np.array([])
        for x in range(k):
            ax = np.sum([C.a[x] * (1 - C.u) for C in CC]) / (N - np.sum([C.u for C in CC]))
            a = np.append(a, ax)
    elif any(C.u == 0 for C in CC):
        CCdog = [C for C in CC if C.u == 0]
        gamma = len(CCdog)

        b = np.array([])
        for x in range(kappa):
            b = np.append(b, np.sum([C.b[x] for C in CCdog]) / gamma)

        a = np.array([])
        for x in range(k):
            a = np.append(a, np.sum([C.a[x] for C in CCdog]) / gamma)
    else:
        b = np.array([])
        for x in range(kappa):
            b = np.append(b, 0)

        a = np.array([])
        for x in range(k):
            a = np.append(a, np.sum([C.a[x] for C in CC]) / N)
    
    b = np.round(b, decimals=6)
    a = np.round(a, decimals=6)

    if epistemic:
        return Hyperopinion(k, b, a).maximize_uncertainty()
    else:
        # print("aleatory")
        return Hyperopinion(k, b, a)
