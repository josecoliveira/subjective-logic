import numpy as np

from .HyperopinionDecorator import HyperopinionDecorator
from .HyperopinionInterface import HyperopinionInterface
from .Hyperopinion import Hyperopinion

def cumulative_fusion(CC):
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

        u = np.prod([C.u for C in CC])/dem

        if all(C.u == 1 for C in CC):
            a = np.array([])
            for x in range(k):
                a = np.append(a, np.sum([C.a[x] for C in CC]) / N)
        else:
            a = np.array([])
            for x in range(k):
                anum = np.sum([C.a[x] * np.prod([Cj.u for Cj in CC if Cj != C]) for C in CC]) - np.sum([C.a[x] for C in CC]) * np.prod([C.u for C in CC])
                adem = np.sum([np.prod([Cj.u for Cj in CC if Cj != C]) for C in CC]) - N * np.prod([C.u for C in CC])
                a = np.append(a, anum/adem)
    else:
        CCdog = [C for C in CC if C.u == 0]
        gamma = len(CCdog)

        b = np.array([])
        for x in range(kappa):
            b = np.append(b, np.sum([C.b[x] for C in CCdog]) / gamma)
        
        u = 0

        a = np.array([])
        for x in range(k):
            a = np.append(a, np.sum([C.a[x] for C in CCdog]) / gamma)
    
    return Hyperopinion(k, b, a)





# def averaging_fusion(ha=None, hb=None, opinions_list=None):
#     if opinions_list is None:
#         ra = np.zeros(ha.kappa)


#         pass
#     else:
#         if ha.k != hb.k:
#             raise Exception("The hyperopinions have different cardinalities.")

#         haa = ha.a[0:ha.k]
#         hba = hb.a[0:hb.k]

#         k = ha.k
#         if ha.u != 0 or hb.u != 0:
#             b = (ha.b * hb.u + hb.b * ha.u) / (ha.u + hb.u)
#             # u = (2 * ha.u * hb.u) / (ha.u + hb.u)
#             a = (ha.a + hb.a) / 2
#         elif ha.u == 0 and hb.u == 0:
#             b = 0.5 * ha.b + 0.5 * hb.b
#             a = 0.5 * haa + 0.5 * hba

#         return Hyperopinion(k, b, a)
