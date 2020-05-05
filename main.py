from Hyperopinion import Hyperopinion, DecisionMaking


def main():
    hyp = DecisionMaking(Hyperopinion(3, [4 / 9, 3 / 9, 2 / 9, 0, 0, 0]))
    print(hyp.b)
    print(hyp.u)
    print(hyp.a)
    print(sum(hyp.b) + hyp.u)
    print(hyp.P)
    print(hyp.J)


if __name__ == "__main__":
    main()
