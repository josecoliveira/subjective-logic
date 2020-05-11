from Hyperopinion import Hyperopinion, DecisionMaking


def main():
    _1A = DecisionMaking(Hyperopinion(3, [0, 0, 0, 0, 0, 0]), [0, 0, 100])
    _1B = DecisionMaking(Hyperopinion(3, [4/9, 3/9, 2/9, 0, 0, 0]), [0, 0, 100])
    d = _1A.decide({0}, _1B, {1})
    if d is not None:
        print('1A' if d else '1B')

    _2A = DecisionMaking(Hyperopinion(3, [0, 0, 0, 0, 0, 0]), [100, 0, 100])
    _2B = DecisionMaking(Hyperopinion(3, [4 / 9, 3 / 9, 2 / 9, 0, 0, 0]), [0, 100, 100])
    d = _1A.decide({0, 2}, _1B, {1, 2})
    if d is not None:
        print('2A' if d else '2B')



if __name__ == "__main__":
    main()
