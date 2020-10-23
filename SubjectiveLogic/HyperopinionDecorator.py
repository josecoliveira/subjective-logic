from .HyperopinionInterface import HyperopinionInterface


class HyperopinionDecorator(HyperopinionInterface):

    def __init__(self, hyperopinion: HyperopinionInterface) -> None:
        self.hyperopinion: HyperopinionInterface = hyperopinion

    def __getattr__(self, item):
        return getattr(self.hyperopinion, item)