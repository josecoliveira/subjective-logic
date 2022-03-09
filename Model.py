import numpy as np

from SubjectiveLogic import Hyperopinion, HyperopinionInterface
from SubjectiveLogic.Trust import trust_discount_2e
from typing import List, Callable
import matplotlib.pyplot as plt

BeliefArray = List[float]
InfluenceGraph = List[List[float]]

State = List['HyperopinionInterface']
TrustGraph = List[List['HyperopinionInterface']]
TrustArray = List['HyperopinionInterface']

epsilon = 0.01

class AKVModel:
    belief_array: BeliefArray
    influence_graph: InfluenceGraph
    states: List[BeliefArray]

    @property
    def num_agents(self):
        return len(self.belief_array)

    def __init__(self, belief_array: BeliefArray, influence_graph: InfluenceGraph) -> None:
        self.belief_array = belief_array
        self.influence_graph = influence_graph
        self.states = [belief_array]

    @staticmethod
    def classic_update(belief_state_i: float, belief_state_j: float, influence: float) -> float:
        return belief_state_i + influence * (belief_state_j - belief_state_i)

    def overall_classic_update(self) -> BeliefArray:
        A = len(self.belief_array)
        new_belief_array = np.array([])
        for ai in range(A):
            new_belief_array = np.append(new_belief_array, (1 / A) * np.sum(
                [AKVModel.classic_update(self.belief_array[ai], self.belief_array[aj], self.influence_graph[aj][ai]) for
                 aj in range(A)]))
        self.belief_array = new_belief_array
        return self.belief_array
    
    def simulate(self, n: int) -> None:
        for _ in range(n):
            self.overall_classic_update()
            self.states += [self.belief_array]


    class InitialConfigurations:
        @staticmethod
        def uniform(num_agents: int) -> BeliefArray:
            belief_array = np.array([i/(num_agents - 1) for i in range(num_agents)])
            belief_array[0] = epsilon
            belief_array[-1] = 1 - epsilon
            return belief_array
        
        def mildly(num_agents: int) -> BeliefArray:
            middle = np.ceil(num_agents / 2)
            return [0.2 + 0.2 * i / middle if i < middle else 0.6 + 0.2 * (i - middle) / (num_agents - middle) for i in range(num_agents)]
        
        def extreme(num_agents: int) -> BeliefArray:
            middle = np.ceil(num_agents / 2)
            belief_array = np.array([0.2 * i / middle if i < middle else 0.8 + 0.2 * (i - middle) / (num_agents - middle) for i in range(num_agents)])
            belief_array[0] = epsilon
            return belief_array
        
        def tripolar(num_agents: int) -> BeliefArray:
            beliefs = [0.0] * num_agents
            first_third = num_agents // 3
            middle_third = np.ceil(num_agents * 2 / 3) - first_third
            last_third = num_agents - middle_third - first_third
            offset = 0
            for i, segment in enumerate((first_third, middle_third, last_third)):
                for j in range(int(segment)):
                    beliefs[int(j+offset)] = 0.2 * j / segment + (0.4 * i)
                offset += segment
            beliefs[0] = epsilon
            return np.array(beliefs)
    
    class InfluenceGraphs:
        def clique(num_agents: int, influence: float):
            influence_graph = np.full((num_agents, num_agents), influence)
            for i in range(num_agents):
                influence_graph[i, i] = 1
            return influence_graph
        
        def circular(num_agents: int, influence: float):
            inf_graph = np.zeros((num_agents, num_agents))
            for i in range(num_agents):
                inf_graph[i, i] = 1.0
                inf_graph[i, (i + 1) % num_agents] = influence
            return inf_graph
        
        def disconnected(num_agents: int, influence: float):
            inf_graph = np.zeros((num_agents, num_agents))
            middle = int(np.ceil(num_agents / 2))
            inf_graph[:middle, :middle] = influence
            inf_graph[middle:, middle:] = influence
            for i in range(num_agents):
                inf_graph[i, i] = 1
            return inf_graph
        
        def faintly(num_agents: int, weak_influence: float, strong_influence: float):
            inf_graph = np.full((num_agents, num_agents), weak_influence)
            middle = int(np.ceil(num_agents / 2))
            inf_graph[:middle, :middle] = strong_influence
            inf_graph[middle:, middle:] = strong_influence
            for i in range(num_agents):
                inf_graph[i, i] = 1
            return inf_graph

        def two_influencers_balanced(num_agents, influencers_incoming_value, influencers_outgoing_value, others_belief_value):
            inf_graph = np.full((num_agents, num_agents), others_belief_value)
            inf_graph[0, :-1] = influencers_outgoing_value
            inf_graph[-1, 1:] = influencers_outgoing_value
            inf_graph[1:,0] = influencers_incoming_value
            inf_graph[:-1, -1] = influencers_incoming_value
            for i in range(num_agents):
                inf_graph[i, i] = 1
            return inf_graph

        def two_influencers_unbalanced(num_agents, influencers_outgoing_value_first, influencers_outgoing_value_second, influencers_incoming_value_first, influencers_incoming_value_second, others_belief_value):
            inf_graph = np.full((num_agents,num_agents), others_belief_value)
            inf_graph[0, :-1] = influencers_outgoing_value_first
            inf_graph[-1, 1:] = influencers_outgoing_value_second
            inf_graph[1:, 0] = influencers_incoming_value_first
            inf_graph[:-1, -1] = influencers_incoming_value_second
            for i in range(num_agents):
                inf_graph[i, i] = 1
            return inf_graph


class SLModel:
    state: State
    trust_graph: TrustGraph
    states: List[State]

    @property
    def num_agents(self):
        return len(self.state)

    def __init__(self, state: State, trust_graph: TrustGraph) -> None:
        self.state = state
        self.trust_graph = trust_graph
        self.states = [state]

    def update(self,
               fusion_operator: Callable[[List['HyperopinionInterface']], State],
               truster_index: int) -> HyperopinionInterface:
        truster_opinion = self.state[truster_index]
        trust_array: TrustArray = self.trust_graph[truster_index]
        discount_array = [trust_discount_2e(trust_array[trustee_index], self.state[trustee_index]) for trustee_index in
                          range(len(self.state)) if truster_index != trustee_index]
        # print("Discount array", [truster_opinion] + discount_array)
        new_opinion = fusion_operator([truster_opinion] + discount_array, epistemic=False)
        return new_opinion

    def overall_update(self, fusion_operator: Callable[[List['HyperopinionInterface']], State]) -> State:
        new_state: State = []
        for i in range(len(self.state)):
            new_state.append(self.update(fusion_operator, i))
        self.state = new_state
        return self.state
    
    @staticmethod
    def opinion_to_belief_state(opinion: HyperopinionInterface) -> float:
        return opinion.P[0]

    def belief_array(self):
        return [None if None else SLModel.opinion_to_belief_state(opinion) for opinion in self.state]
    
    def remove_dogmatic_opinions(self):
        for i in range(len(self.state)):
            if np.array_equal(self.state[i].b, [1, 0]):
                self.state[i] = Hyperopinion(2, [1 - epsilon, 0])
            elif np.array_equal(self.state[i].b, [0, 1]):
                self.state[i] = Hyperopinion(2, [0, 1 - epsilon])

    def simulate(self, n: int, fusion_operator: Callable[[List['HyperopinionInterface']], State]) -> None:
        for _ in range(n):
            self.overall_update(fusion_operator)
            self.remove_dogmatic_opinions()
            self.states += [self.state]


def belief_state_to_opinion(n: float) -> HyperopinionInterface:
    opinion = Hyperopinion(2, [n, 1 - n]).maximize_uncertainty()
    if opinion.b[0] == 1:
        opinion = Hyperopinion(2, [1 - epsilon, 0])
    elif opinion.b[1] == 1:
        opinion = Hyperopinion(2, [0, 1 - epsilon])
    return opinion

def akv_to_sl(akv_model: AKVModel) -> SLModel:
    n_agents = len(akv_model.belief_array)
    state: State = [belief_state_to_opinion(belief_state) for belief_state in akv_model.belief_array]
    trust_graph: TrustGraph = []
    for i in range(n_agents):
        trust_array: TrustArray = []
        for j in range(n_agents):
            if i != j:
                influence = akv_model.influence_graph[j][i]
                trust_array.append(Hyperopinion(2, [influence, 1 - influence]))
            else:
                trust_array.append(None)
        trust_graph.append(trust_array)
    return SLModel(state, trust_graph)


def plot(akv_states: List[BeliefArray], sl_states: List[State], num_agents: int, num_steps: int):
    fig, ax = plt.subplots(1, 2)
    fig.set(figwidth=16)
    ax[0].set_title("Old model")
    ax[1].set_title("SL model")
    for i in range(num_agents):
        ax[0].plot(list(range(num_steps + 1)), [akv_states[j][i] for j in range(num_steps + 1)])
        ax[1].plot(list(range(num_steps + 1)), [sl_states[j][i].P[0] for j in range(num_steps + 1)])
    return fig, ax


# def update(
#         fusion_operator: Callable[[List['HyperopinionInterface']], State],
#         truster_index: int,
#         trust_array: TrustArray,
#         state: State) -> HyperopinionInterface:
#     truster_opinion = state[truster_index]
#     discount_array = [trust_discount_2e(trust_array[trustee_index], state[trustee_index])
#                       for trustee_index in range(len(state)) if truster_index != trustee_index]
#     new_opinion = fusion_operator([truster_opinion] + discount_array)
#     return new_opinion


# def classic_update(
#         belief_state_i: float,
#         belief_state_j: float,
#         influence: float):
#     return belief_state_i + influence * (belief_state_j - belief_state_i)


# def overall_classic_update(
#         belief_array,
#         influence_graph):
#     A = len(belief_array)
#     new_belief_array = np.array([])
#     for ai in range(A):
#         new_belief_array = np.append(new_belief_array, (1 / A) * np.sum(
#             [classic_update(belief_array[ai], belief_array[aj], influence_graph[ai][aj]) for aj in range(A)]))
#     return new_belief_array
