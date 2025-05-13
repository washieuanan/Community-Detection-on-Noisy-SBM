from abc import (ABC, abstractmethod)
from typing import List, Tuple
import numpy as np 
import networkx as nx 

class Observation(ABC):
    ''' 
    Abstract base class for observation methods. 
    Subclasses must implement the observe method. 
    '''

    def __init__(self, graph: nx.Graph, seed: int):
        self.graph = graph
        self.rng = np.random.default_rng(seed)
        self.observations = []

    @abstractmethod
    def observe(self):
        '''
        Perform observation on the graph. Must return a list of node-pair observations. 

        Returns: 
    
        something
        '''
        raise NotImplementedError("Subclasses must implement the observe method")