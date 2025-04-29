from typing import List, Tuple, Any
import os
import sys
from abc import ABC, abstractmethod
import numpy as np
import networkx as nx


class Detection(ABC):
    """
    Abstract base class for detection methods.
    Subclasses must implement the output method.

    Parameters
    ----------
    graph : nx.Graph
        The underlying graph on which detection is performed.
    observations : List[List[Any]]
        A list of observed node-pairs; each observation is a length-2 list/tuple.
    """

    def __init__(
        self,
        graph: nx.Graph,
        observations: List[Tuple[Any, Any]], 
        k: int
    ):
        if not isinstance(graph, nx.Graph):
            raise TypeError("`graph` must be a networkx.Graph instance")
        if not isinstance(observations, list) or \
           not all(isinstance(obs, (list, tuple)) and len(obs) == 2 for obs in observations):
            raise ValueError(
                "`observations` must be a list of length-2 lists or tuples"
            )

        self.graph = graph
        self.observations = observations
        self.k = k

    @abstractmethod
    def output(self) -> np.ndarray:
        """
        Perform detection/inference based on the graph and observations.

        Returns
        -------
        result : np.ndarray
            A NumPy array of detection outputs (e.g., labels, scores, etc.).
        """
        raise NotImplementedError("Subclasses must implement the output method")
