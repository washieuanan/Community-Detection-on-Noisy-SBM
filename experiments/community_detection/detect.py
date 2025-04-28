import os
import sys
from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from typing import List, Tuple, Any

# Ensure repo root is on sys.path for package imports
_repo_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


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
        observations: List[Tuple[Any, Any]]
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
