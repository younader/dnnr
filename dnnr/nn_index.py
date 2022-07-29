from __future__ import annotations

import abc
import dataclasses
from typing import Any, TypeVar, Union

import numpy as np
import sklearn.base
import sklearn.neighbors

T = TypeVar('T')


class BaseIndex(sklearn.base.BaseEstimator, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, x: np.ndarray) -> None:
        """Builds the index.

        Args:
            x: with shape (n_samples, n_features). Used to build the
                index.
            **kwargs: Additional arguments passed to the index.
        """

    @abc.abstractmethod
    def query_knn(self, v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the (indices, distances) of the k nearest neighbors of v.

        Args:
            v: with shape (n_samples, n_features)
            k: number of neighbors to return

        Returns:
            A tuple of (indices, distances) of the k nearest neighbors of v.
        """


@dataclasses.dataclass
class KDTreeIndex(BaseIndex):
    metric: str = "euclidean"
    leaf_size: int = 40
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def fit(self, x: np.ndarray) -> None:
        self.index = sklearn.neighbors.KDTree(
            x, metric=self.metric, leaf_size=self.leaf_size, **self.kwargs
        )

    def query_knn(self, v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "index"):
            raise ValueError("Index not fitted.")
        distances, indices = self.index.query([v], k=k)
        return indices[0], distances[0][0]


@dataclasses.dataclass
class AnnoyIndex(BaseIndex):
    metric: str = "euclidean"
    n_trees: int = 50

    def fit(self, x: np.ndarray) -> None:
        import annoy

        self.index = annoy.AnnoyIndex(x.shape[1], self.metric)

        for i, v in zip(range(len(x)), x):
            self.index.add_item(i, v)
        self.index.build(self.n_trees)

    def query_knn(self, v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "index"):
            raise ValueError("Index not fitted.")

        indices, distances = self.index.get_nns_by_vector(
            v, k, include_distances=True
        )
        return indices, distances


def get_index_class(index: Union[type[BaseIndex], str]) -> type[BaseIndex]:
    """Returns the corresponding index class based on the passed string.

    Args:
        index: either a string of the index name or a class
    """
    if isinstance(index, type) and issubclass(index, BaseIndex):
        return index

    if index == "annoy":
        return AnnoyIndex
    elif index == "kd_tree":
        return KDTreeIndex
    else:
        raise ValueError(f"Index {index} not supported")
