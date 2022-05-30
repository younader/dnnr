from __future__ import annotations

import abc
from typing import Any, TypeVar, Union, cast

import numpy as np
from sklearn.neighbors import KDTree

T = TypeVar('T')


class BaseIndex(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def build(cls: type[T], x: np.ndarray, **kwargs: dict[str, Any]) -> T:
        """Builds the index.

        Args:
            x: with shape (n_samples, n_features). Used to build the
                index.
            **kwargs: Additional arguments passed to the index.
        """
        pass

    @abc.abstractmethod
    def query_knn(self, v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the (indices, distances) of the k nearest neighbors of v.

        Args:
            v: with shape (n_samples, n_features)
            k: number of neighbors to return

        Returns:
            A tuple of (indices, distances) of the k nearest neighbors of v.
        """
        pass


class KDTreeIndex(BaseIndex):
    @classmethod
    def build(cls, x: np.ndarray, **kwargs: dict[str, Any]) -> KDTreeIndex:
        return cls(x, **kwargs)

    def __init__(self, x: np.ndarray, **kwargs: dict[str, Any]) -> None:
        self.index = KDTree(x, **kwargs)

    def query_knn(self, v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        distances, indices = self.index.query([v], k=k)
        return indices[0], distances[0][0]


class AnnoyIndex(BaseIndex):
    def __init__(self, vector_length: int, metric: str = "euclidean") -> None:
        import annoy

        super().__init__()

        self.vector_length = vector_length
        self.metric = metric
        self.index = annoy.AnnoyIndex(self.vector_length, self.metric)

    @classmethod
    def build(cls, x: np.ndarray, **kwargs: dict[str, Any]) -> AnnoyIndex:
        n_trees = kwargs.get('n_trees', 50)
        metric = cast(str, kwargs.get('metric', 'euclidean'))
        index = AnnoyIndex(x.shape[1], metric=metric)

        for i, v in zip(range(len(x)), x):
            index.index.add_item(i, v)
        index.index.build(n_trees)
        return index

    def query_knn(self, v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
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
