import pandas as pd       
import numpy as np
from sklearn.decomposition import PCA


class DimensionalityReducer:
    """Class to apply dimensionality reduction methods (currently only PCA).
    1. Write a function to convert the high dimensional data to 2 dimensional."""

    def __init__(self, method: str, df: pd.DataFrame, n_components: int = 2):
        self.method = method.upper()
        self.df = df
        self.n_components = n_components

    def transform(self) -> np.ndarray:
        """
        Apply the specified reduction method and return an array of shape (n_samples, n_components).
        """
        if self.method == "PCA":
            pca = PCA(n_components=self.n_components)
            return pca.fit_transform(self.df.values)
        raise ValueError(f"Unknown reduction method: {self.method}")
    
    