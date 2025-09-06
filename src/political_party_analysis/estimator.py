import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture


class DensityEstimator:
    """Class to estimate Density/Distribution of the given data.
    
    1. Model the distribution of the political party dataset in reduced dimensions
    2. Randomly sample 10 parties from this distribution
    3. Map the sampled parties back to the original high-dimensional space
    """

    def __init__(self, data: pd.DataFrame, dim_reducer, high_dim_feature_names):
        """
        Args:
            data (pd.DataFrame): 2D reduced data (e.g., output of PCA)
            dim_reducer: Dimensionality reducer instance (must have `.model` with `inverse_transform`)
            high_dim_feature_names (list): Names of the original high-dimensional features
        """
        self.data = data
        self.dim_reducer_model = dim_reducer.model  # e.g., a PCA instance
        self.feature_names = high_dim_feature_names
        self.gmm = None  # to hold the fitted density model

    def fit_density(self, n_components: int = 1):
        """Fits a Gaussian Mixture Model (GMM) to the reduced data."""
        self.gmm = GaussianMixture(n_components=n_components, random_state=42)
        self.gmm.fit(self.data)
    
    def sample_points(self, n_samples: int = 10) -> pd.DataFrame:
        """Samples new points from the fitted GMM model.
        
        Returns:
            pd.DataFrame: Sampled data in the 2D reduced space.
        """
        if self.gmm is None:
            raise ValueError("You must fit the density model first using `fit_density()`.")
        samples, _ = self.gmm.sample(n_samples)
        return pd.DataFrame(samples, columns=self.data.columns)

    def inverse_transform(self, sampled_df: pd.DataFrame) -> pd.DataFrame:
        """Maps the sampled 2D data back to the original high-dimensional space.

        Args:
            sampled_df (pd.DataFrame): Data sampled from the reduced 2D space.

        Returns:
            pd.DataFrame: Inverse transformed data in the original high-dimensional feature space.
        """
        original_data = self.dim_reducer_model.inverse_transform(sampled_df)
        return pd.DataFrame(original_data, columns=self.feature_names)
