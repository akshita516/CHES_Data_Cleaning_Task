from pathlib import Path
from typing import List
from urllib.request import urlretrieve

import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """Class to load and preprocess the CHES political parties dataset."""

    data_url: str = "https://www.chesdata.eu/s/CHES2019V3.dta"

    def __init__(self):
        # Download raw data into self.party_data
        self.party_data: pd.DataFrame = self._download_data()
        # Columns to drop prior to modeling
        self.non_features: List[str] = []
        # Columns to set as DataFrame index
        self.index: List[str] = ["party_id", "party", "country"]

    def _download_data(self) -> pd.DataFrame:
        """Retrieve the CHES .dta file and load into a pandas DataFrame."""
        data_path, _ = urlretrieve(
            self.data_url,
            Path(__file__).parents[2].joinpath("data", "CHES2019V3.dta"),
        )
        return pd.read_stata(data_path)

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows based on the DataFrame's current index."""
        return df[~df.index.duplicated(keep='first')]

    def remove_nonfeature_cols(
        self, df: pd.DataFrame, non_features: List[str], index: List[str]
    ) -> pd.DataFrame:
        """Drop non-feature columns and set the specified index columns."""
        df = df.drop(columns=non_features, errors='ignore')
        df = df.set_index(index)
        return df

    def handle_NaN_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that are entirely NaN and impute remaining NaNs with column mean."""
        df = df.dropna(axis=1, how='all')
        numeric = df.select_dtypes(include='number').columns
        # Use .loc to avoid SettingWithCopyWarning
        df.loc[:, numeric] = df[numeric].fillna(df[numeric].mean())
        return df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric features using StandardScaler."""
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            scaler = StandardScaler()
            # Use .loc to avoid SettingWithCopyWarning
            df.loc[:, numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df

    def preprocess_data(self) -> pd.DataFrame:
        """Combine all preprocessing steps and return cleaned DataFrame."""
        df = self.party_data.copy()
        df = self.remove_nonfeature_cols(df, self.non_features, self.index)
        df = self.handle_NaN_values(df)
        df = self.remove_duplicates(df)
        df = self.scale_features(df)
        self.party_data = df
        return df