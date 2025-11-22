"""
kmeans.py
A small, dependency-light KMeans implementation that works with pandas.DataFrame.
Features / fixes:
- Robust euclidean distance handling for (Series, Series) and (DataFrame, Series).
- k-means++ style centroid initialization (farthest-first variant from given algorithm).
- Correct grouping, centroid update with safe handling of empty clusters (retain previous centroid).
- Proper stopping criteria (stop when groups OR centroids don't change, or max iterations reached).
- Clear type hints and docstrings.
"""

from typing import Optional
import numpy as np
import pandas as pd


class KMeans:
    def __init__(self, k: int, start_var: int, end_var: int, num_observations: int, data: pd.DataFrame, random_state: Optional[int] = 1):
        """
        Args:
            k: number of clusters
            start_var: start column index (inclusive) for clustering features
            end_var: end column index (exclusive) for clustering features
            num_observations: expected number of rows in the data (used for initialization shapes)
            data: original DataFrame (used for column names)
            random_state: seed for reproducibility
        """
        if not (0 <= start_var < end_var <= data.shape[1]):
            raise ValueError("Invalid start_var/end_var for data columns.")
        self.k = int(k)
        self.start_var = int(start_var)
        self.end_var = int(end_var)
        self.num_observations = int(num_observations)
        self.columns = list(data.columns[self.start_var:self.end_var])
        self.centroids = pd.DataFrame(columns=self.columns)
        self.rng = np.random.RandomState(random_state)

    # -----------------------
    # Utility: Euclidean distance
    # -----------------------
    def _to_float_series(self, s: pd.Series) -> pd.Series:
        return s.astype(float)

    def get_euclidean_distance(self, point1, point2):
        """
        Compute Euclidean distance.

        Allowed shapes:
          - Series, Series -> scalar float
          - DataFrame, Series -> Series of row-wise distances

        Behaviour:
          - Aligns series indices / dataframe columns to guarantee correct subtraction order.
        """
        # Series & Series
        if isinstance(point1, pd.Series) and isinstance(point2, pd.Series):
            p1 = self._to_float_series(point1.reindex(point2.index))
            p2 = self._to_float_series(point2)
            diff = p1 - p2
            return float(np.sqrt((diff ** 2).sum()))

        # DataFrame & Series (row-wise broadcast)
        if isinstance(point1, pd.DataFrame) and isinstance(point2, pd.Series):
            df = point1.astype(float, copy=True)
            s = self._to_float_series(point2.reindex(df.columns))
            diff = df - s
            # sum across columns (axis=1)
            return np.sqrt((diff ** 2).sum(axis=1)).reset_index(drop=True)

        raise TypeError("get_euclidean_distance only supports (Series, Series) or (DataFrame, Series).")

    # -----------------------
    # Initialization
    # -----------------------
    def initialize_centroids(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Initialize centroids with a farthest-first (kmeans++ like) approach:
        1. choose one random point
        2. for each remaining centroid, choose the point with maximum distance to nearest centroid
        Returns DataFrame of shape (k, n_features)
        """
        features = data.iloc[:, self.start_var:self.end_var].reset_index(drop=True).astype(float)
        n = features.shape[0]
        if self.k > n:
            raise ValueError("k cannot be greater than number of observations.")

        centroids = []
        # pick first randomly
        first_idx = self.rng.randint(0, n)
        centroids.append(features.iloc[first_idx].copy())

        # pick remaining
        for _ in range(1, self.k):
            # compute distance of each point to nearest centroid
            dists = pd.DataFrame()
            for c in centroids:
                dists = pd.concat([dists, self.get_euclidean_distance(features, c)], axis=1) if not dists.empty else self.get_euclidean_distance(features, c)
            # if dists is Series (only one centroid) ensure DataFrame
            if isinstance(dists, pd.Series):
                dists = dists.to_frame(0)
            min_dists = dists.min(axis=1)
            next_idx = int(min_dists.idxmax())
            centroids.append(features.iloc[next_idx].copy())

        cent_df = pd.DataFrame(centroids).reset_index(drop=True)
        cent_df.columns = self.columns
        self.centroids = cent_df
        return self.centroids

    # -----------------------
    # Assign groups
    # -----------------------
    def group_observations(self, data: pd.DataFrame) -> pd.Series:
        """
        For each data row (using feature slice), compute nearest centroid index.
        Returns: pd.Series of length = data.shape[0] with dtype int32.
        """
        features = data.iloc[:, self.start_var:self.end_var].reset_index(drop=True).astype(float)
        if self.centroids.empty:
            raise RuntimeError("Centroids are not initialized.")
        distances = pd.DataFrame(index=features.index)
        for i in range(self.k):
            distances[i] = self.get_euclidean_distance(features, self.centroids.iloc[i])
        groups = distances.idxmin(axis=1).astype('int32')
        return groups

    # -----------------------
    # Adjust centroids
    # -----------------------
    def adjust_centroids(self, data: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
        """
        Compute new centroids as mean of assigned points.
        If a cluster has zero points, keep previous centroid for that cluster.
        Returns DataFrame of shape (k, n_features)
        """
        features = data.iloc[:, self.start_var:self.end_var].reset_index(drop=True).astype(float)
        grouped = pd.concat([features, groups.reset_index(drop=True).rename('group')], axis=1)
        centroids = grouped.groupby('group').mean()

        # ensure index 0..k-1 exist
        centroids = centroids.reindex(range(self.k))

        # if any centroid row is NaN (empty cluster), replace with previous centroid
        if not self.centroids.empty:
            prev = self.centroids.reset_index(drop=True).astype(float)
            # fill missing rows from prev (alignment by columns)
            for col in self.columns:
                if col not in centroids.columns:
                    centroids[col] = np.nan
            centroids = centroids[self.columns]
            # fill NaNs row-wise from prev
            centroids = centroids.fillna(prev)
        else:
            # if no previous centroids (shouldn't happen), fill missing with zeros
            centroids = centroids[self.columns].fillna(0.0)

        centroids = centroids.reset_index(drop=True)
        return centroids

    # -----------------------
    # Train
    # -----------------------
    def train(self, data: pd.DataFrame, iters: int = 100) -> pd.Series:
        """
        Run k-means clustering.
        Stopping: stop when groups do not change OR centroids do not change OR max iterations reached.
        Returns final groups (pd.Series).
        """
        if self.centroids.empty:
            self.initialize_centroids(data)

        prev_groups = pd.Series([-1] * data.shape[0], dtype='int32')
        for i in range(1, iters + 1):
            groups = self.group_observations(data)
            centroids = self.adjust_centroids(data, groups)

            groups_unchanged = groups.equals(prev_groups)
            centroids_unchanged = centroids.reset_index(drop=True).equals(self.centroids.reset_index(drop=True))

            self.centroids = centroids
            prev_groups = groups

            # Stop if *either* groups or centroids unchanged
            if groups_unchanged or centroids_unchanged:
                # print debug info
                # print(f"Stopped at iteration {i}: groups_unchanged={groups_unchanged}, centroids_unchanged={centroids_unchanged}")
                break

        return prev_groups