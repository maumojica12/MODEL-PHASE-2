"""
rule_miner.py
Apriori-style frequent itemset miner and rule generator.

Design choices & fixes:
- get_support treats columns as boolean indicators. If not boolean, it coerces:
    - numeric: non-zero -> True
    - object/string: non-null and not empty -> True
- merge_itemsets uses proper Apriori join (only join itemsets that share first k-1 items).
- get_frequent_itemsets uses support threshold as absolute count.
- get_association_rules returns rules as tuples (X, Y, support_XY, confidence)
- avoids duplicate symmetric rules unless they are genuinely different antecedent/consequent pairs.
"""

from typing import List, Tuple
import pandas as pd
import itertools


class RuleMiner:
    def __init__(self, support_t: int, confidence_t: float):
        """
        Args:
            support_t: minimum support count (absolute number of rows)
            confidence_t: minimum confidence (0..1)
        """
        self.support_t = int(support_t)
        if not (0.0 <= confidence_t <= 1.0):
            raise ValueError("confidence_t must be between 0 and 1.")
        self.confidence_t = float(confidence_t)

    def _to_boolean_df(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert dataframe columns to boolean indicators for presence of item.
        - bool columns kept
        - numeric -> non-zero True
        - object/string -> non-empty, non-null True
        """
        df_bool = pd.DataFrame(index=data.index)
        for col in data.columns:
            series = data[col]
            if pd.api.types.is_bool_dtype(series):
                df_bool[col] = series.fillna(False).astype(bool)
            elif pd.api.types.is_numeric_dtype(series):
                df_bool[col] = (series.fillna(0) != 0)
            else:
                # treat non-empty strings / non-null as True
                df_bool[col] = series.fillna("").astype(str).str.strip().ne("")
        return df_bool

    def get_support(self, data: pd.DataFrame, itemset: List[str]) -> int:
        """
        Return support count (number of rows where ALL items in itemset are True).
        """
        df_bool = self._to_boolean_df(data)
        if len(itemset) == 0:
            return 0
        subset = df_bool[itemset]
        # all True across columns
        return int(subset.all(axis=1).sum())

    def merge_itemsets(self, itemsets: List[List[str]]) -> List[List[str]]:
        """
        Given frequent (k)-itemsets (list-of-lists with sorted items),
        produce candidate (k+1)-itemsets by joining pairs that share first k-1 items.
        This prevents generating spurious candidates.
        """
        if not itemsets:
            return []
        k = len(itemsets[0])
        if k == 1:
            # simple pairwise union but keep order
            new = []
            n = len(itemsets)
            for i in range(n):
                for j in range(i + 1, n):
                    cand = sorted(list(set(itemsets[i]) | set(itemsets[j])))
                    if cand not in new:
                        new.append(cand)
            return new

        new_itemsets = []
        # join if first k-1 items equal (itemsets are sorted)
        itemsets_sorted = [sorted(it) for it in itemsets]
        itemsets_sorted = sorted(itemsets_sorted)
        for i in range(len(itemsets_sorted)):
            for j in range(i + 1, len(itemsets_sorted)):
                a = itemsets_sorted[i]
                b = itemsets_sorted[j]
                if a[:k-1] == b[:k-1]:
                    cand = sorted(list(set(a) | set(b)))
                    if len(cand) == k + 1 and cand not in new_itemsets:
                        new_itemsets.append(cand)
        return new_itemsets

    def get_frequent_itemsets(self, data: pd.DataFrame) -> List[List[str]]:
        """
        Apriori main loop returning list of all frequent itemsets (as lists).
        Uses absolute support threshold (support_t).
        """
        df_bool = self._to_boolean_df(data)
        # start with singletons
        itemsets = [[col] for col in df_bool.columns]
        frequent = []
        current = itemsets
        while current:
            next_freq = []
            for it in current:
                sup = self.get_support(df_bool, it)
                if sup >= self.support_t:
                    next_freq.append(it)
            if not next_freq:
                break
            # add next_freq to final frequent list
            frequent.extend(next_freq)
            # generate candidates for next size
            current = self.merge_itemsets(next_freq)
        return frequent

    def get_confidence(self, data: pd.DataFrame, X: List[str], Y: List[str]) -> float:
        """
        confidence(X -> Y) = support(X ∪ Y) / support(X)
        """
        df_bool = self._to_boolean_df(data)
        sup_X = self.get_support(df_bool, X)
        if sup_X == 0:
            return 0.0
        sup_XY = self.get_support(df_bool, X + Y)
        return float(sup_XY / sup_X)

    def get_association_rules(self, data: pd.DataFrame) -> List[Tuple[List[str], List[str], int, float]]:
        """
        Returns list of rules as tuples: (antecedent_list, consequent_list, support_XY, confidence)
        Only returns rules with confidence >= confidence_t.

        Note: support returned is support(X ∪ Y) as absolute count.
        """
        df_bool = self._to_boolean_df(data)
        frequent_itemsets = self.get_frequent_itemsets(df_bool)
        rules = []
        for itemset in frequent_itemsets:
            if len(itemset) < 2:
                continue
            # generate all non-empty proper subsets for antecedent
            for r in range(1, len(itemset)):
                for antecedent in itertools.combinations(itemset, r):
                    antecedent = list(antecedent)
                    consequent = sorted(list(set(itemset) - set(antecedent)))
                    confidence = self.get_confidence(df_bool, antecedent, consequent)
                    support_xy = self.get_support(df_bool, antecedent + consequent)
                    if confidence >= self.confidence_t:
                        rules.append((sorted(antecedent), consequent, int(support_xy), float(confidence)))
        return rules