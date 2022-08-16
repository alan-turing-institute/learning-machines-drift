"""
HyperTransformer class available only in SDMetrics>=0.6.0 where KSTest is removed.
https://github.com/sdv-dev/SDMetrics/blob/06b99e71a4d7a5444d0a2712282f20a949fc5d23/sdmetrics/utils.py#L125-L216

Including here as importable learning_machines_drift code pending resolution of missing function.
"""
import numpy as np
import pandas as pd


class HyperTransformer:
    """HyperTransformer class.
    The ``HyperTransformer`` class contains a set of transforms to transform one or
    more columns based on each column's data type.
    """

    column_transforms = {}
    column_kind = {}

    def fit(self, data):
        """Fit the HyperTransformer to the given data.
        Args:
            data (pandas.DataFrame):
                The data to transform.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        for field in data:
            kind = data[field].dropna().infer_objects().dtype.kind
            self.column_kind[field] = kind

            if kind == "i" or kind == "f":
                # Numerical column.
                self.column_transforms[field] = {"mean": data[field].mean()}
            elif kind == "b":
                # Boolean column.
                numeric = pd.to_numeric(data[field], errors="coerce").astype(float)
                self.column_transforms[field] = {"mode": numeric.mode().iloc[0]}
            elif kind == "O":
                # Categorical column.
                col_data = pd.DataFrame({"field": data[field]})
                enc = OneHotEncoder()
                enc.fit(col_data)
                self.column_transforms[field] = {"one_hot_encoder": enc}
            elif kind == "M":
                # Datetime column.
                nulls = data[field].isna()
                integers = (
                    pd.to_numeric(data[field], errors="coerce")
                    .to_numpy()
                    .astype(np.float64)
                )
                integers[nulls] = np.nan
                self.column_transforms[field] = {"mean": pd.Series(integers).mean()}

    def transform(self, data):
        """Transform the given data based on the data type of each column.
        Args:
            data (pandas.DataFrame):
                The data to transform.
        Returns:
            pandas.DataFrame:
                The transformed data.
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        for field in data:
            transform_info = self.column_transforms[field]

            kind = self.column_kind[field]
            if kind == "i" or kind == "f":
                # Numerical column.
                data[field] = data[field].fillna(transform_info["mean"])
            elif kind == "b":
                # Boolean column.
                data[field] = pd.to_numeric(data[field], errors="coerce").astype(float)
                data[field] = data[field].fillna(transform_info["mode"])
            elif kind == "O":
                # Categorical column.
                col_data = pd.DataFrame({"field": data[field]})
                out = transform_info["one_hot_encoder"].transform(col_data).toarray()
                transformed = pd.DataFrame(
                    out, columns=[f"value{i}" for i in range(np.shape(out)[1])]
                )
                data = data.drop(columns=[field])
                data = pd.concat([data, transformed.set_index(data.index)], axis=1)
            elif kind == "M":
                # Datetime column.
                nulls = data[field].isna()
                integers = (
                    pd.to_numeric(data[field], errors="coerce")
                    .to_numpy()
                    .astype(np.float64)
                )
                integers[nulls] = np.nan
                data[field] = pd.Series(integers)
                data[field] = data[field].fillna(transform_info["mean"])

        return data

    def fit_transform(self, data):
        """Fit and transform the given data based on the data type of each column.
        Args:
            data (pandas.DataFrame):
                The data to transform.
        Returns:
            pandas.DataFrame:
                The transformed data.
        """
        self.fit(data)
        return self.transform(data)
