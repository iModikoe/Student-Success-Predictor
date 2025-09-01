from __future__ import annotations
import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def train_test_split_df(df, target, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) == 2 else None
    )
    return X_train, X_test, y_train, y_test
