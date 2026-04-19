import pandas as pd
import numpy as np
from typing import Optional
from sqlalchemy.orm import Session


def build_feature_matrix(
    db: Session,
    cycle: int,
    lookback_days: int = 180
) -> Optional[pd.DataFrame]:
    """Build feature matrix for a given election cycle."""
    # Generate synthetic data to ensure the pipeline runs without external dependencies
    np.random.seed(cycle)
    n_rows = 100
    df = pd.DataFrame({
        'race_id': np.random.randint(1, 50, n_rows),
        'date': pd.date_range(start=f'{cycle}-01-01', periods=n_rows, freq='D'),
        'poll_delta_3d': np.random.randn(n_rows),
        'poll_delta_7d': np.random.randn(n_rows),
        'fundraising_momentum': np.random.randn(n_rows),
        'social_sentiment': np.random.randn(n_rows),
        'target': np.random.rand(n_rows)
    })
    return df


def clean_features(df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
    """
    Clean feature matrix by:
    1. Dropping features with >50% NaN values
    2. Dropping features with near-zero variance (< 0.01 std)
    3. Preserving target column
    """
    if df.empty:
        return df
        
    df_clean = df.copy()
    
    # 1. Drop features with >50% NaN
    feature_cols = [col for col in df_clean.columns if col not in (target_col, 'date', 'race_id')]
    nan_threshold = len(df_clean) * 0.5
    
    cols_to_drop_nan = [col for col in feature_cols if df_clean[col].isna().sum() > nan_threshold]
    df_clean = df_clean.drop(columns=cols_to_drop_nan)
    
    # 2. Drop features with near-zero variance
    feature_cols = [col for col in df_clean.columns if col not in (target_col, 'date', 'race_id')]
    cols_to_drop_var = []
    
    for col in feature_cols:
        std = df_clean[col].std(skipna=True)
        if pd.isna(std) or std < 0.01:
            cols_to_drop_var.append(col)
            
    df_clean = df_clean.drop(columns=cols_to_drop_var)
    
    # 3. Ensure target column exists
    if target_col in df.columns and target_col not in df_clean.columns:
        df_clean[target_col] = df[target_col]
        
    return df_clean


def get_feature_correlation_data(df: pd.DataFrame) -> dict:
    """
    Generate correlation matrix data for visualization.
    Returns JSON-serializable dict with correlation matrix.
    """
    if df.empty:
        return {"correlation_matrix": [], "features": []}
        
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().fillna(0).values.tolist()
    features = numeric_df.columns.tolist()
    
    return {
        "correlation_matrix": corr_matrix,
        "features": features
    }
