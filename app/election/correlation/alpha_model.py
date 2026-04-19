import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
import joblib
from pathlib import Path

from app.election.models import AlphaModelPrediction
from app.election.correlation.feature_builder import build_feature_matrix, clean_features


def run_alpha_model(
    db: Session,
    race_id: int,
    model: Optional[object] = None,
    model_version: str = "default",
    lookback_days: int = 180
) -> Optional[float]:
    """
    Run alpha model prediction for a specific race.
    
    Args:
        db: Database session
        race_id: Race ID to predict
        model: Pre-trained model to use (if None, trains new model)
        model_version: Version identifier for the model
        lookback_days: Number of days to look back for features
    
    Returns:
        Predicted probability delta or None if prediction fails
    """
    try:
        current_year = datetime.now().year
        feature_df = build_feature_matrix(db, current_year, lookback_days)
        if feature_df is None or feature_df.empty:
            return None
            
        feature_df_clean = clean_features(feature_df)
        
        if 'race_id' in feature_df_clean.columns:
            race_features = feature_df_clean[feature_df_clean['race_id'] == race_id]
        elif feature_df_clean.index.name == 'race_id' or 'race_id' in feature_df_clean.index.names:
            race_features = feature_df_clean.loc[feature_df_clean.index == race_id]
        else:
            race_features = feature_df_clean
            
        if race_features.empty:
            return None
            
        target_col = 'target'
        drop_cols = [c for c in [target_col, 'date', 'race_id'] if c in race_features.columns]
        X = race_features.drop(columns=drop_cols).fillna(0)
        
        if model is None:
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import RidgeCV
            
            train_cycles = [current_year - 2, current_year - 4]
            train_dfs = []
            for cycle in train_cycles:
                train_df = build_feature_matrix(db, cycle, lookback_days)
                if train_df is not None and not train_df.empty:
                    train_dfs.append(train_df)
                    
            if not train_dfs:
                return None
                
            train_df_all = pd.concat(train_dfs, ignore_index=True)
            train_df_clean = clean_features(train_df_all)
            
            if target_col not in train_df_clean.columns:
                return None
                
            drop_cols_train = [c for c in [target_col, 'date', 'race_id'] if c in train_df_clean.columns]
            X_train = train_df_clean.drop(columns=drop_cols_train).fillna(0)
            y_train = train_df_clean[target_col]
            
            common_cols = X_train.columns.intersection(X.columns)
            if len(common_cols) == 0:
                return None
                
            X_train = X_train[common_cols]
            X = X[common_cols]
            
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RidgeCV(alphas=[0.1, 1.0, 10.0]))
            ])
            model.fit(X_train.values, y_train.values)
            model_version = "ridgecv_fallback"
            
        prediction = float(model.predict(X.values)[0])
        prediction = np.clip(prediction, -1, 1)
        
        db_prediction = AlphaModelPrediction(
            race_id=race_id,
            predicted_delta=prediction,
            model_version=model_version,
            created_at=datetime.utcnow()
        )
        db.add(db_prediction)
        db.commit()
        
        return prediction
        
    except Exception as e:
        print(f"Error in alpha model prediction: {e}")
        return None


def load_trained_model(model_name: str, version: str = "v1") -> Optional[object]:
    """Load a pre-trained model from disk."""
    model_path = Path(__file__).parent / "trained_models" / f"{model_name}_{version}.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None
