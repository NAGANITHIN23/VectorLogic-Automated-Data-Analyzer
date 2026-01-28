import pandas as pd
import numpy as np
import featuretools as ft
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    
    def __init__(self):
        self.feature_matrix = None
        self.feature_defs = None
        
    def generate_features(self, df: pd.DataFrame, max_depth: int = 2) -> Tuple[pd.DataFrame, List[str]]:
        try:
            if df.empty or len(df.columns) == 0:
                return df, []
            
            df_clean = df.copy()
            
            if 'index' not in df_clean.columns:
                df_clean['index'] = range(len(df_clean))
            
            es = ft.EntitySet(id='data')
            
            es = es.add_dataframe(
                dataframe_name='main',
                dataframe=df_clean,
                index='index'
            )
            
            feature_matrix, feature_defs = ft.dfs(
                entityset=es,
                target_dataframe_name='main',
                max_depth=max_depth,
                verbose=False
            )
            
            self.feature_matrix = feature_matrix
            self.feature_defs = feature_defs
            
            new_features = [f.get_name() for f in feature_defs if f.get_name() not in df.columns]
            
            logger.info(f'Generated {len(new_features)} new features')
            
            return feature_matrix, new_features
            
        except Exception as e:
            logger.error(f'Feature generation failed: {str(e)}')
            return df, []
    
    def rank_features(self, X: pd.DataFrame, y: pd.Series = None, method: str = 'variance') -> List[Dict[str, Any]]:
        try:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return []
            
            X_numeric = X[numeric_cols].fillna(0)
            
            if method == 'variance':
                scores = X_numeric.var()
            elif method == 'correlation' and y is not None:
                scores = X_numeric.corrwith(y).abs()
            else:
                scores = X_numeric.var()
            
            scores = scores.sort_values(ascending=False)
            
            ranked_features = []
            for idx, (feature, score) in enumerate(scores.items(), 1):
                ranked_features.append({
                    'rank': idx,
                    'feature': feature,
                    'score': float(score),
                    'method': method
                })
            
            return ranked_features
            
        except Exception as e:
            logger.error(f'Feature ranking failed: {str(e)}')
            return []
    
    def get_feature_statistics(self, feature_matrix: pd.DataFrame, top_n: int = 20) -> Dict[str, Any]:
        try:
            numeric_features = feature_matrix.select_dtypes(include=[np.number])
            
            stats = {
                'total_features': len(feature_matrix.columns),
                'numeric_features': len(numeric_features.columns),
                'missing_values': int(feature_matrix.isnull().sum().sum()),
                'feature_types': {}
            }
            
            for col in feature_matrix.columns[:top_n]:
                stats['feature_types'][col] = str(feature_matrix[col].dtype)
            
            return stats
            
        except Exception as e:
            logger.error(f'Feature statistics failed: {str(e)}')
            return {}