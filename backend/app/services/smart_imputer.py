import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

class SmartImputer:
    
    def __init__(self):
        self.imputers = {}
        self.encoders = {}
        self.imputation_log = []
        
    def impute_simple(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        try:
            df_imputed = df.copy()
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if len(numeric_cols) > 0 and strategy in ['mean', 'median']:
                numeric_imputer = SimpleImputer(strategy=strategy)
                df_imputed[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
                
                self.imputation_log.append({
                    'method': 'simple',
                    'strategy': strategy,
                    'columns': list(numeric_cols)
                })
            
            if len(categorical_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df_imputed[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
                
                self.imputation_log.append({
                    'method': 'simple',
                    'strategy': 'most_frequent',
                    'columns': list(categorical_cols)
                })
            
            return df_imputed
            
        except Exception as e:
            logger.error(f'Simple imputation failed: {str(e)}')
            return df
    
    def impute_knn(self, df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
        try:
            df_imputed = df.copy()
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                logger.warning('No numeric columns for KNN imputation')
                return df
            
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            encoded_data = df[numeric_cols].copy()
            
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    le = LabelEncoder()
                    mask = df[col].notna()
                    encoded_vals = np.zeros(len(df))
                    encoded_vals[mask] = le.fit_transform(df.loc[mask, col])
                    encoded_vals[~mask] = np.nan
                    encoded_data[col] = encoded_vals
                    self.encoders[col] = le
            
            knn_imputer = KNNImputer(n_neighbors=n_neighbors)
            imputed_array = knn_imputer.fit_transform(encoded_data)
            
            result_df = pd.DataFrame(imputed_array, columns=encoded_data.columns, index=df.index)
            
            for col in categorical_cols:
                if col in self.encoders:
                    result_df[col] = self.encoders[col].inverse_transform(result_df[col].astype(int))
            
            for col in df.columns:
                if col not in result_df.columns:
                    result_df[col] = df[col]
            
            self.imputation_log.append({
                'method': 'knn',
                'n_neighbors': n_neighbors,
                'columns': list(encoded_data.columns)
            })
            
            return result_df[df.columns]
            
        except Exception as e:
            logger.error(f'KNN imputation failed: {str(e)}')
            return df
    
    def impute_iterative(self, df: pd.DataFrame, max_iter: int = 10) -> pd.DataFrame:
        try:
            df_imputed = df.copy()
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                logger.warning('No numeric columns for iterative imputation')
                return df
            
            iterative_imputer = IterativeImputer(max_iter=max_iter, random_state=42)
            df_imputed[numeric_cols] = iterative_imputer.fit_transform(df[numeric_cols])
            
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df_imputed[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            
            self.imputation_log.append({
                'method': 'iterative',
                'max_iter': max_iter,
                'numeric_columns': list(numeric_cols),
                'categorical_columns': list(categorical_cols)
            })
            
            return df_imputed
            
        except Exception as e:
            logger.error(f'Iterative imputation failed: {str(e)}')
            return df
    
    def impute_smart(self, df: pd.DataFrame, recommendations: Dict[str, Dict]) -> pd.DataFrame:
        try:
            df_imputed = df.copy()
            
            for col, rec in recommendations.items():
                if col not in df.columns or df[col].isnull().sum() == 0:
                    continue
                
                strategy = rec.get('strategy', 'mean')
                
                if strategy == 'drop_column':
                    df_imputed = df_imputed.drop(columns=[col])
                    continue
                
                if strategy in ['mean', 'median', 'mode']:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        if strategy == 'mean':
                            df_imputed[col].fillna(df[col].mean(), inplace=True)
                        elif strategy == 'median':
                            df_imputed[col].fillna(df[col].median(), inplace=True)
                    else:
                        df_imputed[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
            
            self.imputation_log.append({
                'method': 'smart',
                'recommendations_applied': len(recommendations)
            })
            
            return df_imputed
            
        except Exception as e:
            logger.error(f'Smart imputation failed: {str(e)}')
            return df
    
    def get_imputation_summary(self) -> List[Dict[str, Any]]:
        return self.imputation_log