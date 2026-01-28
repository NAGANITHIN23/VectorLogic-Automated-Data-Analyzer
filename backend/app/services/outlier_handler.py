import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import logging

logger = logging.getLogger(__name__)

class OutlierHandler:
    
    def __init__(self):
        self.outlier_info = {}
        
    def detect_outliers_iqr(self, df: pd.DataFrame, multiplier: float = 1.5) -> Dict[str, Any]:
        try:
            outliers = {}
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    outliers[col] = {
                        'count': int(outlier_count),
                        'percentage': float((outlier_count / len(df)) * 100),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound),
                        'outlier_indices': df[outlier_mask].index.tolist()[:100]
                    }
            
            return outliers
            
        except Exception as e:
            logger.error(f'IQR outlier detection failed: {str(e)}')
            return {}
    
    def detect_outliers_zscore(self, df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, Any]:
        try:
            outliers = {}
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                col_data = df[col].dropna()
                
                if len(col_data) == 0:
                    continue
                
                z_scores = np.abs(stats.zscore(col_data))
                outlier_mask = z_scores > threshold
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    outlier_indices = col_data[outlier_mask].index.tolist()
                    
                    outliers[col] = {
                        'count': int(outlier_count),
                        'percentage': float((outlier_count / len(col_data)) * 100),
                        'threshold': float(threshold),
                        'outlier_indices': outlier_indices[:100]
                    }
            
            return outliers
            
        except Exception as e:
            logger.error(f'Z-score outlier detection failed: {str(e)}')
            return {}
    
    def detect_outliers_isolation_forest(self, df: pd.DataFrame, contamination: float = 0.1) -> Dict[str, Any]:
        try:
            numeric_df = df.select_dtypes(include=[np.number]).dropna()
            
            if len(numeric_df) < 10 or len(numeric_df.columns) == 0:
                return {'message': 'Insufficient data for Isolation Forest'}
            
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            predictions = iso_forest.fit_predict(numeric_df)
            
            outlier_mask = predictions == -1
            outlier_count = outlier_mask.sum()
            
            result = {
                'total_outliers': int(outlier_count),
                'percentage': float((outlier_count / len(numeric_df)) * 100),
                'outlier_indices': numeric_df[outlier_mask].index.tolist()[:100],
                'method': 'isolation_forest'
            }
            
            return result
            
        except Exception as e:
            logger.error(f'Isolation Forest detection failed: {str(e)}')
            return {}
    
    def treat_outliers_winsorize(self, df: pd.DataFrame, limits: Tuple[float, float] = (0.05, 0.05)) -> pd.DataFrame:
        try:
            df_treated = df.copy()
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                lower_limit = df[col].quantile(limits[0])
                upper_limit = df[col].quantile(1 - limits[1])
                
                df_treated[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
            
            return df_treated
            
        except Exception as e:
            logger.error(f'Winsorization failed: {str(e)}')
            return df
    
    def treat_outliers_cap(self, df: pd.DataFrame, method: str = 'iqr', multiplier: float = 1.5) -> pd.DataFrame:
        try:
            df_treated = df.copy()
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - multiplier * IQR
                    upper_bound = Q3 + multiplier * IQR
                    
                    df_treated[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            return df_treated
            
        except Exception as e:
            logger.error(f'Outlier capping failed: {str(e)}')
            return df
    
    def treat_outliers_remove(self, df: pd.DataFrame, outlier_indices: List[int]) -> pd.DataFrame:
        try:
            df_treated = df.drop(index=outlier_indices)
            df_treated = df_treated.reset_index(drop=True)
            
            return df_treated
            
        except Exception as e:
            logger.error(f'Outlier removal failed: {str(e)}')
            return df
    
    def recommend_outlier_treatment(self, df: pd.DataFrame, col: str) -> Dict[str, str]:
        try:
            outliers_iqr = self.detect_outliers_iqr(df[[col]])
            
            if col not in outliers_iqr:
                return {
                    'recommendation': 'no_treatment',
                    'reason': 'No outliers detected'
                }
            
            outlier_pct = outliers_iqr[col]['percentage']
            
            if outlier_pct < 1:
                return {
                    'recommendation': 'remove',
                    'reason': 'Very few outliers, safe to remove'
                }
            elif outlier_pct < 5:
                return {
                    'recommendation': 'cap',
                    'reason': 'Moderate outliers, cap at IQR bounds'
                }
            else:
                return {
                    'recommendation': 'winsorize',
                    'reason': 'Many outliers, use winsorization to preserve distribution shape'
                }
                
        except Exception as e:
            logger.error(f'Treatment recommendation failed: {str(e)}')
            return {'recommendation': 'no_treatment', 'reason': 'Error in analysis'}