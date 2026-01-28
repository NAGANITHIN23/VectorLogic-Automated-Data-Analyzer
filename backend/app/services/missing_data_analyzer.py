import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class MissingDataAnalyzer:
    
    def __init__(self):
        self.missing_patterns = None
        self.missing_mechanism = None
        
    def analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            missing_percentage = (missing_cells / total_cells) * 100
            
            column_missing = {}
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    column_missing[col] = {
                        'count': int(missing_count),
                        'percentage': float((missing_count / len(df)) * 100)
                    }
            
            missing_rows = df.isnull().any(axis=1).sum()
            complete_rows = len(df) - missing_rows
            
            patterns = self._identify_missing_patterns(df)
            
            result = {
                'overall': {
                    'total_missing_cells': int(missing_cells),
                    'missing_percentage': float(missing_percentage),
                    'rows_with_missing': int(missing_rows),
                    'complete_rows': int(complete_rows)
                },
                'by_column': column_missing,
                'patterns': patterns
            }
            
            return result
            
        except Exception as e:
            logger.error(f'Missing pattern analysis failed: {str(e)}')
            return {}
    
    def _identify_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            missing_matrix = df.isnull().astype(int)
            
            pattern_counts = missing_matrix.groupby(list(missing_matrix.columns)).size()
            pattern_counts = pattern_counts.sort_values(ascending=False)
            
            top_patterns = []
            for idx, (pattern, count) in enumerate(pattern_counts.head(10).items()):
                if isinstance(pattern, tuple):
                    pattern_dict = {col: bool(val) for col, val in zip(df.columns, pattern)}
                else:
                    pattern_dict = {df.columns[0]: bool(pattern)}
                
                top_patterns.append({
                    'rank': idx + 1,
                    'count': int(count),
                    'percentage': float((count / len(df)) * 100),
                    'pattern': pattern_dict
                })
            
            return {
                'unique_patterns': len(pattern_counts),
                'top_10_patterns': top_patterns
            }
            
        except Exception as e:
            logger.error(f'Pattern identification failed: {str(e)}')
            return {}
    
    def test_missing_mechanism(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            results = {}
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in df.columns:
                if df[col].isnull().sum() == 0:
                    continue
                
                missing_indicator = df[col].isnull().astype(int)
                
                correlations = {}
                for other_col in numeric_cols:
                    if other_col != col and df[other_col].notna().sum() > 0:
                        corr, p_value = stats.pointbiserialr(missing_indicator, df[other_col].fillna(df[other_col].mean()))
                        
                        if abs(corr) > 0.1 and p_value < 0.05:
                            correlations[other_col] = {
                                'correlation': float(corr),
                                'p_value': float(p_value)
                            }
                
                if len(correlations) == 0:
                    mechanism = 'MCAR'
                    explanation = 'Missing Completely At Random - no correlation with other variables'
                elif len(correlations) <= 2:
                    mechanism = 'MAR'
                    explanation = 'Missing At Random - correlated with observed variables'
                else:
                    mechanism = 'MNAR'
                    explanation = 'Missing Not At Random - complex missing pattern'
                
                results[col] = {
                    'mechanism': mechanism,
                    'explanation': explanation,
                    'correlated_with': correlations
                }
            
            return results
            
        except Exception as e:
            logger.error(f'Missing mechanism test failed: {str(e)}')
            return {}
    
    def recommend_imputation_strategy(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        try:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            col_type = df[col].dtype
            
            if missing_pct > 70:
                return {
                    'strategy': 'drop_column',
                    'reason': 'More than 70% missing - too sparse for reliable imputation'
                }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                nunique = df[col].nunique()
                
                if nunique < 10:
                    return {
                        'strategy': 'mode',
                        'reason': 'Low cardinality numeric - use most frequent value'
                    }
                
                skewness = df[col].skew()
                
                if abs(skewness) < 0.5:
                    return {
                        'strategy': 'mean',
                        'reason': 'Normally distributed - mean imputation appropriate'
                    }
                else:
                    return {
                        'strategy': 'median',
                        'reason': 'Skewed distribution - median more robust'
                    }
            
            else:
                return {
                    'strategy': 'mode',
                    'reason': 'Categorical variable - use most frequent value'
                }
                
        except Exception as e:
            logger.error(f'Strategy recommendation failed: {str(e)}')
            return {'strategy': 'drop', 'reason': 'Error in analysis'}