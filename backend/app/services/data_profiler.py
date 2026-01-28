import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
import json
import logging

logger = logging.getLogger(__name__)

class DataProfiler:
    
    @staticmethod
    def get_basic_info(df: pd.DataFrame) -> Dict[str, Any]:
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            return {
                "shape": {"rows": len(df), "columns": len(df.columns)},
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "datetime_columns": datetime_cols,
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
                "duplicates": int(df.duplicated().sum())
            }
        except Exception as e:
            logger.error(f"Error in get_basic_info: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def get_missing_analysis(df: pd.DataFrame) -> Dict[str, Any]:
        try:
            missing_counts = df.isnull().sum()
            missing_pct = (missing_counts / len(df) * 100).round(2)
            
            missing_data = []
            for col in df.columns:
                if missing_counts[col] > 0:
                    missing_data.append({
                        "column": col,
                        "count": int(missing_counts[col]),
                        "percentage": float(missing_pct[col])
                    })
            
            missing_data.sort(key=lambda x: x['percentage'], reverse=True)
            
            return {
                "total_missing": int(missing_counts.sum()),
                "columns_with_missing": missing_data,
                "completely_empty_columns": [col for col in df.columns if missing_counts[col] == len(df)]
            }
        except Exception as e:
            logger.error(f"Error in get_missing_analysis: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def get_numeric_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                return {"message": "No numeric columns found"}
            
            stats_dict = {}
            for col in numeric_df.columns:
                col_data = numeric_df[col].dropna()
                if len(col_data) > 0:
                    stats_dict[col] = {
                        "count": int(len(col_data)),
                        "mean": float(col_data.mean()),
                        "median": float(col_data.median()),
                        "std": float(col_data.std()) if len(col_data) > 1 else 0,
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "q25": float(col_data.quantile(0.25)),
                        "q75": float(col_data.quantile(0.75))
                    }
            
            return stats_dict
        except Exception as e:
            logger.error(f"Error in get_numeric_statistics: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, method: str = "iqr") -> Dict[str, Any]:
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            outliers = {}
            
            for col in numeric_df.columns:
                col_data = numeric_df[col].dropna()
                
                if len(col_data) < 4:
                    continue
                
                if method == "iqr":
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_mask = (numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)
                else:
                    z_scores = np.abs(stats.zscore(col_data))
                    outlier_mask = pd.Series(False, index=numeric_df.index)
                    outlier_mask.loc[col_data.index] = z_scores > 3
                
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    outliers[col] = {
                        "count": int(outlier_count),
                        "percentage": round(outlier_count / len(df) * 100, 2)
                    }
            
            return outliers
        except Exception as e:
            logger.error(f"Error in detect_outliers: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def get_correlation_matrix(df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.shape[1] < 2:
                return {"message": "Insufficient numeric columns for correlation"}
            
            corr_matrix = numeric_df.corr()
            
            # Create interactive heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title="Correlation Matrix Heatmap",
                xaxis_title="Features",
                yaxis_title="Features",
                height=600,
                width=800
            )
            
            heatmap_json = fig.to_json()
            
            # Find high correlations
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > threshold:
                        high_correlations.append({
                            "column1": corr_matrix.columns[i],
                            "column2": corr_matrix.columns[j],
                            "correlation": round(float(corr_value), 3),
                            "strength": "Strong" if abs(corr_value) > 0.7 else "Moderate"
                        })
            
            high_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            return {
                "matrix": corr_matrix.round(3).to_dict(),
                "high_correlations": high_correlations,
                "heatmap": heatmap_json
            }
        except Exception as e:
            logger.error(f"Error in correlation_matrix: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def get_categorical_analysis(df: pd.DataFrame, top_n: int = 10) -> Dict[str, Any]:
        try:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_cols) == 0:
                return {"message": "No categorical columns found"}
            
            analysis = {}
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                unique_count = df[col].nunique()
                
                analysis[col] = {
                    "unique_count": int(unique_count),
                    "top_values": {str(k): int(v) for k, v in value_counts.head(top_n).items()},
                    "cardinality_ratio": round(unique_count / len(df), 4)
                }
            
            return analysis
        except Exception as e:
            logger.error(f"Error in categorical_analysis: {str(e)}")
            return {"error": str(e)}
    @staticmethod
    def test_normality(df: pd.DataFrame) -> Dict[str, Any]:
        from scipy import stats
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                return {'message': 'No numeric columns for normality testing'}
            
            results = {}
            
            for col in numeric_df.columns:
                col_data = numeric_df[col].dropna()
                
                if len(col_data) < 8:
                    continue
                
                statistic, p_value = stats.shapiro(col_data[:5000])
                
                results[col] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'is_normal': p_value > 0.05,
                    'interpretation': 'normal' if p_value > 0.05 else 'not normal'
                }
            
            return results
            
        except Exception as e:
            logger.error(f'Normality test failed: {str(e)}')
            return {'error': str(e)}
    
    @staticmethod
    def detect_data_types(df: pd.DataFrame) -> Dict[str, Any]:
        try:
            type_detection = {}
            
            for col in df.columns:
                col_data = df[col].dropna()
                
                if len(col_data) == 0:
                    type_detection[col] = 'empty'
                    continue
                
                if pd.api.types.is_numeric_dtype(col_data):
                    unique_ratio = col_data.nunique() / len(col_data)
                    
                    if unique_ratio > 0.95:
                        type_detection[col] = 'identifier'
                    elif col_data.nunique() < 10:
                        type_detection[col] = 'categorical_numeric'
                    else:
                        type_detection[col] = 'continuous_numeric'
                else:
                    unique_ratio = col_data.nunique() / len(col_data)
                    
                    if unique_ratio > 0.95:
                        type_detection[col] = 'text_identifier'
                    else:
                        type_detection[col] = 'categorical_text'
            
            return type_detection
            
        except Exception as e:
            logger.error(f'Type detection failed: {str(e)}')
            return {'error': str(e)}
    
