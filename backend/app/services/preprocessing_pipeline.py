import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from app.services.missing_data_analyzer import MissingDataAnalyzer
from app.services.smart_imputer import SmartImputer
from app.services.outlier_handler import OutlierHandler

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    
    def __init__(self):
        self.missing_analyzer = MissingDataAnalyzer()
        self.imputer = SmartImputer()
        self.outlier_handler = OutlierHandler()
        self.pipeline_log = []
        
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            missing_patterns = self.missing_analyzer.analyze_missing_patterns(df)
            missing_mechanisms = self.missing_analyzer.test_missing_mechanism(df)
            
            outliers_iqr = self.outlier_handler.detect_outliers_iqr(df)
            outliers_zscore = self.outlier_handler.detect_outliers_zscore(df)
            outliers_isolation = self.outlier_handler.detect_outliers_isolation_forest(df)
            
            recommendations = {}
            for col in df.columns:
                col_rec = {}
                
                if col in missing_mechanisms:
                    imputation_rec = self.missing_analyzer.recommend_imputation_strategy(df, col)
                    col_rec['imputation'] = imputation_rec
                
                if col in outliers_iqr:
                    outlier_rec = self.outlier_handler.recommend_outlier_treatment(df, col)
                    col_rec['outlier_treatment'] = outlier_rec
                
                if col_rec:
                    recommendations[col] = col_rec
            
            quality_report = {
                'missing_data': {
                    'patterns': missing_patterns,
                    'mechanisms': missing_mechanisms
                },
                'outliers': {
                    'iqr_method': outliers_iqr,
                    'zscore_method': outliers_zscore,
                    'isolation_forest': outliers_isolation
                },
                'recommendations': recommendations
            }
            
            self.pipeline_log.append({
                'step': 'quality_analysis',
                'status': 'completed'
            })
            
            return quality_report
            
        except Exception as e:
            logger.error(f'Data quality analysis failed: {str(e)}')
            return {}
    
    def preprocess_automatic(self, df: pd.DataFrame, strategy: str = 'smart') -> Dict[str, Any]:
        try:
            original_shape = df.shape
            df_processed = df.copy()
            
            quality_report = self.analyze_data_quality(df)
            
            if strategy == 'smart':
                recommendations = quality_report.get('recommendations', {})
                imputation_recs = {col: rec['imputation'] for col, rec in recommendations.items() if 'imputation' in rec}
                df_processed = self.imputer.impute_smart(df_processed, imputation_recs)
            elif strategy == 'knn':
                df_processed = self.imputer.impute_knn(df_processed)
            elif strategy == 'iterative':
                df_processed = self.imputer.impute_iterative(df_processed)
            elif strategy == 'simple_mean':
                df_processed = self.imputer.impute_simple(df_processed, strategy='mean')
            elif strategy == 'simple_median':
                df_processed = self.imputer.impute_simple(df_processed, strategy='median')
            
            outliers_detected = quality_report.get('outliers', {}).get('iqr_method', {})
            if len(outliers_detected) > 0:
                df_processed = self.outlier_handler.treat_outliers_cap(df_processed, method='iqr')
            
            processed_shape = df_processed.shape
            
            validation = self.validate_preprocessing(df, df_processed)
            
            result = {
                'original_shape': original_shape,
                'processed_shape': processed_shape,
                'rows_changed': original_shape[0] - processed_shape[0],
                'columns_changed': original_shape[1] - processed_shape[1],
                'imputation_log': self.imputer.get_imputation_summary(),
                'validation': validation,
                'quality_report': quality_report
            }
            
            self.pipeline_log.append({
                'step': 'preprocessing',
                'strategy': strategy,
                'status': 'completed'
            })
            
            return result
            
        except Exception as e:
            logger.error(f'Automatic preprocessing failed: {str(e)}')
            return {}
    
    def validate_preprocessing(self, df_original: pd.DataFrame, df_processed: pd.DataFrame) -> Dict[str, Any]:
        try:
            validation = {}
            
            missing_before = df_original.isnull().sum().sum()
            missing_after = df_processed.isnull().sum().sum()
            
            validation['missing_values'] = {
                'before': int(missing_before),
                'after': int(missing_after),
                'reduction': int(missing_before - missing_after),
                'reduction_percentage': float(((missing_before - missing_after) / missing_before * 100) if missing_before > 0 else 0)
            }
            
            numeric_cols = df_original.select_dtypes(include=[np.number]).columns
            
            distribution_changes = {}
            for col in numeric_cols:
                if col in df_processed.columns:
                    mean_before = df_original[col].mean()
                    mean_after = df_processed[col].mean()
                    std_before = df_original[col].std()
                    std_after = df_processed[col].std()
                    
                    distribution_changes[col] = {
                        'mean_change': float(abs(mean_after - mean_before) / abs(mean_before) * 100) if mean_before != 0 else 0,
                        'std_change': float(abs(std_after - std_before) / abs(std_before) * 100) if std_before != 0 else 0
                    }
            
            validation['distribution_stability'] = distribution_changes
            
            validation['data_integrity'] = {
                'shape_preserved': df_original.shape == df_processed.shape,
                'columns_preserved': list(df_original.columns) == list(df_processed.columns),
                'dtypes_preserved': (df_original.dtypes == df_processed.dtypes).all()
            }
            
            return validation
            
        except Exception as e:
            logger.error(f'Preprocessing validation failed: {str(e)}')
            return {}
    
    def get_pipeline_summary(self) -> List[Dict[str, Any]]:
        return self.pipeline_log