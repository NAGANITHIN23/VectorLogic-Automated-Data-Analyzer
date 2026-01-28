import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    
    @staticmethod
    def generate_recommendations(df: pd.DataFrame, profile_data: Dict) -> List[Dict[str, str]]:
        recommendations = []
        
        try:
            missing_analysis = profile_data.get('missing_analysis', {})
            for col_info in missing_analysis.get('columns_with_missing', []):
                if col_info['percentage'] > 50:
                    recommendations.append({
                        "column": col_info['column'],
                        "issue": f"{col_info['percentage']}% missing values",
                        "action": "Consider dropping this column",
                        "severity": "high",
                        "priority": 1
                    })
                elif col_info['percentage'] > 20:
                    recommendations.append({
                        "column": col_info['column'],
                        "issue": f"{col_info['percentage']}% missing values",
                        "action": "Use advanced imputation",
                        "severity": "medium",
                        "priority": 2
                    })
            
            basic_info = profile_data.get('basic_info', {})
            if basic_info.get('duplicates', 0) > 0:
                dup_count = basic_info['duplicates']
                recommendations.append({
                    "column": "all",
                    "issue": f"{dup_count} duplicate rows",
                    "action": "Remove duplicates",
                    "severity": "medium",
                    "priority": 2
                })
            
            outlier_data = profile_data.get('outliers', {})
            for col, outlier_info in outlier_data.items():
                if outlier_info['percentage'] > 10:
                    recommendations.append({
                        "column": col,
                        "issue": f"{outlier_info['percentage']}% outliers",
                        "action": "Investigate and handle outliers",
                        "severity": "medium",
                        "priority": 2
                    })
            
            recommendations.sort(key=lambda x: x['priority'])
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
        
        return recommendations
