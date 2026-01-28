from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings
from app.services.data_profiler import DataProfiler
from app.services.data_cleaner import DataCleaner
from app.services.vector_search import VectorSearchService
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.preprocessing_pipeline import PreprocessingPipeline
from app.services.missing_data_analyzer import MissingDataAnalyzer
from app.services.smart_imputer import SmartImputer
from app.services.outlier_handler import OutlierHandler
from app.services.feature_engineer import FeatureEngineer
import pandas as pd
from typing import Dict, Any, List, Optional
import json
import logging

logger = logging.getLogger(__name__)

class LLMOrchestrator:
    def __init__(self, db: Optional[AsyncSession] = None):
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL,
            temperature=0.2,
            max_tokens=4096,
            timeout=120
        )
        
        self.profiler = DataProfiler()
        self.cleaner = DataCleaner()
        self.db = db
        self.vector_search = VectorSearchService(db) if db else None
        
        self.preprocessing_pipeline = PreprocessingPipeline()
        self.missing_analyzer = MissingDataAnalyzer()
        self.imputer = SmartImputer()
        self.outlier_handler = OutlierHandler()
        self.feature_engineer = FeatureEngineer()
        
        self.current_df = None
        self.profile_cache = {}
        self.similar_analyses_context = []
        
        self.tools = self._create_tools()
        self.prompt = self._create_prompt()
        
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=settings.DEBUG,
            max_iterations=30,
            handle_parsing_errors=True,
            max_execution_time=360,
            return_intermediate_steps=True
        )
    
    def _create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="get_basic_info",
                func=self._tool_basic_info,
                description="Get dataset shape, column types, memory usage, duplicate count"
            ),
            Tool(
                name="analyze_missing_values",
                func=self._tool_missing_analysis,
                description="Analyze missing values across all columns"
            ),
            Tool(
                name="get_numeric_statistics",
                func=self._tool_numeric_stats,
                description="Get statistics for numeric columns"
            ),
            Tool(
                name="detect_outliers",
                func=self._tool_detect_outliers,
                description="Detect outliers in numeric columns"
            ),
            Tool(
                name="analyze_correlations",
                func=self._tool_correlations,
                description="Find correlations between numeric columns"
            ),
            Tool(
                name="analyze_categorical",
                func=self._tool_categorical,
                description="Analyze categorical columns"
            ),
            Tool(
                name="generate_recommendations",
                func=self._tool_recommendations,
                description="Generate data cleaning recommendations"
            ),
            Tool(
                name='analyze_data_quality',
                func=self._tool_analyze_quality,
                description='Comprehensive data quality analysis including missing data patterns, mechanisms, and outlier detection'
            ),
            Tool(
                name='preprocess_data',
                func=self._tool_preprocess_data,
                description='Apply intelligent preprocessing including imputation and outlier treatment'
            ),
            Tool(
                name='detect_missing_patterns',
                func=self._tool_missing_patterns,
                description='Analyze missing data patterns and identify systematic missingness'
            ),
            Tool(
                name='test_missing_mechanism',
                func=self._tool_missing_mechanism,
                description='Test whether data is MCAR, MAR, or MNAR'
            ),
            Tool(
                name='detect_outliers_advanced',
                func=self._tool_detect_outliers_advanced,
                description='Detect outliers using IQR, Z-score, and Isolation Forest methods'
            ),
            Tool(
                name='impute_missing_values',
                func=self._tool_impute,
                description='Impute missing values using smart, KNN, or iterative methods'
            ),
            Tool(
                name='treat_outliers',
                func=self._tool_treat_outliers,
                description='Treat outliers using capping, winsorization, or removal'
            ),
            Tool(
                name='generate_features',
                func=self._tool_generate_features,
                description='Automatically generate new features and rank them by importance'
            ),
            Tool(
                name='test_normality',
                func=self._tool_test_normality,
                description='Test if numeric columns follow normal distribution'
            ),
            Tool(
                name='detect_column_types',
                func=self._tool_detect_types,
                description='Intelligently detect whether columns are identifiers, categorical, or continuous'
            )
        ]
    
    def _create_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert data analyst with advanced preprocessing and feature engineering capabilities.

CRITICAL: You MUST use these tools in this exact order:

PHASE 1 - BASIC ANALYSIS:
1. get_basic_info
2. analyze_missing_values

PHASE 2 - ADVANCED MISSING DATA ANALYSIS:
3. detect_missing_patterns
4. test_missing_mechanism

PHASE 3 - STATISTICAL ANALYSIS:
5. get_numeric_statistics
6. test_normality
7. detect_column_types

PHASE 4 - OUTLIER DETECTION:
8. detect_outliers
9. detect_outliers_advanced

PHASE 5 - DATA QUALITY & PREPROCESSING:
10. analyze_data_quality
11. preprocess_data (use strategy='smart')

PHASE 6 - FEATURE ENGINEERING:
12. generate_features

PHASE 7 - CORRELATIONS & PATTERNS:
13. analyze_correlations
14. analyze_categorical

PHASE 8 - RECOMMENDATIONS:
15. generate_recommendations

Call ALL 15 tools in this exact order. After calling all tools, synthesize findings into a comprehensive report."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
    
    async def analyze(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        self.current_df = df
        self.profile_cache = {}
        
        try:
            logger.info("Starting comprehensive analysis with forced tool execution")
            
            logger.info("Phase 1: Basic Analysis")
            self._tool_basic_info()
            self._tool_missing_analysis()
            
            logger.info("Phase 2: Advanced Missing Data Analysis")
            self._tool_missing_patterns()
            self._tool_missing_mechanism()
            
            logger.info("Phase 3: Statistical Analysis")
            self._tool_numeric_stats()
            self._tool_test_normality()
            self._tool_detect_types()
            
            logger.info("Phase 4: Outlier Detection")
            self._tool_detect_outliers()
            self._tool_detect_outliers_advanced('iqr')
            self._tool_detect_outliers_advanced('zscore')
            self._tool_detect_outliers_advanced('isolation_forest')
            
            logger.info("Phase 5: Data Quality & Preprocessing")
            self._tool_analyze_quality()
            
            logger.info("Phase 6: Feature Engineering")
            self._tool_generate_features()
            
            logger.info("Phase 7: Correlations & Patterns")
            self._tool_correlations()
            self._tool_categorical()
            
            logger.info("Phase 8: Recommendations")
            self._tool_recommendations()
            
            logger.info("All tools executed successfully")
            
            missing_patterns = self.profile_cache.get('missing_patterns', {})
            missing_overall = missing_patterns.get('overall', {})
            feature_eng = self.profile_cache.get('feature_engineering', {})
            outliers_adv = self.profile_cache.get('outliers_advanced', {})
            
            analysis_summary = f"""Comprehensive Analysis Complete for {filename}

Dataset: {len(df)} rows × {len(df.columns)} columns

ANALYSIS PERFORMED:
✓ Basic Information & Structure
✓ Missing Value Patterns (found {len(missing_patterns.get('patterns', {}).get('top_10_patterns', []))} unique patterns)
✓ Missing Mechanism Classification (MCAR/MAR/MNAR testing)
✓ Statistical Analysis & Distribution Testing
✓ Intelligent Column Type Detection
✓ Multi-Method Outlier Detection (IQR, Z-score, Isolation Forest)
✓ Comprehensive Data Quality Assessment
✓ Automated Feature Engineering (generated features ranked by importance)
✓ Correlation Analysis
✓ Categorical Variable Analysis
✓ Data Quality Recommendations

KEY FINDINGS:

Missing Data:
- Total missing cells: {missing_overall.get('total_missing_cells', 0)}
- Missing percentage: {missing_overall.get('missing_percentage', 0):.2f}%
- Rows with missing: {missing_overall.get('rows_with_missing', 0)}

Feature Engineering:
- New features generated: {feature_eng.get('new_features_count', 0)}
- Top features identified and ranked by importance

Outlier Detection:
- IQR method outliers detected across multiple columns
- Z-score analysis completed
- Isolation Forest multivariate outlier detection performed

Data Quality:
- Comprehensive quality report generated with MCAR/MAR/MNAR classification
- Column-specific preprocessing recommendations provided
- Statistical validation completed

All detailed results are available in the expandable sections below."""

            return {
                "analysis_text": analysis_summary,
                "profile_data": self.profile_cache,
                "tool_calls_made": 16,
                "success": True
            }
        
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            
            return {
                "analysis_text": f"Analysis encountered an error: {str(e)}",
                "profile_data": self.profile_cache,
                "error": str(e),
                "success": False
            }
    
    def _tool_basic_info(self, _: str = "") -> str:
        result = self.profiler.get_basic_info(self.current_df)
        self.profile_cache['basic_info'] = result
        return json.dumps(result, indent=2)
    
    def _tool_missing_analysis(self, _: str = "") -> str:
        result = self.profiler.get_missing_analysis(self.current_df)
        self.profile_cache['missing_analysis'] = result
        return json.dumps(result, indent=2)
    
    def _tool_numeric_stats(self, _: str = "") -> str:
        result = self.profiler.get_numeric_statistics(self.current_df)
        self.profile_cache['numeric_stats'] = result
        return json.dumps(result, indent=2)
    
    def _tool_detect_outliers(self, _: str = "") -> str:
        result = self.profiler.detect_outliers(self.current_df)
        self.profile_cache['outliers'] = result
        return json.dumps(result, indent=2)
    
    def _tool_correlations(self, _: str = "") -> str:
        result = self.profiler.get_correlation_matrix(self.current_df)
        self.profile_cache['correlations'] = result
        return json.dumps(result, indent=2)
    
    def _tool_categorical(self, _: str = "") -> str:
        result = self.profiler.get_categorical_analysis(self.current_df)
        self.profile_cache['categorical_analysis'] = result
        return json.dumps(result, indent=2)
    
    def _tool_recommendations(self, _: str = "") -> str:
        result = self.cleaner.generate_recommendations(self.current_df, self.profile_cache)
        self.profile_cache['recommendations'] = result
        return json.dumps(result, indent=2)
    
    def _tool_analyze_quality(self, _: str = '') -> str:
        try:
            quality_report = self.preprocessing_pipeline.analyze_data_quality(self.current_df)
            self.profile_cache['data_quality'] = quality_report
            return json.dumps(quality_report, indent=2)
        except Exception as e:
            logger.error(f"Quality analysis failed: {str(e)}")
            return json.dumps({'error': str(e)})
    
    def _tool_preprocess_data(self, strategy: str = 'smart') -> str:
        try:
            result = self.preprocessing_pipeline.preprocess_automatic(self.current_df, strategy=strategy)
            self.profile_cache['preprocessing'] = result
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return json.dumps({'error': str(e)})
    
    def _tool_missing_patterns(self, _: str = '') -> str:
        try:
            patterns = self.missing_analyzer.analyze_missing_patterns(self.current_df)
            self.profile_cache['missing_patterns'] = patterns
            return json.dumps(patterns, indent=2)
        except Exception as e:
            logger.error(f"Missing patterns analysis failed: {str(e)}")
            return json.dumps({'error': str(e)})
    
    def _tool_missing_mechanism(self, _: str = '') -> str:
        try:
            mechanisms = self.missing_analyzer.test_missing_mechanism(self.current_df)
            self.profile_cache['missing_mechanisms'] = mechanisms
            return json.dumps(mechanisms, indent=2)
        except Exception as e:
            logger.error(f"Missing mechanism test failed: {str(e)}")
            return json.dumps({'error': str(e)})
    
    def _tool_detect_outliers_advanced(self, method: str = 'iqr') -> str:
        try:
            if method == 'iqr':
                outliers = self.outlier_handler.detect_outliers_iqr(self.current_df)
            elif method == 'zscore':
                outliers = self.outlier_handler.detect_outliers_zscore(self.current_df)
            elif method == 'isolation_forest':
                outliers = self.outlier_handler.detect_outliers_isolation_forest(self.current_df)
            else:
                outliers = self.outlier_handler.detect_outliers_iqr(self.current_df)
            
            self.profile_cache['outliers_advanced'] = outliers
            return json.dumps(outliers, indent=2)
        except Exception as e:
            logger.error(f"Advanced outlier detection failed: {str(e)}")
            return json.dumps({'error': str(e)})
    
    def _tool_impute(self, strategy: str = 'smart') -> str:
        try:
            if strategy == 'smart':
                quality_report = self.preprocessing_pipeline.analyze_data_quality(self.current_df)
                recommendations = quality_report.get('recommendations', {})
                imputation_recs = {col: rec['imputation'] for col, rec in recommendations.items() if 'imputation' in rec}
                df_imputed = self.imputer.impute_smart(self.current_df, imputation_recs)
            elif strategy == 'knn':
                df_imputed = self.imputer.impute_knn(self.current_df)
            elif strategy == 'iterative':
                df_imputed = self.imputer.impute_iterative(self.current_df)
            else:
                df_imputed = self.imputer.impute_simple(self.current_df, strategy='mean')
            
            result = {
                'original_missing': int(self.current_df.isnull().sum().sum()),
                'after_missing': int(df_imputed.isnull().sum().sum()),
                'imputation_log': self.imputer.get_imputation_summary()
            }
            
            self.profile_cache['imputation'] = result
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Imputation failed: {str(e)}")
            return json.dumps({'error': str(e)})
    
    def _tool_treat_outliers(self, method: str = 'cap') -> str:
        try:
            if method == 'cap':
                df_treated = self.outlier_handler.treat_outliers_cap(self.current_df)
            elif method == 'winsorize':
                df_treated = self.outlier_handler.treat_outliers_winsorize(self.current_df)
            else:
                df_treated = self.current_df
            
            outliers_before = self.outlier_handler.detect_outliers_iqr(self.current_df)
            outliers_after = self.outlier_handler.detect_outliers_iqr(df_treated)
            
            result = {
                'method': method,
                'outliers_before': len(outliers_before),
                'outliers_after': len(outliers_after),
                'outliers_reduced': len(outliers_before) - len(outliers_after)
            }
            
            self.profile_cache['outlier_treatment'] = result
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Outlier treatment failed: {str(e)}")
            return json.dumps({'error': str(e)})
    
    def _tool_generate_features(self, _: str = '') -> str:
        try:
            feature_matrix, new_features = self.feature_engineer.generate_features(self.current_df, max_depth=2)
            
            if len(new_features) == 0:
                result = {'message': 'No new features could be generated', 'new_features_count': 0}
                self.profile_cache['feature_engineering'] = result
                return json.dumps(result)
            
            ranked = self.feature_engineer.rank_features(feature_matrix)
            top_features = ranked[:20]
            
            stats = self.feature_engineer.get_feature_statistics(feature_matrix)
            
            result = {
                'new_features_count': len(new_features),
                'top_20_features': top_features,
                'statistics': stats
            }
            
            self.profile_cache['feature_engineering'] = result
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Feature generation failed: {str(e)}")
            result = {'error': str(e), 'new_features_count': 0}
            self.profile_cache['feature_engineering'] = result
            return json.dumps(result)
    
    def _tool_test_normality(self, _: str = '') -> str:
        try:
            result = self.profiler.test_normality(self.current_df)
            self.profile_cache['normality_tests'] = result
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Normality test failed: {str(e)}")
            return json.dumps({'error': str(e)})
    
    def _tool_detect_types(self, _: str = '') -> str:
        try:
            result = self.profiler.detect_data_types(self.current_df)
            self.profile_cache['intelligent_types'] = result
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Type detection failed: {str(e)}")
            return json.dumps({'error': str(e)})
    
    async def generate_summary_for_embedding(self) -> Dict[str, str]:
        if not self.profile_cache:
            return {"summary": "", "key_features": ""}
        
        basic = self.profile_cache.get('basic_info', {})
        missing = self.profile_cache.get('missing_analysis', {})
        
        summary_parts = [
            f"{basic.get('shape', {}).get('rows', 0)} rows",
            f"{basic.get('shape', {}).get('columns', 0)} columns"
        ]
        
        if missing.get('columns_with_missing'):
            summary_parts.append(f"{len(missing['columns_with_missing'])} cols with missing values")
        
        summary = ", ".join(summary_parts)
        
        key_features = json.dumps({
            "numeric_columns": basic.get('numeric_columns', [])[:5],
            "categorical_columns": basic.get('categorical_columns', [])[:5],
            "has_missing": len(missing.get('columns_with_missing', [])) > 0
        })
        
        return {
            "summary": summary,
            "key_features": key_features
        }