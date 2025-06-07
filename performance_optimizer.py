import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class OptimizationResult:
    optimized_params: Dict[str, Any]
    expected_performance: float
    performance_bounds: Tuple[float, float]
    risk_metrics: Dict[str, float]
    optimization_path: List[Dict]

class PerformanceOptimizer:
    def __init__(self, data_window_size: int = 252):
        self.data_window_size = data_window_size
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def analyze_strategy_performance(self, returns: pd.Series,
                                  feature_data: pd.DataFrame) -> Dict:
        """
        Analyze strategy performance and identify key drivers
        """
        # Prepare data
        X = feature_data.iloc[-self.data_window_size:]
        y = returns.iloc[-self.data_window_size:]
        X_scaled = self.scaler.fit_transform(X)
        
        # Train random forest to identify feature importance
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Calculate feature importance
        importance = pd.Series(
            model.feature_importances_,
            index=feature_data.columns
        ).sort_values(ascending=False)
        
        self.feature_importance = importance
        
        # Calculate performance attribution
        attribution = {}
        for feature, imp in importance.items():
            attribution[feature] = np.mean(X[feature] * imp * y)
            
        return {
            'feature_importance': importance.to_dict(),
            'performance_attribution': attribution,
            'model_r2': model.score(X_scaled, y)
        }
    
    def optimize_parameters(self, objective_func: callable,
                          param_bounds: Dict[str, Tuple[float, float]],
                          constraints: Optional[List[Dict]] = None,
                          initial_guess: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """
        Optimize strategy parameters using advanced optimization techniques
        """
        # Prepare optimization inputs
        param_names = list(param_bounds.keys())
        bounds = [param_bounds[p] for p in param_names]
        
        if initial_guess is None:
            initial_guess = {p: np.mean(b) for p, b in param_bounds.items()}
            
        x0 = [initial_guess[p] for p in param_names]
        
        # Track optimization path
        optimization_path = []
        
        def objective_wrapper(x):
            params = dict(zip(param_names, x))
            result = objective_func(params)
            optimization_path.append({
                'params': params.copy(),
                'performance': result
            })
            return -result  # Minimize negative performance (maximize performance)
        
        # Convert constraints to scipy format
        if constraints:
            scipy_constraints = []
            for constraint in constraints:
                def constraint_func(x, c=constraint):
                    params = dict(zip(param_names, x))
                    return c['func'](params)
                scipy_constraints.append({
                    'type': constraint['type'],
                    'fun': constraint_func
                })
        else:
            scipy_constraints = ()
            
        # Run optimization
        result = minimize(
            objective_wrapper,
            x0,
            bounds=bounds,
            constraints=scipy_constraints,
            method='SLSQP'
        )
        
        optimized_params = dict(zip(param_names, result.x))
        
        # Calculate performance bounds using bootstrap
        performance_samples = self._bootstrap_performance(
            objective_func, optimized_params, n_samples=1000
        )
        
        performance_bounds = (
            np.percentile(performance_samples, 2.5),
            np.percentile(performance_samples, 97.5)
        )
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(performance_samples)
        
        return OptimizationResult(
            optimized_params=optimized_params,
            expected_performance=-result.fun,
            performance_bounds=performance_bounds,
            risk_metrics=risk_metrics,
            optimization_path=optimization_path
        )
    
    def _bootstrap_performance(self, objective_func: callable,
                             params: Dict[str, float],
                             n_samples: int = 1000) -> np.ndarray:
        """
        Bootstrap performance to estimate confidence intervals
        """
        performances = []
        for _ in range(n_samples):
            # Add random noise to parameters
            noisy_params = {
                k: v * (1 + np.random.normal(0, 0.1))
                for k, v in params.items()
            }
            performances.append(objective_func(noisy_params))
            
        return np.array(performances)
    
    def _calculate_risk_metrics(self, performance_samples: np.ndarray) -> Dict[str, float]:
        """
        Calculate risk metrics from bootstrap samples
        """
        return {
            'mean_performance': np.mean(performance_samples),
            'performance_std': np.std(performance_samples),
            'var_95': np.percentile(performance_samples, 5),
            'cvar_95': np.mean(performance_samples[performance_samples <= np.percentile(performance_samples, 5)]),
            'downside_deviation': np.std(performance_samples[performance_samples < 0]),
            'skewness': float(pd.Series(performance_samples).skew()),
            'kurtosis': float(pd.Series(performance_samples).kurtosis())
        }
    
    def analyze_parameter_sensitivity(self, objective_func: callable,
                                   base_params: Dict[str, float],
                                   param_ranges: Dict[str, np.ndarray]) -> Dict[str, pd.DataFrame]:
        """
        Analyze strategy sensitivity to parameter changes
        """
        sensitivity_results = {}
        
        for param_name, param_range in param_ranges.items():
            performances = []
            for value in param_range:
                test_params = base_params.copy()
                test_params[param_name] = value
                performance = objective_func(test_params)
                performances.append(performance)
                
            sensitivity = pd.DataFrame({
                'parameter_value': param_range,
                'performance': performances
            })
            
            # Calculate sensitivity metrics
            sensitivity['performance_change'] = sensitivity['performance'].pct_change()
            sensitivity['elasticity'] = (sensitivity['performance_change'] /
                                      sensitivity['parameter_value'].pct_change())
            
            sensitivity_results[param_name] = sensitivity
            
        return sensitivity_results
    
    def generate_optimization_report(self, opt_result: OptimizationResult,
                                  sensitivity_analysis: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive optimization report
        """
        report = {
            'optimized_parameters': {
                param: f"{value:.4f}"
                for param, value in opt_result.optimized_params.items()
            },
            'performance_metrics': {
                'expected_performance': f"{opt_result.expected_performance:.4f}",
                'performance_bounds': (
                    f"{opt_result.performance_bounds[0]:.4f}",
                    f"{opt_result.performance_bounds[1]:.4f}"
                )
            },
            'risk_metrics': {
                metric: f"{value:.4f}"
                for metric, value in opt_result.risk_metrics.items()
            }
        }
        
        if sensitivity_analysis:
            sensitivity_metrics = {}
            for param, analysis in sensitivity_analysis.items():
                sensitivity_metrics[param] = {
                    'mean_elasticity': f"{analysis['elasticity'].mean():.4f}",
                    'max_impact': f"{analysis['performance_change'].abs().max():.4f}",
                    'optimal_range': (
                        f"{analysis.loc[analysis['performance'].idxmax(), 'parameter_value']:.4f}",
                        f"{analysis.loc[analysis['performance'].idxmin(), 'parameter_value']:.4f}"
                    )
                }
            report['sensitivity_analysis'] = sensitivity_metrics
            
        # Add optimization convergence info
        convergence_path = pd.DataFrame(opt_result.optimization_path)
        report['optimization_path'] = {
            'iterations': len(convergence_path),
            'performance_improvement': f"{(convergence_path['performance'].max() - convergence_path['performance'].iloc[0]) / abs(convergence_path['performance'].iloc[0]) * 100:.2f}%",
            'convergence_achieved': convergence_path['performance'].diff().abs().iloc[-5:].mean() < 1e-6
        }
        
        return report
