import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import norm, t
import multiprocessing as mp
from datetime import datetime, timedelta

@dataclass
class MarketScenario:
    name: str
    price_path: np.ndarray
    volatility: float
    regime: str
    event_timeline: List[Dict]
    risk_factors: Dict[str, float]

@dataclass
class StressTestResult:
    scenario_name: str
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    worst_drawdown: float
    recovery_time: Optional[int]
    stress_indicators: Dict[str, float]

class MarketSimulator:
    def __init__(self, base_price: float = 100.0, cpu_count: Optional[int] = None):
        self.base_price = base_price
        self.cpu_count = cpu_count or mp.cpu_count()
        self.risk_free_rate = 0.02
        self.scenarios = {}
        
    def generate_market_scenarios(self, n_days: int = 252,
                                n_scenarios: int = 1000,
                                volatility_range: Tuple[float, float] = (0.1, 0.4),
                                regime_probs: Optional[Dict[str, float]] = None) -> Dict[str, MarketScenario]:
        """
        Generate diverse market scenarios with regime shifts and events
        """
        if regime_probs is None:
            regime_probs = {
                'normal': 0.6,
                'high_volatility': 0.2,
                'trending': 0.2
            }
            
        scenarios = {}
        regimes = list(regime_probs.keys())
        regime_probabilities = list(regime_probs.values())
        
        for i in range(n_scenarios):
            # Generate base price path with GBM
            volatility = np.random.uniform(*volatility_range)
            dt = 1/252
            drift = self.risk_free_rate - 0.5 * volatility**2
            
            price_path = np.zeros(n_days)
            price_path[0] = self.base_price
            
            # Add regime shifts
            n_regimes = np.random.randint(2, 5)
            regime_changes = sorted(np.random.choice(n_days-1, n_regimes-1, replace=False))
            current_regime = np.random.choice(regimes, p=regime_probabilities)
            
            regime_timeline = []
            last_change = 0
            
            for change_point in regime_changes + [n_days]:
                # Adjust parameters based on regime
                if current_regime == 'high_volatility':
                    vol_mult = 2.0
                    drift_mult = 0.5
                elif current_regime == 'trending':
                    vol_mult = 0.7
                    drift_mult = 2.0
                else:  # normal
                    vol_mult = 1.0
                    drift_mult = 1.0
                    
                # Generate price path for current regime
                for t in range(last_change, change_point):
                    dW = np.random.normal(0, np.sqrt(dt))
                    price_path[t] = price_path[t-1] * np.exp(
                        drift * drift_mult * dt + volatility * vol_mult * dW
                    )
                    
                regime_timeline.append({
                    'start': last_change,
                    'end': change_point,
                    'regime': current_regime
                })
                
                last_change = change_point
                current_regime = np.random.choice(regimes, p=regime_probabilities)
                
            # Add market events
            n_events = np.random.poisson(2)
            events = []
            
            for _ in range(n_events):
                event_time = np.random.randint(0, n_days)
                event_type = np.random.choice(['shock', 'rally', 'correction'])
                magnitude = np.random.uniform(0.02, 0.08)
                
                if event_type == 'shock':
                    price_path[event_time:] *= (1 - magnitude)
                elif event_type == 'rally':
                    price_path[event_time:] *= (1 + magnitude)
                else:  # correction
                    correction_length = min(n_days - event_time,
                                         np.random.randint(5, 20))
                    correction = np.linspace(0, magnitude, correction_length)
                    price_path[event_time:event_time+correction_length] *= (1 - correction)
                    
                events.append({
                    'time': event_time,
                    'type': event_type,
                    'magnitude': magnitude
                })
                
            # Calculate risk factors
            returns = np.diff(price_path) / price_path[:-1]
            
            risk_factors = {
                'volatility': np.std(returns) * np.sqrt(252),
                'skewness': pd.Series(returns).skew(),
                'kurtosis': pd.Series(returns).kurtosis(),
                'var_95': np.percentile(returns, 5),
                'max_drawdown': self._calculate_max_drawdown(price_path)
            }
            
            scenario = MarketScenario(
                name=f'scenario_{i}',
                price_path=price_path,
                volatility=volatility,
                regime=regime_timeline,
                event_timeline=events,
                risk_factors=risk_factors
            )
            
            scenarios[scenario.name] = scenario
            
        self.scenarios = scenarios
        return scenarios
    
    def run_stress_test(self, strategy_func: callable,
                       scenario_filters: Optional[Dict] = None) -> Dict[str, StressTestResult]:
        """
        Run comprehensive stress tests on trading strategy
        """
        if scenario_filters:
            test_scenarios = {
                name: scenario
                for name, scenario in self.scenarios.items()
                if self._filter_scenario(scenario, scenario_filters)
            }
        else:
            test_scenarios = self.scenarios
            
        results = {}
        
        with mp.Pool(self.cpu_count) as pool:
            stress_results = pool.starmap(
                self._test_scenario,
                [(name, scenario, strategy_func)
                 for name, scenario in test_scenarios.items()]
            )
            
        for result in stress_results:
            results[result.scenario_name] = result
            
        return results
    
    def _test_scenario(self, name: str,
                      scenario: MarketScenario,
                      strategy_func: callable) -> StressTestResult:
        """
        Test strategy performance on a specific scenario
        """
        # Run strategy
        strategy_returns = strategy_func(scenario.price_path)
        
        # Calculate performance metrics
        cum_returns = np.cumprod(1 + strategy_returns)
        drawdowns = self._calculate_drawdown_series(cum_returns)
        
        performance_metrics = {
            'total_return': cum_returns[-1] - 1,
            'sharpe_ratio': self._calculate_sharpe_ratio(strategy_returns),
            'sortino_ratio': self._calculate_sortino_ratio(strategy_returns),
            'calmar_ratio': self._calculate_calmar_ratio(strategy_returns, drawdowns)
        }
        
        # Calculate risk metrics
        risk_metrics = {
            'volatility': np.std(strategy_returns) * np.sqrt(252),
            'var_95': np.percentile(strategy_returns, 5),
            'cvar_95': np.mean(strategy_returns[strategy_returns <= np.percentile(strategy_returns, 5)]),
            'max_drawdown': np.max(drawdowns)
        }
        
        # Calculate stress indicators
        stress_indicators = {
            'regime_stability': self._calculate_regime_stability(strategy_returns, scenario.regime),
            'event_resilience': self._calculate_event_resilience(strategy_returns, scenario.event_timeline),
            'tail_risk_exposure': self._calculate_tail_risk(strategy_returns)
        }
        
        # Calculate recovery metrics
        worst_drawdown = np.max(drawdowns)
        recovery_time = self._calculate_recovery_time(cum_returns, drawdowns)
        
        return StressTestResult(
            scenario_name=name,
            performance_metrics=performance_metrics,
            risk_metrics=risk_metrics,
            worst_drawdown=worst_drawdown,
            recovery_time=recovery_time,
            stress_indicators=stress_indicators
        )
    
    def _filter_scenario(self, scenario: MarketScenario,
                        filters: Dict) -> bool:
        """
        Filter scenarios based on criteria
        """
        for key, value in filters.items():
            if key == 'min_volatility' and scenario.risk_factors['volatility'] < value:
                return False
            elif key == 'max_volatility' and scenario.risk_factors['volatility'] > value:
                return False
            elif key == 'regime' and value not in [r['regime'] for r in scenario.regime]:
                return False
            elif key == 'min_drawdown' and scenario.risk_factors['max_drawdown'] < value:
                return False
                
        return True
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """
        Calculate maximum drawdown from price series
        """
        peaks = np.maximum.accumulate(prices)
        drawdowns = (peaks - prices) / peaks
        return np.max(drawdowns)
    
    def _calculate_drawdown_series(self, cum_returns: np.ndarray) -> np.ndarray:
        """
        Calculate drawdown series
        """
        peaks = np.maximum.accumulate(cum_returns)
        drawdowns = (peaks - cum_returns) / peaks
        return drawdowns
    
    def _calculate_recovery_time(self, cum_returns: np.ndarray,
                               drawdowns: np.ndarray) -> Optional[int]:
        """
        Calculate recovery time from worst drawdown
        """
        worst_dd_idx = np.argmax(drawdowns)
        if worst_dd_idx == len(drawdowns) - 1:
            return None
            
        peak_value = cum_returns[worst_dd_idx]
        for i in range(worst_dd_idx + 1, len(cum_returns)):
            if cum_returns[i] >= peak_value:
                return i - worst_dd_idx
                
        return None
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate annualized Sharpe ratio
        """
        excess_returns = returns - self.risk_free_rate / 252
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sortino ratio using downside deviation
        """
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
        return np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, returns: np.ndarray,
                              drawdowns: np.ndarray) -> float:
        """
        Calculate Calmar ratio
        """
        annual_return = np.mean(returns) * 252
        max_drawdown = np.max(drawdowns)
        return annual_return / max_drawdown if max_drawdown > 0 else float('inf')
    
    def _calculate_regime_stability(self, returns: np.ndarray,
                                  regime_timeline: List[Dict]) -> float:
        """
        Calculate strategy stability across different regimes
        """
        regime_metrics = {}
        for regime in regime_timeline:
            regime_returns = returns[regime['start']:regime['end']]
            regime_metrics[regime['regime']] = {
                'mean': np.mean(regime_returns),
                'std': np.std(regime_returns)
            }
            
        # Calculate consistency across regimes
        means = np.array([m['mean'] for m in regime_metrics.values()])
        stds = np.array([m['std'] for m in regime_metrics.values()])
        
        return 1 / (1 + np.std(means) / np.mean(stds))
    
    def _calculate_event_resilience(self, returns: np.ndarray,
                                  events: List[Dict]) -> float:
        """
        Calculate strategy resilience to market events
        """
        if not events:
            return 1.0
            
        event_impacts = []
        for event in events:
            pre_event = returns[max(0, event['time']-5):event['time']]
            post_event = returns[event['time']:min(len(returns), event['time']+5)]
            
            pre_vol = np.std(pre_event)
            post_vol = np.std(post_event)
            
            impact = abs(post_vol - pre_vol) / pre_vol if pre_vol > 0 else 1.0
            event_impacts.append(impact)
            
        return 1 / (1 + np.mean(event_impacts))
    
    def _calculate_tail_risk(self, returns: np.ndarray) -> float:
        """
        Calculate tail risk exposure using extreme value theory
        """
        sorted_returns = np.sort(returns)
        tail_threshold = np.percentile(returns, 5)
        tail_returns = sorted_returns[sorted_returns <= tail_threshold]
        
        if len(tail_returns) < 2:
            return 0.0
            
        # Fit Generalized Pareto Distribution
        try:
            _, loc, scale = t.fit(tail_returns)
            return scale / abs(loc) if loc != 0 else scale
        except:
            return np.std(tail_returns) / abs(np.mean(tail_returns))
