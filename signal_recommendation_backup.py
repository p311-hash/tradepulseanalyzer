"""
Enhanced signal recommendation module for binary options trading.
This module provides advanced signal recommendations with fallback mechanisms.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import json
import os
import sys

# Add TradePulseAnalyzer to path if needed
if os.path.exists('TradePulseAnalyzer'):
    sys.path.append('TradePulseAnalyzer')

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedSignalRecommender:
    """
    Enhanced signal recommender with fallback mechanisms for reliable signal generation.
    Provides robust recommendations even when some data sources are unavailable.
    """
    
    def __init__(self, 
                 pairs: List[str] = None, 
                 timeframes: List[str] = None,
                 min_confidence: float = 65.0,
                 use_correlation: bool = True,
                 use_sentiment: bool = True):
        """
        Initialize the enhanced signal recommender.
        
        Args:
            pairs: List of trading pairs to analyze
            timeframes: List of timeframes to analyze
            min_confidence: Minimum confidence threshold for recommendations
            use_correlation: Whether to use correlation analysis
            use_sentiment: Whether to use sentiment analysis
        """
        self.pairs = pairs or ['EURUSD_otc', 'GBPUSD_otc', 'USDJPY_otc', 'AUDUSD_otc']
        self.timeframes = timeframes or ['1m', '5m', '15m']
        self.min_confidence = min_confidence
        self.use_correlation = use_correlation
        self.use_sentiment = use_sentiment
        
        # Initialize components with lazy loading
        self._technical_analyzer = None
        self._sentiment_analyzer = None
        self._correlation_analyzer = None
        
        # Cache for analysis results
        self.analysis_cache = {}
        self.cache_duration = timedelta(minutes=5)
        
        logger.info(f"Initialized EnhancedSignalRecommender with {len(self.pairs)} pairs and {len(self.timeframes)} timeframes")
    
    def get_recommendations(self, top_n: int = 3) -> List[Dict]:
        """
        Get top signal recommendations across all pairs and timeframes.
        
        Args:
            top_n: Number of top recommendations to return
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            logger.info(f"Generating recommendations for {len(self.pairs)} pairs")
            
            all_opportunities = []
            
            # Analyze each pair
            for pair in self.pairs:
                try:
                    # Get technical analysis with fallback
                    technical_data = self._get_technical_analysis_with_fallback(pair)
                    
                    # Skip if no technical data available
                    if not technical_data:
                        logger.warning(f"No technical data available for {pair}, skipping")
                        continue
                    
                    # Get sentiment data with fallback
                    sentiment_data = self._get_sentiment_data_with_fallback(pair) if self.use_sentiment else None
                    
                    # Get correlation data with fallback
                    correlation_data = self._get_correlation_matrix_with_fallback(pair) if self.use_correlation else None
                    
                    # Calculate enhanced score
                    enhanced_score = self._calculate_enhanced_score(
                        technical_data=technical_data,
                        sentiment_data=sentiment_data,
                        correlation_data=correlation_data,
                        pair=pair
                    )
                    
                    # Skip if score is below threshold
                    if abs(enhanced_score['score']) < 20 or enhanced_score['confidence'] < self.min_confidence:
                        logger.info(f"Score for {pair} below threshold, skipping")
                        continue
                    
                    # Add to opportunities
                    all_opportunities.append({
                        'pair': pair,
                        'direction': 'BUY' if enhanced_score['score'] > 0 else 'SELL',
                        'score': enhanced_score['score'],
                        'confidence': enhanced_score['confidence'],
                        'risk_level': self._calculate_risk_level(
                            technical_data, sentiment_data, correlation_data
                        ),
                        'explanation': self._generate_enhanced_explanation(
                            technical_data, sentiment_data, correlation_data, enhanced_score
                        ),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    logger.error(f"Error analyzing {pair}: {str(e)}")
                    continue
            
            # Select best opportunities
            best_opportunities = self._select_best_enhanced_opportunity(all_opportunities, top_n)
            
            logger.info(f"Generated {len(best_opportunities)} recommendations")
            return best_opportunities
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def _get_technical_analysis_with_fallback(self, pair: str) -> Optional[Dict]:
        """
        Get technical analysis data with fallback mechanisms.
        
        Args:
            pair: Trading pair symbol
            
        Returns:
            Dictionary with technical analysis data or None if unavailable
        """
        try:
            # Check cache first
            cache_key = f"tech_{pair}"
            if cache_key in self.analysis_cache:
                cached_data = self.analysis_cache[cache_key]
                if datetime.now() - cached_data['timestamp'] < self.cache_duration:
                    logger.debug(f"Using cached technical analysis for {pair}")
                    return cached_data['data']
            
            # Try to import required modules
            try:
                from technical_analysis import TechnicalAnalyzer
                from utils import fetch_market_data
                
                # Fetch market data for each timeframe
                timeframe_data = {}
                for tf in self.timeframes:
                    data = fetch_market_data(pair=pair, timeframe=tf)
                    if data is not None and not data.empty:
                        timeframe_data[tf] = data
                
                # Skip if no data available
                if not timeframe_data:
                    logger.warning(f"No market data available for {pair}")
                    return None
                
                # Analyze each timeframe
                timeframe_analysis = {}
                for tf, data in timeframe_data.items():
                    analyzer = TechnicalAnalyzer(data)
                    analysis = analyzer.analyze()
                    timeframe_analysis[tf] = analysis
                
                # Cache the result
                result = {
                    'pair': pair,
                    'timeframes': timeframe_analysis,
                    'timestamp': datetime.now()
                }
                
                self.analysis_cache[cache_key] = {
                    'data': result,
                    'timestamp': datetime.now()
                }
                
                return result
                
            except ImportError:
                logger.warning("TechnicalAnalyzer not available, using fallback")
                
                # Fallback to AdaptiveTechnicalAnalyzer if available
                try:
                    from technical_analysis import AdaptiveTechnicalAnalyzer
                    from utils import fetch_market_data
                    
                    # Fetch market data for primary timeframe
                    data = fetch_market_data(pair=pair, timeframe=self.timeframes[0])
                    if data is None or data.empty:
                        logger.warning(f"No market data available for {pair}")
                        return None
                    
                    # Analyze with adaptive analyzer
                    analyzer = AdaptiveTechnicalAnalyzer(use_ml=False)
                    analysis = analyzer.analyze(data)
                    
                    # Format result
                    result = {
                        'pair': pair,
                        'timeframes': {self.timeframes[0]: analysis},
                        'timestamp': datetime.now()
                    }
                    
                    # Cache the result
                    self.analysis_cache[cache_key] = {
                        'data': result,
                        'timestamp': datetime.now()
                    }
                    
                    return result
                    
                except ImportError:
                    logger.warning("AdaptiveTechnicalAnalyzer not available, using basic fallback")
                    
                    # Basic fallback with minimal functionality
                    try:
                        import pandas as pd
                        import numpy as np
                        from utils import fetch_market_data
                        
                        # Fetch market data
                        data = fetch_market_data(pair=pair, timeframe=self.timeframes[0])
                        if data is None or data.empty:
                            logger.warning(f"No market data available for {pair}")
                            return None
                        
                        # Calculate basic indicators
                        data['ma_fast'] = data['close'].rolling(window=10).mean()
                        data['ma_slow'] = data['close'].rolling(window=30).mean()
                        
                        # Generate basic signal
                        latest = data.iloc[-1]
                        signal = 'BUY' if latest['ma_fast'] > latest['ma_slow'] else 'SELL'
                        
                        # Format result
                        result = {
                            'pair': pair,
                            'timeframes': {
                                self.timeframes[0]: {
                                    'signal': {
                                        'direction': signal,
                                        'confidence': 60
                                    }
                                }
                            },
                            'timestamp': datetime.now()
                        }
                        
                        # Cache the result
                        self.analysis_cache[cache_key] = {
                            'data': result,
                            'timestamp': datetime.now()
                        }
                        
                        return result
                        
                    except Exception as e:
                        logger.error(f"Error in basic technical analysis fallback: {str(e)}")
                        return None
        
        except Exception as e:
            logger.error(f"Error getting technical analysis: {str(e)}")
            return None
    
    def _get_sentiment_data_with_fallback(self, pair: str) -> Optional[Dict]:
        """
        Get sentiment analysis data with fallback mechanisms.
        
        Args:
            pair: Trading pair symbol
            
        Returns:
            Dictionary with sentiment data or None if unavailable
        """
        try:
            # Check cache first
            cache_key = f"sentiment_{pair}"
            if cache_key in self.analysis_cache:
                cached_data = self.analysis_cache[cache_key]
                if datetime.now() - cached_data['timestamp'] < self.cache_duration:
                    logger.debug(f"Using cached sentiment data for {pair}")
                    return cached_data['data']
            
            # Try to import sentiment analyzer
            try:
                from sentiment_analysis import SentimentAnalyzer
                from utils import fetch_market_data
                
                # Fetch market data for sentiment analysis
                data = fetch_market_data(pair=pair, timeframe=self.timeframes[0])
                if data is None or data.empty:
                    logger.warning(f"No market data available for sentiment analysis of {pair}")
                    return None
                
                # Analyze sentiment
                analyzer = SentimentAnalyzer()
                sentiment = analyzer.analyze_market_sentiment(pair, data)
                
                # Cache the result
                self.analysis_cache[cache_key] = {
                    'data': sentiment,
                    'timestamp': datetime.now()
                }
                
                return sentiment
                
            except ImportError:
                logger.warning("SentimentAnalyzer not available, using fallback")
                
                # Try enhanced sentiment analyzer
                try:
                    from enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer
                    
                    # Analyze sentiment
                    analyzer = EnhancedSentimentAnalyzer()
                    sentiment = analyzer.analyze(pair)
                    
                    # Cache the result
                    self.analysis_cache[cache_key] = {
                        'data': sentiment,
                        'timestamp': datetime.now()
                    }
                    
                    return sentiment
                    
                except ImportError:
                    logger.warning("EnhancedSentimentAnalyzer not available, using basic fallback")
                    
                    # Basic fallback with minimal functionality
                    try:
                        # Generate basic sentiment data
                        sentiment = {
                            'overall_sentiment': 'NEUTRAL',
                            'combined_score': 0,
                            'confidence': 50,
                            'technical_sentiment': {
                                'sentiment': 'NEUTRAL',
                                'score': 0,
                                'confidence': 50
                            },
                            'news_sentiment': {
                                'sentiment': 'NEUTRAL',
                                'score': 0,
                                'confidence': 0
                            },
                            'social_sentiment': {
                                'sentiment': 'NEUTRAL',
                                'score': 0,
                                'confidence': 0
                            }
                        }
                        
                        # Cache the result
                        self.analysis_cache[cache_key] = {
                            'data': sentiment,
                            'timestamp': datetime.now()
                        }
                        
                        return sentiment
                        
                    except Exception as e:
                        logger.error(f"Error in basic sentiment fallback: {str(e)}")
                        return None
        
        except Exception as e:
            logger.error(f"Error getting sentiment data: {str(e)}")
            return None
    
    def _get_correlation_matrix_with_fallback(self, pair: str) -> Optional[Dict]:
        """
        Get correlation matrix with fallback mechanisms.
        
        Args:
            pair: Trading pair symbol
            
        Returns:
            Dictionary with correlation data or None if unavailable
        """
        try:
            # Check cache first
            cache_key = f"correlation_{pair}"
            if cache_key in self.analysis_cache:
                cached_data = self.analysis_cache[cache_key]
                if datetime.now() - cached_data['timestamp'] < self.cache_duration:
                    logger.debug(f"Using cached correlation data for {pair}")
                    return cached_data['data']
            
            # Try to import correlation analyzer
            try:
                from signal_correlation import SignalCorrelationAnalyzer
                
                # Get recent signals
                recent_signals = self._get_recent_signals()
                
                # Analyze correlations
                analyzer = SignalCorrelationAnalyzer()
                
                # Determine signal direction from technical analysis
                tech_data = self._get_technical_analysis_with_fallback(pair)
                if tech_data and tech_data['timeframes']:
                    primary_tf = list(tech_data['timeframes'].keys())[0]
                    signal_direction = tech_data['timeframes'][primary_tf]['signal']['direction']
                else:
                    signal_direction = 'NEUTRAL'
                
                # Analyze correlated signals
                correlation_data = analyzer.analyze_correlated_signals(
                    pair=pair,
                    signal=signal_direction,
                    recent_signals=recent_signals
                )
                
                # Cache the result
                self.analysis_cache[cache_key] = {
                    'data': correlation_data,
                    'timestamp': datetime.now()
                }
                
                return correlation_data
                
            except ImportError:
                logger.warning("SignalCorrelationAnalyzer not available, using fallback")
                
                # Use default correlations
                try:
                    # Default correlations
                    DEFAULT_CORRELATIONS = {
                        'EURUSD_otc': {
                            'GBPUSD_otc': 0.85,
                            'USDCHF_otc': -0.90,
                            'USDJPY_otc': -0.60,
                            'AUDUSD_otc': 0.65
                        },
                        'GBPUSD_otc': {
                            'EURUSD_otc': 0.85,
                            'USDCHF_otc': -0.75,
                            'USDJPY_otc': -0.55,
                            'AUDUSD_otc': 0.60
                        },
                        'USDJPY_otc': {
                            'EURUSD_otc': -0.60,
                            'GBPUSD_otc': -0.55,
                            'EURJPY_otc': 0.70,
                            'GBPJPY_otc': 0.75
                        },
                        'AUDUSD_otc': {
                            'EURUSD_otc': 0.65,
                            'GBPUSD_otc': 0.60,
                            'NZDUSD_otc': 0.80,
                            'USDCAD_otc': -0.65
                        }
                    }
                    
                    # Get correlations for the pair
                    correlations = DEFAULT_CORRELATIONS.get(pair, {})
                    
                    # Format result
                    result = {
                        'correlated_confirmation': False,
                        'confirmation_level': 0,
                        'confirming_pairs': [],
                        'contradicting_pairs': [],
                        'correlation_boost': 0
                    }
                    
                    # Cache the result
                    self.analysis_cache[cache_key] = {
                        'data': result,
                        'timestamp': datetime.now()
                    }
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error in correlation fallback: {str(e)}")
                    return None
        
        except Exception as e:
            logger.error(f"Error getting correlation data: {str(e)}")
            return None
    
    def _get_recent_signals(self) -> Dict[str, Dict]:
        """
        Get recent signals for all pairs.
        
        Returns:
            Dictionary mapping pairs to their recent signals
        """
        try:
            # Try to load signal history
            signal_history_path = 'signal_history.json'
            if os.path.exists(signal_history_path):
                with open(signal_history_path, 'r') as f:
                    history = json.load(f)
                
                # Extract recent signals
                recent_signals = {}
                if 'signals' in history:
                    for signal in history['signals']:
                        pair = signal.get('pair')
                        if pair and pair not in recent_signals:
                            recent_signals[pair] = {
                                'signal': signal.get('direction', 'NEUTRAL'),
                                'confidence': signal.get('confidence', 0),
                                'timestamp': signal.get('timestamp', '')
                            }
                
                return recent_signals
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting recent signals: {str(e)}")
            return {}
    
    def _calculate_enhanced_score(self, 
                                technical_data: Dict, 
                                sentiment_data: Optional[Dict] = None,
                                correlation_data: Optional[Dict] = None,
                                pair: str = None) -> Dict:
        """
        Calculate enhanced score combining technical, sentiment, and correlation data.
        
        Args:
            technical_data: Technical analysis data
            sentiment_data: Sentiment analysis data
            correlation_data: Correlation analysis data
            pair: Trading pair symbol
            
        Returns:
            Dictionary with enhanced score and confidence
        """
        try:
            # Base weights
            weights = {
                'technical': 0.7,
                'sentiment': 0.2,
                'correlation': 0.1
            }
            
            # Initialize scores
            technical_score = 0
            technical_confidence = 0
            sentiment_score = 0
            sentiment_confidence = 0
            correlation_boost = 0
            
            # Process technical data
            if technical_data and 'timeframes' in technical_data:
                # Calculate weighted score across timeframes
                tf_scores = []
                tf_confidences = []
                
                for tf, analysis in technical_data['timeframes'].items():
                    if 'signal' in analysis:
                        signal = analysis['signal']
                        direction = signal.get('direction', 'NEUTRAL')
                        confidence = signal.get('confidence', 0)
                        
                        # Convert direction to score
                        if direction == 'BUY':
                            score = 100
                        elif direction == 'SELL':
                            score = -100
                        else:
                            score = 0
                        
                        # Weight by timeframe
                        weight = self._get_timeframe_weight(tf)
                        tf_scores.append(score * weight)
                        tf_confidences.append(confidence * weight)
                
                # Calculate weighted average
                if tf_scores:
                    total_weight = sum(self._get_timeframe_weight(tf) for tf in technical_data['timeframes'])
                    technical_score = sum(tf_scores) / total_weight
                    technical_confidence = sum(tf_confidences) / total_weight
            
            # Process sentiment data
            if sentiment_data:
                sentiment_score = sentiment_data.get('combined_score', 0)
                sentiment_confidence = sentiment_data.get('confidence', 0)
            
            # Process correlation data
            if correlation_data:
                correlation_boost = correlation_data.get('correlation_boost', 0)
            
            # Calculate final score
            final_score = (
                technical_score * weights['technical'] +
                sentiment_score * weights['sentiment']
            )
            
            # Apply correlation boost
            if correlation_boost != 0:
                # Apply boost with diminishing returns
                boost_factor = 1 + (correlation_boost / 100)
                final_score *= boost_factor
            
            # Calculate final confidence
            final_confidence = (
                technical_confidence * weights['technical'] +
                sentiment_confidence * weights['sentiment']
            )
            
            # Ensure confidence is within bounds
            final_confidence = max(0, min(100, final_confidence))
            
            return {
                'score': final_score,
                'confidence': final_confidence,
                'technical_score': technical_score,
                'technical_confidence': technical_confidence,
                'sentiment_score': sentiment_score,
                'sentiment_confidence': sentiment_confidence,
                'correlation_boost': correlation_boost
            }
            
        except Exception as e:
            logger.error(f"Error calculating enhanced score: {str(e)}")
            return {
                'score': 0,
                'confidence': 0,
                'technical_score': 0,
                'technical_confidence': 0,
                'sentiment_score': 0,
                'sentiment_confidence': 0,
                'correlation_boost': 0
            }
    
    def _select_best_enhanced_opportunity(self, opportunities: List[Dict], top_n: int = 3) -> List[Dict]:
        """
        Select the best trading opportunities from the list.
        
        Args:
            opportunities: List of trading opportunities
            top_n: Number of top opportunities to select
            
        Returns:
            List of selected opportunities
        """
        try:
            if not opportunities:
                return []
            
            # Sort by confidence and score
            sorted_opportunities = sorted(
                opportunities,
                key=lambda x: (x['confidence'], abs(x['score'])),
                reverse=True
            )
            
            # Select top N
            return sorted_opportunities[:top_n]
            
        except Exception as e:
            logger.error(f"Error selecting best opportunities: {str(e)}")
            return opportunities[:min(top_n, len(opportunities))]
    
    def _calculate_risk_level(self, 
                            technical_data: Dict, 
                            sentiment_data: Optional[Dict] = None,
                            correlation_data: Optional[Dict] = None) -> str:
        """
        Calculate risk level for a trading opportunity.
        
        Args:
            technical_data: Technical analysis data
            sentiment_data: Sentiment analysis data
            correlation_data: Correlation analysis data
            
        Returns:
            Risk level string ('LOW', 'MEDIUM', 'HIGH')
        """
        try:
            risk_factors = []
            
            # Technical risk factors
            if technical_data and 'timeframes' in technical_data:
                # Check for agreement across timeframes
                signals = []
                for tf, analysis in technical_data['timeframes'].items():
                    if 'signal' in analysis:
                        signals.append(analysis['signal'].get('direction', 'NEUTRAL'))
                
                # Count unique signals
                unique_signals = set(signals)
                if len(unique_signals) == 1 and 'NEUTRAL' not in unique_signals:
                    risk_factors.append(-1)  # Lower risk when all timeframes agree
                elif len(unique_signals) > 1 and 'NEUTRAL' not in unique_signals:
                    risk_factors.append(1)  # Higher risk when timeframes disagree
            
            # Sentiment risk factors
            if sentiment_data:
                sentiment_confidence = sentiment_data.get('confidence', 0)
                if sentiment_confidence < 40:
                    risk_factors.append(1)  # Higher risk with low sentiment confidence
                elif sentiment_confidence > 70:
                    risk_factors.append(-1)  # Lower risk with high sentiment confidence
            
            # Correlation risk factors
            if correlation_data:
                confirmation_level = correlation_data.get('confirmation_level', 0)
                if confirmation_level > 50:
                    risk_factors.append(-1)  # Lower risk with high correlation confirmation
                elif confirmation_level < 0:
                    risk_factors.append(1)  # Higher risk with negative correlation confirmation
            
            # Calculate overall risk level
            risk_score = sum(risk_factors)
            
            if risk_score < -1:
                return 'LOW'
            elif risk_score > 1:
                return 'HIGH'
            else:
                return 'MEDIUM'
                
        except Exception as e:
            logger.error(f"Error calculating risk level: {str(e)}")
            return 'MEDIUM'  # Default to medium risk
    
    def _generate_enhanced_explanation(self,
                                     technical_data: Dict,
                                     sentiment_data: Optional[Dict] = None,
                                     correlation_data: Optional[Dict] = None,
                                     enhanced_score: Dict = None) -> str:
        """
        Generate enhanced explanation for the trading recommendation.
        
        Args:
            technical_data: Technical analysis data
            sentiment_data: Sentiment analysis data
            correlation_data: Correlation analysis data
            enhanced_score: Enhanced score data
            
        Returns:
            Explanation string
        """
        try:
            explanation_parts = []
            
            # Determine signal direction
            if enhanced_score and 'score' in enhanced_score:
                direction = 'BUY' if enhanced_score['score'] > 0 else 'SELL'
                confidence = enhanced_score.get('confidence', 0)
                
                # Add signal summary
                explanation_parts.append(
                    f"{direction} signal with {confidence:.1f}% confidence."
                )
            
            # Add technical analysis explanation
            if technical_data and 'timeframes' in technical_data:
                tech_explanations = []
                
                for tf, analysis in technical_data['timeframes'].items():
                    if 'signal' in analysis:
                        signal = analysis['signal']
                        direction = signal.get('direction', 'NEUTRAL')
                        
                        if direction != 'NEUTRAL':
                            tech_explanations.append(
                                f"{tf} timeframe: {direction} signal"
                            )
                
                if tech_explanations:
                    explanation_parts.append(
                        "Technical Analysis: " + "; ".join(tech_explanations)
                    )
            
            # Add sentiment explanation
            if sentiment_data:
                sentiment = sentiment_data.get('overall_sentiment', 'NEUTRAL')
                score = sentiment_data.get('combined_score', 0)
                
                if sentiment != 'NEUTRAL':
                    explanation_parts.append(
                        f"Market Sentiment: {sentiment} (score: {score:.1f})"
                    )
            
            # Add correlation explanation
            if correlation_data:
                confirmation = correlation_data.get('correlated_confirmation', False)
                level = correlation_data.get('confirmation_level', 0)
                
                if confirmation:
                    confirming_pairs = correlation_data.get('confirming_pairs', [])
                    if confirming_pairs:
                        pairs_str = ", ".join([p for p, _ in confirming_pairs[:3]])
                        explanation_parts.append(
                            f"Correlation: Signal confirmed by {pairs_str} (level: {level:.1f}%)"
                        )
                elif level < 0:
                    contradicting_pairs = correlation_data.get('contradicting_pairs', [])
                    if contradicting_pairs:
                        pairs_str = ", ".join([p for p, _ in contradicting_pairs[:3]])
                        explanation_parts.append(
                            f"Warning: Signal contradicted by {pairs_str} (level: {abs(level):.1f}%)"
                        )
            
            # Combine all explanations
            if explanation_parts:
                return " ".join(explanation_parts)
            else:
                return "No detailed explanation available."
                
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return "Signal generated based on technical analysis."
    
    def _get_timeframe_weight(self, timeframe: str) -> float:
        """
        Get weight for a timeframe in signal calculation.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '5m', '15m')
            
        Returns:
            Weight value
        """
        # Higher weights for longer timeframes
        weights = {
            '1m': 0.6,
            '3m': 0.7,
            '5m': 0.8,
            '10m': 0.9,
            '15m': 1.0,
            '30m': 1.1,
            '1h': 1.2
        }
        
        return weights.get(timeframe, 0.7)  # Default weight if not specified
i m p o r t  
 r a n d o m  
 f r o m  
 d a t e t i m e  
 i m p o r t  
 d a t e t i m e  
 