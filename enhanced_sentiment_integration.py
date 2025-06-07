"""
Enhanced Sentiment Analysis Integration
Complete implementation with real API connections and signal integration.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import config

logger = logging.getLogger(__name__)

class EnhancedSentimentIntegration:
    """
    Complete sentiment analysis integration with multiple data sources.
    """
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.sentiment_cache = {}
        self.cache_duration = timedelta(minutes=15)
        
        # API configurations
        self.apis = {
            'twitter': {
                'enabled': hasattr(config, 'TWITTER_API_KEY') and config.TWITTER_API_KEY,
                'api_key': getattr(config, 'TWITTER_API_KEY', ''),
                'api_secret': getattr(config, 'TWITTER_API_SECRET', ''),
                'bearer_token': getattr(config, 'TWITTER_BEARER_TOKEN', ''),
                'rate_limit': 300  # requests per 15 minutes
            },
            'reddit': {
                'enabled': hasattr(config, 'REDDIT_CLIENT_ID') and config.REDDIT_CLIENT_ID,
                'client_id': getattr(config, 'REDDIT_CLIENT_ID', ''),
                'client_secret': getattr(config, 'REDDIT_CLIENT_SECRET', ''),
                'user_agent': getattr(config, 'REDDIT_USER_AGENT', 'TradePulse/1.0'),
                'rate_limit': 60  # requests per minute
            },
            'news': {
                'enabled': hasattr(config, 'NEWS_API_KEY') and config.NEWS_API_KEY,
                'api_key': getattr(config, 'NEWS_API_KEY', ''),
                'rate_limit': 1000  # requests per day
            },
            'stocktwits': {
                'enabled': hasattr(config, 'STOCKTWITS_API_KEY') and config.STOCKTWITS_API_KEY,
                'api_key': getattr(config, 'STOCKTWITS_API_KEY', ''),
                'rate_limit': 200  # requests per hour
            }
        }
        
        # Sentiment keywords for different assets
        self.asset_keywords = {
            'EURUSD': ['EUR', 'USD', 'euro', 'dollar', 'ECB', 'Fed', 'Federal Reserve'],
            'GBPUSD': ['GBP', 'USD', 'pound', 'sterling', 'dollar', 'BoE', 'Fed'],
            'BTCUSD': ['BTC', 'bitcoin', 'crypto', 'cryptocurrency', 'digital currency'],
            'ETHUSD': ['ETH', 'ethereum', 'crypto', 'smart contract', 'DeFi'],
            'GOLD': ['gold', 'precious metals', 'safe haven', 'inflation hedge'],
            'OIL': ['oil', 'crude', 'petroleum', 'energy', 'OPEC']
        }
        
        # Sentiment weights for signal integration
        self.sentiment_weights = {
            'twitter': 0.3,
            'reddit': 0.25,
            'news': 0.35,
            'stocktwits': 0.1
        }
        
    async def get_comprehensive_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive sentiment analysis for a trading symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'BTCUSD')
            
        Returns:
            Comprehensive sentiment data
        """
        try:
            # Check cache first
            cache_key = f"sentiment_{symbol}"
            if cache_key in self.sentiment_cache:
                cached_time, cached_data = self.sentiment_cache[cache_key]
                if datetime.now() - cached_time < self.cache_duration:
                    return cached_data
            
            # Get sentiment from all available sources
            sentiment_tasks = []
            
            if self.apis['twitter']['enabled']:
                sentiment_tasks.append(self._get_twitter_sentiment(symbol))
            
            if self.apis['reddit']['enabled']:
                sentiment_tasks.append(self._get_reddit_sentiment(symbol))
            
            if self.apis['news']['enabled']:
                sentiment_tasks.append(self._get_news_sentiment(symbol))
            
            if self.apis['stocktwits']['enabled']:
                sentiment_tasks.append(self._get_stocktwits_sentiment(symbol))
            
            # Execute all sentiment analysis tasks
            sentiment_results = await asyncio.gather(*sentiment_tasks, return_exceptions=True)
            
            # Process results
            sentiment_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'sources': {},
                'overall_sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'signal_strength': 0.0,
                'volume_score': 0.0
            }
            
            # Combine sentiment from all sources
            total_weighted_sentiment = 0.0
            total_weight = 0.0
            total_volume = 0
            
            source_names = ['twitter', 'reddit', 'news', 'stocktwits']
            
            for i, result in enumerate(sentiment_results):
                if isinstance(result, Exception):
                    logger.error(f"Error getting sentiment from {source_names[i]}: {result}")
                    continue
                
                if result and 'sentiment_score' in result:
                    source_name = source_names[i]
                    weight = self.sentiment_weights.get(source_name, 0.1)
                    
                    sentiment_data['sources'][source_name] = result
                    total_weighted_sentiment += result['sentiment_score'] * weight
                    total_weight += weight
                    total_volume += result.get('volume', 0)
            
            # Calculate overall sentiment
            if total_weight > 0:
                overall_score = total_weighted_sentiment / total_weight
                sentiment_data['overall_sentiment'] = self._score_to_sentiment(overall_score)
                sentiment_data['confidence'] = min(total_weight, 1.0)
                sentiment_data['signal_strength'] = abs(overall_score) * sentiment_data['confidence']
                sentiment_data['volume_score'] = min(total_volume / 100, 1.0)  # Normalize volume
            
            # Cache the result
            self.sentiment_cache[cache_key] = (datetime.now(), sentiment_data)
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error getting comprehensive sentiment: {e}")
            return self._get_neutral_sentiment(symbol)
    
    async def _get_twitter_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment from Twitter API v2."""
        try:
            keywords = self.asset_keywords.get(symbol, [symbol])
            query = ' OR '.join(keywords)
            
            headers = {
                'Authorization': f"Bearer {self.apis['twitter']['bearer_token']}",
                'Content-Type': 'application/json'
            }
            
            params = {
                'query': f"({query}) -is:retweet lang:en",
                'max_results': 100,
                'tweet.fields': 'created_at,public_metrics,context_annotations'
            }
            
            async with aiohttp.ClientSession() as session:
                url = "https://api.twitter.com/2/tweets/search/recent"
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Twitter API error: {response.status}")
                        return {}
                    
                    data = await response.json()
                    
                    if 'data' not in data:
                        return {'sentiment_score': 0.0, 'volume': 0, 'confidence': 0.0}
                    
                    tweets = data['data']
                    sentiments = []
                    total_engagement = 0
                    
                    for tweet in tweets:
                        text = tweet['text']
                        
                        # Analyze sentiment
                        blob_sentiment = TextBlob(text).sentiment.polarity
                        vader_sentiment = self.vader.polarity_scores(text)['compound']
                        
                        # Average the two sentiment scores
                        avg_sentiment = (blob_sentiment + vader_sentiment) / 2
                        sentiments.append(avg_sentiment)
                        
                        # Calculate engagement
                        metrics = tweet.get('public_metrics', {})
                        engagement = (
                            metrics.get('like_count', 0) +
                            metrics.get('retweet_count', 0) * 2 +
                            metrics.get('reply_count', 0)
                        )
                        total_engagement += engagement
                    
                    if sentiments:
                        avg_sentiment = np.mean(sentiments)
                        sentiment_std = np.std(sentiments)
                        confidence = max(0.1, 1.0 - sentiment_std)  # Lower std = higher confidence
                        
                        return {
                            'sentiment_score': avg_sentiment,
                            'volume': len(tweets),
                            'confidence': confidence,
                            'engagement': total_engagement,
                            'source': 'twitter'
                        }
                    
                    return {'sentiment_score': 0.0, 'volume': 0, 'confidence': 0.0}
                    
        except Exception as e:
            logger.error(f"Error getting Twitter sentiment: {e}")
            return {}
    
    async def _get_reddit_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment from Reddit API."""
        try:
            # Reddit OAuth authentication
            auth_data = {
                'grant_type': 'client_credentials'
            }
            
            auth = aiohttp.BasicAuth(
                self.apis['reddit']['client_id'],
                self.apis['reddit']['client_secret']
            )
            
            headers = {
                'User-Agent': self.apis['reddit']['user_agent']
            }
            
            async with aiohttp.ClientSession() as session:
                # Get access token
                async with session.post(
                    'https://www.reddit.com/api/v1/access_token',
                    data=auth_data,
                    auth=auth,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        logger.error(f"Reddit auth error: {response.status}")
                        return {}
                    
                    token_data = await response.json()
                    access_token = token_data['access_token']
                
                # Search for posts
                keywords = self.asset_keywords.get(symbol, [symbol])
                query = ' OR '.join(keywords)
                
                search_headers = {
                    'Authorization': f"Bearer {access_token}",
                    'User-Agent': self.apis['reddit']['user_agent']
                }
                
                params = {
                    'q': query,
                    'sort': 'new',
                    'limit': 50,
                    't': 'day'  # Last 24 hours
                }
                
                subreddits = ['investing', 'stocks', 'forex', 'cryptocurrency', 'wallstreetbets']
                all_posts = []
                
                for subreddit in subreddits:
                    url = f"https://oauth.reddit.com/r/{subreddit}/search"
                    async with session.get(url, headers=search_headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            posts = data.get('data', {}).get('children', [])
                            all_posts.extend(posts)
                
                if not all_posts:
                    return {'sentiment_score': 0.0, 'volume': 0, 'confidence': 0.0}
                
                sentiments = []
                total_score = 0
                
                for post in all_posts:
                    post_data = post['data']
                    title = post_data.get('title', '')
                    text = post_data.get('selftext', '')
                    combined_text = f"{title} {text}"
                    
                    if len(combined_text.strip()) > 10:  # Minimum text length
                        # Analyze sentiment
                        blob_sentiment = TextBlob(combined_text).sentiment.polarity
                        vader_sentiment = self.vader.polarity_scores(combined_text)['compound']
                        
                        avg_sentiment = (blob_sentiment + vader_sentiment) / 2
                        sentiments.append(avg_sentiment)
                        
                        # Weight by Reddit score
                        score = post_data.get('score', 0)
                        total_score += max(score, 1)  # Minimum weight of 1
                
                if sentiments:
                    # Weighted average sentiment
                    weighted_sentiment = np.average(sentiments, weights=[max(1, s) for s in sentiments])
                    confidence = min(len(sentiments) / 20, 1.0)  # More posts = higher confidence
                    
                    return {
                        'sentiment_score': weighted_sentiment,
                        'volume': len(sentiments),
                        'confidence': confidence,
                        'total_score': total_score,
                        'source': 'reddit'
                    }
                
                return {'sentiment_score': 0.0, 'volume': 0, 'confidence': 0.0}
                
        except Exception as e:
            logger.error(f"Error getting Reddit sentiment: {e}")
            return {}
    
    async def _get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment from news articles."""
        try:
            keywords = self.asset_keywords.get(symbol, [symbol])
            query = ' OR '.join(keywords)
            
            params = {
                'q': query,
                'apiKey': self.apis['news']['api_key'],
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 50,
                'from': (datetime.now() - timedelta(hours=24)).isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                url = "https://newsapi.org/v2/everything"
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"News API error: {response.status}")
                        return {}
                    
                    data = await response.json()
                    articles = data.get('articles', [])
                    
                    if not articles:
                        return {'sentiment_score': 0.0, 'volume': 0, 'confidence': 0.0}
                    
                    sentiments = []
                    
                    for article in articles:
                        title = article.get('title', '')
                        description = article.get('description', '')
                        content = f"{title} {description}"
                        
                        if len(content.strip()) > 20:
                            # Analyze sentiment
                            blob_sentiment = TextBlob(content).sentiment.polarity
                            vader_sentiment = self.vader.polarity_scores(content)['compound']
                            
                            avg_sentiment = (blob_sentiment + vader_sentiment) / 2
                            sentiments.append(avg_sentiment)
                    
                    if sentiments:
                        avg_sentiment = np.mean(sentiments)
                        confidence = min(len(sentiments) / 30, 1.0)
                        
                        return {
                            'sentiment_score': avg_sentiment,
                            'volume': len(sentiments),
                            'confidence': confidence,
                            'source': 'news'
                        }
                    
                    return {'sentiment_score': 0.0, 'volume': 0, 'confidence': 0.0}
                    
        except Exception as e:
            logger.error(f"Error getting news sentiment: {e}")
            return {}
    
    async def _get_stocktwits_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment from StockTwits API."""
        try:
            # StockTwits uses different symbol format
            stocktwits_symbol = symbol.replace('USD', '').replace('EUR', '').replace('GBP', '')
            
            params = {
                'access_token': self.apis['stocktwits']['api_key']
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"https://api.stocktwits.com/api/2/streams/symbol/{stocktwits_symbol}.json"
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"StockTwits API error: {response.status}")
                        return {}
                    
                    data = await response.json()
                    messages = data.get('messages', [])
                    
                    if not messages:
                        return {'sentiment_score': 0.0, 'volume': 0, 'confidence': 0.0}
                    
                    sentiments = []
                    bullish_count = 0
                    bearish_count = 0
                    
                    for message in messages:
                        # StockTwits provides sentiment labels
                        entities = message.get('entities', {})
                        sentiment_data = entities.get('sentiment')
                        
                        if sentiment_data:
                            if sentiment_data['basic'] == 'Bullish':
                                sentiments.append(0.5)
                                bullish_count += 1
                            elif sentiment_data['basic'] == 'Bearish':
                                sentiments.append(-0.5)
                                bearish_count += 1
                        else:
                            # Analyze text sentiment if no label
                            text = message.get('body', '')
                            if text:
                                vader_sentiment = self.vader.polarity_scores(text)['compound']
                                sentiments.append(vader_sentiment)
                    
                    if sentiments:
                        avg_sentiment = np.mean(sentiments)
                        confidence = min(len(sentiments) / 20, 1.0)
                        
                        return {
                            'sentiment_score': avg_sentiment,
                            'volume': len(sentiments),
                            'confidence': confidence,
                            'bullish_count': bullish_count,
                            'bearish_count': bearish_count,
                            'source': 'stocktwits'
                        }
                    
                    return {'sentiment_score': 0.0, 'volume': 0, 'confidence': 0.0}
                    
        except Exception as e:
            logger.error(f"Error getting StockTwits sentiment: {e}")
            return {}
    
    def _score_to_sentiment(self, score: float) -> str:
        """Convert numerical sentiment score to categorical sentiment."""
        if score > 0.1:
            return 'BULLISH'
        elif score < -0.1:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _get_neutral_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Return neutral sentiment as fallback."""
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'overall_sentiment': 'NEUTRAL',
            'confidence': 0.0,
            'signal_strength': 0.0,
            'volume_score': 0.0
        }
    
    def integrate_with_signal(self, signal: Dict[str, Any], sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate sentiment analysis with trading signal.
        
        Args:
            signal: Original trading signal
            sentiment_data: Sentiment analysis results
            
        Returns:
            Enhanced signal with sentiment integration
        """
        try:
            # Create enhanced signal
            enhanced_signal = signal.copy()
            enhanced_signal['sentiment'] = sentiment_data
            
            # Adjust signal confidence based on sentiment
            original_confidence = signal.get('confidence', 0.5)
            sentiment_strength = sentiment_data.get('signal_strength', 0.0)
            sentiment_direction = sentiment_data.get('overall_sentiment', 'NEUTRAL')
            signal_direction = signal.get('direction', 'NEUTRAL')
            
            # Check if sentiment aligns with signal
            sentiment_alignment = self._check_sentiment_alignment(signal_direction, sentiment_direction)
            
            if sentiment_alignment:
                # Sentiment supports signal - boost confidence
                confidence_boost = sentiment_strength * 0.2  # Max 20% boost
                enhanced_signal['confidence'] = min(1.0, original_confidence + confidence_boost)
                enhanced_signal['sentiment_boost'] = confidence_boost
            else:
                # Sentiment contradicts signal - reduce confidence
                confidence_reduction = sentiment_strength * 0.15  # Max 15% reduction
                enhanced_signal['confidence'] = max(0.1, original_confidence - confidence_reduction)
                enhanced_signal['sentiment_penalty'] = confidence_reduction
            
            # Add sentiment score to signal
            enhanced_signal['sentiment_score'] = sentiment_data.get('signal_strength', 0.0)
            enhanced_signal['sentiment_volume'] = sentiment_data.get('volume_score', 0.0)
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error integrating sentiment with signal: {e}")
            return signal
    
    def _check_sentiment_alignment(self, signal_direction: str, sentiment_direction: str) -> bool:
        """Check if sentiment aligns with signal direction."""
        if sentiment_direction == 'NEUTRAL':
            return True  # Neutral sentiment doesn't contradict
        
        bullish_signals = ['BUY', 'LONG', 'BULLISH']
        bearish_signals = ['SELL', 'SHORT', 'BEARISH']
        
        if signal_direction.upper() in bullish_signals and sentiment_direction == 'BULLISH':
            return True
        elif signal_direction.upper() in bearish_signals and sentiment_direction == 'BEARISH':
            return True
        
        return False
