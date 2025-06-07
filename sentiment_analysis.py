"""
Sentiment analysis module for market sentiment tracking and analysis.
Combines multiple sentiment indicators to generate a comprehensive market mood assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import requests
from datetime import datetime, timedelta
import json
import os
from textblob import TextBlob
import regex as re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from requests.exceptions import RequestException
import urllib3
from bs4 import BeautifulSoup

# Disable insecure HTTPS warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer with enhanced sentiment tools."""
        self.sentiment_cache = {}
        self.cache_duration = timedelta(minutes=5)  # Shorter cache for binary options
        self.vader = SentimentIntensityAnalyzer()
        
        # API configuration
        self.news_api_key = os.getenv('NEWS_API_KEY', '')
        self.marketaux_api_key = os.getenv('MARKETAUX_API_KEY', '')
        
        # Configure request headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Initialize weights optimized for binary options
        self.weights = {
            'technicals': 0.7,    # Higher weight on technicals for short-term
            'news': 0.2,         # Less weight on news due to shorter timeframes
            'social': 0.1        # Minimal social sentiment for quick trades
        }
        
        # Binary options specific settings
        self.min_payout_threshold = 0.75  # Minimum 75% payout required
        self.expiry_times = [60, 180, 300]  # Common binary option expiry times in seconds
        self.trade_amount = 0  # Will be set based on account balance
        self.max_risk_per_trade = 0.02  # Maximum 2% risk per trade
        
        # Initialize NLP models
        self.initialize_nlp_models()

    def initialize_nlp_models(self):
        """Initialize and download required NLP models."""
        try:
            import nltk
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            logger.warning(f"Could not download NLTK data: {str(e)}")

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis."""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Convert to lowercase
        text = text.lower().strip()
        
        return text

    def get_weighted_sentiment(self, text: str, weights: Dict[str, float] = None) -> Dict:
        """Get sentiment using multiple analyzers with weights."""
        if not text:
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}
        
        if weights is None:
            weights = {
                'vader': 0.6,
                'textblob': 0.4
            }
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_scores = {
            'compound': blob.sentiment.polarity,
            'pos': max(0, blob.sentiment.polarity),
            'neg': abs(min(0, blob.sentiment.polarity)),
            'neu': 1 - abs(blob.sentiment.polarity)
        }
        
        # Weighted combination
        combined = {
            'compound': vader_scores['compound'] * weights['vader'] + textblob_scores['compound'] * weights['textblob'],
            'pos': vader_scores['pos'] * weights['vader'] + textblob_scores['pos'] * weights['textblob'],
            'neg': vader_scores['neg'] * weights['vader'] + textblob_scores['neg'] * weights['textblob'],
            'neu': vader_scores['neu'] * weights['vader'] + textblob_scores['neu'] * weights['textblob']
        }
        
        return combined

    def analyze_technical_sentiment(self, data: pd.DataFrame) -> Dict:
        """Analyze sentiment based on technical indicators."""
        try:
            sentiment_score = 0
            signal_weights = {}
            signal_count = 0
            
            # RSI analysis
            if 'rsi' in data.columns:
                rsi = data['rsi'].iloc[-1]
                rsi_prev = data['rsi'].iloc[-2] if len(data) > 1 else rsi
                
                if rsi < 30 and rsi > rsi_prev:
                    sentiment_score += 1.5
                    signal_weights['rsi'] = 1.5
                elif rsi > 70 and rsi < rsi_prev:
                    sentiment_score -= 1.5
                    signal_weights['rsi'] = -1.5
                elif rsi < 40:
                    sentiment_score += 0.5
                    signal_weights['rsi'] = 0.5
                elif rsi > 60:
                    sentiment_score -= 0.5
                    signal_weights['rsi'] = -0.5
                signal_count += 1
            
            # More technical indicators (MACD, MA, BB, etc.) can be added here
            
            # Calculate weighted sentiment score
            total_weight = sum(abs(w) for w in signal_weights.values())
            if total_weight > 0:
                normalized_score = (sum(signal_weights.values()) / total_weight) * 100
            else:
                normalized_score = 0
            
            return {
                'sentiment': 'BULLISH' if normalized_score > 20 else 'BEARISH' if normalized_score < -20 else 'NEUTRAL',
                'score': normalized_score,
                'confidence': abs(normalized_score)
            }
            
        except Exception as e:
            logger.error(f"Error in technical sentiment analysis: {str(e)}")
            return {'sentiment': 'NEUTRAL', 'score': 0, 'confidence': 0}

    def analyze_news_sentiment(self, symbol: str) -> Dict:
        """Analyze sentiment from news sources."""
        try:
            # Check cache first
            cache_key = f"news_{symbol}"
            if cache_key in self.sentiment_cache:
                cached_result = self.sentiment_cache[cache_key]
                if datetime.now() - cached_result['timestamp'] < self.cache_duration:
                    return cached_result['data']
            
            # Setup request headers
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            # Format symbol for news search
            search_symbol = symbol.replace('_otc', '').lower()
            
            # Fetch financial news
            news_urls = [
                f"https://api.marketaux.com/v1/news/all?symbols={search_symbol}&api_token={os.getenv('MARKETAUX_API_KEY', '')}",
                f"https://newsapi.org/v2/everything?q={search_symbol}&apiKey={os.getenv('NEWS_API_KEY', '')}"
            ]
            
            combined_sentiment = 0
            article_count = 0
            recent_headlines = []
            
            for url in news_urls:
                try:
                    response = requests.get(url, headers=headers, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        articles = data.get('articles', data.get('data', []))
                        
                        for article in articles[:10]:
                            title = article.get('title', '')
                            if title:
                                blob = TextBlob(title)
                                sentiment = blob.sentiment.polarity
                                combined_sentiment += sentiment
                                article_count += 1
                                recent_headlines.append({
                                    'title': title,
                                    'sentiment': sentiment
                                })
                except Exception as e:
                    logger.error(f"Error fetching news from {url}: {str(e)}")
                    continue
            
            if article_count > 0:
                avg_sentiment = combined_sentiment / article_count
                sentiment_score = avg_sentiment * 100
                confidence = min(100, (article_count * 10) * abs(avg_sentiment))
                
                sentiment_label = 'BULLISH' if sentiment_score > 20 else 'BEARISH' if sentiment_score < -20 else 'NEUTRAL'
                
                news_sentiment = {
                    'sentiment': sentiment_label,
                    'score': sentiment_score,
                    'confidence': confidence,
                    'sources': ['financial_news', 'market_news'],
                    'article_count': article_count,
                    'recent_headlines': recent_headlines[:5],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                news_sentiment = {
                    'sentiment': 'NEUTRAL',
                    'score': 0,
                    'confidence': 30,
                    'sources': ['financial_news', 'market_news'],
                    'article_count': 0,
                    'recent_headlines': [],
                    'timestamp': datetime.now().isoformat()
                }
            
            # Cache the result
            try:
                self.sentiment_cache[cache_key] = {
                    'data': news_sentiment,
                    'timestamp': datetime.now()
                }
            except Exception as e:
                logger.error(f"Error caching news sentiment: {str(e)}")
            
            return news_sentiment
            
        except Exception as e:
            logger.error(f"Error in news sentiment analysis: {str(e)}")
            return {'sentiment': 'NEUTRAL', 'score': 0, 'confidence': 0}

    def analyze_social_sentiment(self, symbol: str) -> Dict:
        """Analyze sentiment from social media sources."""
        try:
            # Check cache first
            cache_key = f"social_{symbol}"
            if cache_key in self.sentiment_cache:
                cached_result = self.sentiment_cache[cache_key]
                if datetime.now() - cached_result['timestamp'] < self.cache_duration:
                    return cached_result['data']
            
            # Format symbol for search
            search_symbol = symbol.replace('_otc', '').upper()
            
            social_sentiment = {
                'reddit': {'sentiment': 'NEUTRAL', 'score': 0, 'confidence': 20, 'source': 'reddit'},
                'stocktwits': {'sentiment': 'NEUTRAL', 'score': 0, 'confidence': 20, 'source': 'stocktwits'},
                'twitter': {'sentiment': 'NEUTRAL', 'score': 0, 'confidence': 20, 'source': 'twitter'}
            }
            
            # Calculate combined sentiment
            valid_scores = [s['score'] for s in social_sentiment.values() if s['score'] != 0]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                confidence = min(100, len(valid_scores) * 30)
            else:
                avg_score = 0
                confidence = 0
            
            result = {
                'sentiment': 'BULLISH' if avg_score > 20 else 'BEARISH' if avg_score < -20 else 'NEUTRAL',
                'score': avg_score,
                'confidence': confidence,
                'sources': list(social_sentiment.keys()),
                'details': social_sentiment,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            try:
                self.sentiment_cache[cache_key] = {
                    'data': result,
                    'timestamp': datetime.now()
                }
            except Exception as e:
                logger.error(f"Error caching social sentiment: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in social sentiment analysis: {str(e)}")
            return {'sentiment': 'NEUTRAL', 'score': 0, 'confidence': 0}

    def analyze_market_sentiment(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Analyze overall market sentiment."""
        try:
            tech_sentiment = self.analyze_technical_sentiment(data)
            news_sentiment = self.analyze_news_sentiment(symbol)
            social_sentiment = self.analyze_social_sentiment(symbol)
            
            tech_weight = self.weights['technicals']
            news_weight = self.weights['news']
            social_weight = self.weights['social']
            
            combined_score = (
                tech_sentiment['score'] * tech_weight +
                news_sentiment['score'] * news_weight +
                social_sentiment['score'] * social_weight
            )
            
            confidence = (
                tech_sentiment['confidence'] * tech_weight +
                news_sentiment['confidence'] * news_weight +
                social_sentiment['confidence'] * social_weight
            )
            
            return {
                'sentiment': 'BULLISH' if combined_score > 25 else 'BEARISH' if combined_score < -20 else 'NEUTRAL',
                'score': combined_score,
                'confidence': confidence,
                'combined_score': combined_score,
                'overall_sentiment': 'BULLISH' if combined_score > 25 else 'BEARISH' if combined_score < -20 else 'NEUTRAL',
                'technical': tech_sentiment,
                'news': news_sentiment,
                'social': social_sentiment
            }
            
        except Exception as e:
            logger.error(f"Error in market sentiment analysis: {str(e)}")
            return {
                'sentiment': 'NEUTRAL',
                'score': 0,
                'confidence': 0,
                'combined_score': 0,
                'overall_sentiment': 'NEUTRAL',
                'technical': {'sentiment': 'NEUTRAL', 'score': 0, 'confidence': 0},
                'news': {'sentiment': 'NEUTRAL', 'score': 0, 'confidence': 0},
                'social': {'sentiment': 'NEUTRAL', 'score': 0, 'confidence': 0}
            }
