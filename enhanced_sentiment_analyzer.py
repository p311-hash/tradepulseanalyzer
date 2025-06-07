import os
import logging
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from sentiment_config import SENTIMENT_CONFIG, SENTIMENT_APIS, SIGNAL_ADJUSTMENTS

logger = logging.getLogger(__name__)

class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.config = SENTIMENT_CONFIG
        self.apis = SENTIMENT_APIS
        self.adjustments = SIGNAL_ADJUSTMENTS
        self.cache = {}
        self.vader = SentimentIntensityAnalyzer()

        # Configure request headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    async def analyze_sentiment(self, symbol: str) -> Dict:
        """Analyze sentiment from multiple sources with caching"""
        cache_key = f"sentiment_{symbol}"
        current_time = datetime.now()

        # Check cache
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if (current_time - cached['timestamp']).total_seconds() < self.config['cache_duration'] * 60:
                return cached['data']

        # Gather sentiment data from different sources
        twitter_sentiment = await self._analyze_twitter(symbol)
        stocktwits_sentiment = await self._analyze_stocktwits(symbol)
        news_sentiment = await self._analyze_news(symbol)

        # Combine sentiments with configured weights
        weights = self.config['source_weights']
        combined_score = (
            twitter_sentiment['score'] * weights['twitter'] +
            stocktwits_sentiment['score'] * weights['stocktwits'] +
            news_sentiment['score'] * weights['news']
        )

        # Calculate weighted sentiment metrics
        sentiment_data = {
            'composite_score': combined_score,
            'sources': {
                'twitter': twitter_sentiment,
                'stocktwits': stocktwits_sentiment,
                'news': news_sentiment
            },
            'momentum': self._calculate_momentum(symbol),
            'volume': self._calculate_mention_volume(symbol),
            'sentiment_level': self._get_sentiment_level(combined_score),
            'confidence': self._calculate_confidence(
                twitter_sentiment['confidence'],
                stocktwits_sentiment['confidence'],
                news_sentiment['confidence']
            ),
            'timestamp': current_time.isoformat()
        }

        # Cache the results
        self.cache[cache_key] = {
            'data': sentiment_data,
            'timestamp': current_time
        }

        return sentiment_data

    async def _analyze_twitter(self, symbol: str) -> Dict:
        """Analyze Twitter sentiment with weighted keywords"""
        if not self.apis['twitter']['api_key']:
            return {'score': 0, 'confidence': 0, 'volume': 0}

        try:
            # Implementation of Twitter API calls would go here
            # For now return neutral sentiment
            return {'score': 0, 'confidence': 0, 'volume': 0}
        except Exception as e:
            logger.error(f"Error analyzing Twitter sentiment: {str(e)}")
            return {'score': 0, 'confidence': 0, 'volume': 0}

    async def _analyze_stocktwits(self, symbol: str) -> Dict:
        """Analyze StockTwits sentiment with volume analysis"""
        if not self.apis['stocktwits']['api_key']:
            return {'score': 0, 'confidence': 0, 'volume': 0}

        try:
            # Implementation of StockTwits API calls would go here
            # For now return neutral sentiment
            return {'score': 0, 'confidence': 0, 'volume': 0}
        except Exception as e:
            logger.error(f"Error analyzing StockTwits sentiment: {str(e)}")
            return {'score': 0, 'confidence': 0, 'volume': 0}

    async def _analyze_news(self, symbol: str) -> Dict:
        """Analyze news sentiment with advanced NLP"""
        if not self.apis['news_api']['api_key']:
            return {'score': 0, 'confidence': 0, 'volume': 0}

        try:
            search_symbol = symbol.replace('_otc', '')
            headers = {'Authorization': f"Bearer {self.apis['news_api']['api_key']}"}

            # Fetch recent news
            url = f"https://newsapi.org/v2/everything?q={search_symbol}&sortBy=publishedAt&language=en"
            response = requests.get(url, headers=headers)
            data = response.json()

            if response.status_code != 200:
                return {'score': 0, 'confidence': 0, 'volume': 0}

            articles = data.get('articles', [])
            if not articles:
                return {'score': 0, 'confidence': 0, 'volume': 0}

            # Analyze sentiment for each article
            sentiments = []
            for article in articles[:10]:  # Analyze last 10 articles
                title = article.get('title', '')
                content = article.get('description', '')

                # Combine title and content with weights
                text = f"{title} {content}"
                sentiment = self._analyze_text_sentiment(text)
                sentiments.append(sentiment)

            # Calculate weighted average
            avg_sentiment = np.mean([s['score'] for s in sentiments])
            avg_confidence = np.mean([s['confidence'] for s in sentiments])

            return {
                'score': avg_sentiment,
                'confidence': avg_confidence,
                'volume': len(articles)
            }

        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {str(e)}")
            return {'score': 0, 'confidence': 0, 'volume': 0}

    def _analyze_text_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text using multiple methods"""
        try:
            # Clean text
            text = self._clean_text(text)

            # VADER sentiment
            vader_scores = self.vader.polarity_scores(text)

            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity

            # Weight and combine scores
            score = (vader_scores['compound'] * 0.7 + textblob_polarity * 0.3)
            confidence = (1 - textblob_subjectivity) * 100  # Higher objectivity = higher confidence

            # Check for weighted keywords
            score = self._adjust_for_keywords(text, score)

            return {
                'score': score,
                'confidence': confidence,
                'vader_scores': vader_scores,
                'textblob_scores': {
                    'polarity': textblob_polarity,
                    'subjectivity': textblob_subjectivity
                }
            }

        except Exception as e:
            logger.error(f"Error in text sentiment analysis: {str(e)}")
            return {'score': 0, 'confidence': 0}

    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        if not text:
            return ""

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)

        # Convert to lowercase
        text = text.lower().strip()

        return text

    def _adjust_for_keywords(self, text: str, base_score: float) -> float:
        """Adjust sentiment score based on weighted keywords"""
        score = base_score
        text = text.lower()

        for sentiment, keywords in self.config['keyword_weights'].items():
            for keyword in keywords:
                if keyword in text:
                    if sentiment == 'highly_bullish':
                        score += 0.2
                    elif sentiment == 'bullish':
                        score += 0.1
                    elif sentiment == 'bearish':
                        score -= 0.1
                    elif sentiment == 'highly_bearish':
                        score -= 0.2

        # Ensure score stays within bounds
        return max(-1.0, min(1.0, score))

    def _calculate_momentum(self, symbol: str) -> float:
        """Calculate sentiment momentum over time"""
        try:
            cache_key = f"sentiment_{symbol}"
            if cache_key not in self.cache:
                return 0

            historical = self.cache[cache_key].get('historical', [])
            if len(historical) < 2:
                return 0

            # Calculate momentum using last few sentiment scores
            recent_scores = [h['composite_score'] for h in historical[-5:]]
            momentum = np.gradient(recent_scores).mean()

            return momentum

        except Exception as e:
            logger.error(f"Error calculating momentum: {str(e)}")
            return 0

    def _calculate_mention_volume(self, symbol: str) -> str:
        """Calculate and categorize mention volume"""
        try:
            total_mentions = sum(s['volume'] for s in self.cache.get(f"sentiment_{symbol}", {}).get('sources', {}).values())

            thresholds = self.config['volume_thresholds']
            if total_mentions >= thresholds['high']:
                return 'high'
            elif total_mentions >= thresholds['medium']:
                return 'medium'
            elif total_mentions >= thresholds['low']:
                return 'low'
            else:
                return 'insufficient'

        except Exception as e:
            logger.error(f"Error calculating mention volume: {str(e)}")
            return 'insufficient'

    def _get_sentiment_level(self, score: float) -> str:
        """Get sentiment level based on score"""
        thresholds = self.config['thresholds']

        if score >= thresholds['very_bullish']:
            return 'very_bullish'
        elif score >= thresholds['bullish']:
            return 'bullish'
        elif score <= thresholds['very_bearish']:
            return 'very_bearish'
        elif score <= thresholds['bearish']:
            return 'bearish'
        else:
            return 'neutral'

    def _calculate_confidence(self, *confidence_scores: float) -> float:
        """Calculate overall confidence from multiple sources"""
        valid_scores = [s for s in confidence_scores if s > 0]
        if not valid_scores:
            return 0

        # Weight higher confidence scores more
        weighted_scores = [s * (i + 1) for i, s in enumerate(sorted(valid_scores))]
        weights_sum = sum(i + 1 for i in range(len(valid_scores)))

        return sum(weighted_scores) / weights_sum
