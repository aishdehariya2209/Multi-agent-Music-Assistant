import praw
import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
@dataclass
class Comment:
    text: str
    score: int
    author: str
    created_utc: float
    subreddit: str
    post_title: str
    url: str

@dataclass
class SentimentResult:
    overall_sentiment: Dict[str, float]
    platform_breakdown: Dict[str, Dict[str, float]]
    key_themes: List[Tuple[str, int]]
    temporal_trends: Dict[str, float]
    sample_quotes: Dict[str, List[str]]
    total_comments: int
    confidence_score: float

class RedditSentimentAnalyzer:
    def __init__(self, reddit_config: Dict[str, str], llm):
        """
        Initialize Reddit Sentiment Analyzer
        
        Args:
            reddit_config: Dict with keys: client_id, client_secret, user_agent
            google_api_key: Google API key for Gemini 2.0 Flash
        """
        self.reddit = praw.Reddit(
            client_id=reddit_config['client_id'],
            client_secret=reddit_config['client_secret'],
            user_agent=reddit_config['user_agent']
        )
        
        # Initialize Gemini 2.0 Flash model
        self.llm_client = llm
        self.logger = logging.getLogger(__name__)
        
        # Target subreddits for different types of analysis
        self.music_subreddits = [
            'Music', 'hiphopheads', 'popheads', 'indieheads', 
            'trap', 'rnb', 'rock', 'country', 'jazz'
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess comment text"""
        # Remove Reddit markdown
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'~~(.*?)~~', r'\1', text)      # Strikethrough
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def is_valid_comment(self, comment_text: str) -> bool:
        """Filter out low-quality comments"""
        # Minimum length check
        if len(comment_text.split()) < 3:
            return False
        
        # Filter out common low-effort comments
        low_effort_patterns = [
            r'^(this|fire|trash|mid|goat|facts|cap|fr|real)[\s\.\!]*$',
            r'^[\d\s\.\!\?]*$',  # Only numbers and punctuation
            r'^[emoji\s]*$',     # Only emojis
        ]
        
        for pattern in low_effort_patterns:
            if re.match(pattern, comment_text.lower().strip()):
                return False
        
        return True
    
    def search_artist_mentions(self, artist_name: str, limit: int = 50) -> List[Comment]:
        """Search for artist mentions across multiple subreddits"""
        comments = []
        
        # Search in general music subreddits
        for subreddit_name in self.music_subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search recent posts mentioning the artist
                search_query = f'"{artist_name}" OR {artist_name}'
                posts = subreddit.search(search_query, sort='new', time_filter='month', limit=20)
                
                for post in posts:
                    post.comments.replace_more(limit=0)  # Don't expand "more comments"
                    
                    # Get top-level comments
                    for comment in post.comments[:10]:  # Top 10 comments per post
                        if hasattr(comment, 'body') and comment.body != '[deleted]':
                            cleaned_text = self.clean_text(comment.body)
                            
                            if self.is_valid_comment(cleaned_text) and artist_name.lower() in cleaned_text.lower():
                                comments.append(Comment(
                                    text=cleaned_text,
                                    score=comment.score,
                                    author=str(comment.author) if comment.author else 'deleted',
                                    created_utc=comment.created_utc,
                                    subreddit=subreddit_name,
                                    post_title=post.title,
                                    url=f"https://reddit.com{comment.permalink}"
                                ))
                
                if len(comments) >= limit:
                    break
                    
            except Exception as e:
                self.logger.warning(f"Error scraping r/{subreddit_name}: {e}")
                continue
        
        # Also search for artist-specific subreddit
        artist_sub_name = artist_name.lower().replace(' ', '').replace('-', '')
        try:
            artist_subreddit = self.reddit.subreddit(artist_sub_name)
            recent_posts = artist_subreddit.hot(limit=10)
            
            for post in recent_posts:
                post.comments.replace_more(limit=0)
                for comment in post.comments[:5]:
                    if hasattr(comment, 'body') and comment.body != '[deleted]':
                        cleaned_text = self.clean_text(comment.body)
                        if self.is_valid_comment(cleaned_text):
                            comments.append(Comment(
                                text=cleaned_text,
                                score=comment.score,
                                author=str(comment.author) if comment.author else 'deleted',
                                created_utc=comment.created_utc,
                                subreddit=artist_sub_name,
                                post_title=post.title,
                                url=f"https://reddit.com{comment.permalink}"
                            ))
        except:
            pass  # Artist-specific subreddit might not exist
        
        return comments[:limit]
    
    async def analyze_sentiment_batch(self, comments: List[Comment]) -> Dict:
        """Analyze sentiment using Gemini 2.0 Flash model"""
        # Prepare comments for LLM analysis
        comment_texts = [c.text for c in comments]
        
        # Create analysis prompt
        prompt = f"""
        Analyze the sentiment of these music-related comments about an artist. For each comment, classify the sentiment as:
        - positive (enthusiastic, praising, supportive, loving the music)
        - negative (critical, disappointed, disliking, harsh criticism)
        - neutral (factual, mixed feelings, or unclear sentiment)
        
        Also identify key themes/topics mentioned across all comments such as:
        - vocals/singing
        - production/beats
        - lyrics/songwriting
        - live performance
        - album/project quality
        - artistic growth/evolution
        
        Comments to analyze:
        {json.dumps(comment_texts, indent=2)}
        
        Return ONLY a valid JSON response with this exact structure:
        {{
            "sentiments": ["positive", "negative", "neutral", ...],
            "themes": ["vocals", "production", "lyrics", ...],
            "confidence": 0.85
        }}
        
        Ensure the sentiments array has exactly {len(comment_texts)} elements corresponding to each comment.
        """
        
        try:
            response = await asyncio.to_thread(self.llm_client.invoke, prompt)
            response_text = response.content.strip()
            
            # Clean the response if it has markdown formatting
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].strip()
            
            result = json.loads(response_text)
            
            # Validate the response structure
            if not all(key in result for key in ['sentiments', 'themes', 'confidence']):
                raise ValueError("Invalid response structure")
            
            if len(result['sentiments']) != len(comment_texts):
                print("Sentiment count mismatch")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Gemini analysis failed: {e}")
            raise Exception(f"Sentiment analysis failed: {e}")
    

    def calculate_temporal_trends(self, comments: List[Comment], sentiments: List[str]) -> Dict[str, float]:
        """Calculate sentiment trends over time"""
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)
        
        recent_comments = []
        older_comments = []
        
        for i, comment in enumerate(comments):
            comment_date = datetime.fromtimestamp(comment.created_utc)
            if comment_date > week_ago:
                recent_comments.append(sentiments[i])
            elif comment_date > month_ago:
                older_comments.append(sentiments[i])
        
        def sentiment_score(sentiment_list):
            if not sentiment_list:
                return 0.5
            pos = sentiment_list.count('positive') / len(sentiment_list)
            neg = sentiment_list.count('negative') / len(sentiment_list)
            return pos - neg + 0.5  # Normalize to 0-1 scale
        
        return {
            'recent_week': sentiment_score(recent_comments),
            'past_month': sentiment_score(older_comments),
            'trend_direction': 'improving' if sentiment_score(recent_comments) > sentiment_score(older_comments) else 'declining'
        }
    
    async def analyze_artist_sentiment(self, artist_name: str) -> SentimentResult:
        """Main function to analyze artist sentiment from Reddit"""
        self.logger.info(f"Starting sentiment analysis for: {artist_name}")
        
        # Step 1: Scrape comments
        comments = await asyncio.to_thread(self.search_artist_mentions, artist_name, 100)
        
        if not comments:
            return SentimentResult(
                overall_sentiment={'positive': 0, 'negative': 0, 'neutral': 0},
                platform_breakdown={},
                key_themes=[],
                temporal_trends={},
                sample_quotes={'positive': [], 'negative': [], 'neutral': []},
                total_comments=0,
                confidence_score=0.0
            )
        
        # Step 2: Analyze sentiment
        analysis_result = await self.analyze_sentiment_batch(comments)
        sentiments = analysis_result['sentiments']
        themes = analysis_result['themes']
        confidence = analysis_result['confidence']
        
        # Step 3: Calculate metrics
        sentiment_counts = Counter(sentiments)
        total = len(sentiments)
        
        overall_sentiment = {
            'positive': sentiment_counts['positive'] / total,
            'negative': sentiment_counts['negative'] / total,
            'neutral': sentiment_counts['neutral'] / total
        }
        
        # Platform breakdown
        platform_breakdown = {}
        for subreddit in set(c.subreddit for c in comments):
            subreddit_sentiments = [sentiments[i] for i, c in enumerate(comments) if c.subreddit == subreddit]
            subreddit_counts = Counter(subreddit_sentiments)
            subreddit_total = len(subreddit_sentiments)
            
            platform_breakdown[subreddit] = {
                'positive': subreddit_counts['positive'] / subreddit_total,
                'negative': subreddit_counts['negative'] / subreddit_total,
                'neutral': subreddit_counts['neutral'] / subreddit_total
            }
        
        # Sample quotes
        sample_quotes = {'positive': [], 'negative': [], 'neutral': []}
        for i, comment in enumerate(comments):
            sentiment = sentiments[i]
            if len(sample_quotes[sentiment]) < 3:  # Max 3 examples per sentiment
                sample_quotes[sentiment].append({
                    'text': comment.text[:200] + '...' if len(comment.text) > 200 else comment.text,
                    'score': comment.score,
                    'subreddit': comment.subreddit
                })
        
        # Temporal trends
        temporal_trends = self.calculate_temporal_trends(comments, sentiments)
        
        # Key themes
        key_themes = Counter(themes).most_common(10)
        
        return SentimentResult(
            overall_sentiment=overall_sentiment,
            platform_breakdown=platform_breakdown,
            key_themes=key_themes,
            temporal_trends=temporal_trends,
            sample_quotes=sample_quotes,
            total_comments=total,
            confidence_score=confidence
        )

# Usage example for your agent
async def execute_sentiment_analysis(artist_name: str, reddit_config: Dict, google_api_key: str) -> Dict:
    """
    Main function to be called by your sentiment analysis agent
    
    Args:
        artist_name: Name of the artist to analyze
        reddit_config: Reddit API credentials
        google_api_key: Google API key for Gemini 2.0 Flash
    
    Returns:
        Dictionary with sentiment analysis results
    """
    analyzer = RedditSentimentAnalyzer(reddit_config, google_api_key)
    result = await analyzer.analyze_artist_sentiment(artist_name)
    
    # Format for agent response
    return {
        'artist': artist_name,
        'sentiment_analysis': {
            'overall_sentiment': result.overall_sentiment,
            'breakdown_by_platform': result.platform_breakdown,
            'key_discussion_themes': result.key_themes,
            'temporal_trends': result.temporal_trends,
            'representative_quotes': result.sample_quotes,
            'data_summary': {
                'total_comments_analyzed': result.total_comments,
                'analysis_confidence': result.confidence_score,
                'data_sources': list(result.platform_breakdown.keys())
            }
        }
    }

# Example agent integration
if __name__ == "__main__":
    load_dotenv()
    # Example configuration
    REDDIT_CLIENT = os.eviron["REDDIT_CLIENT_ID"]
    REDDIT_SECRET = os.environ["REDDIT_SECRET"]
    reddit_config = {
        'client_id': REDDIT_CLIENT,
        'client_secret': REDDIT_SECRET,
        'user_agent': 'SentimentAnalysisBot/1.0'
    }
    
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    )
    
    # Example usage
    async def main():
        result = await execute_sentiment_analysis("Drake", reddit_config, llm)
        print(json.dumps(result, indent=2))
    
    asyncio.run(main())