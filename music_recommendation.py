import requests
import json
from typing import List, Dict, Optional, Union
import time
import os
from dotenv import load_dotenv
class LastFMRecommendationTool:
    """
    A recommendation tool using Last.fm API for music recommendations.
    Supports artist, track, and album recommendations based on user preferences.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://ws.audioscrobbler.com/2.0/"
        self.session = requests.Session()
        
    def _make_request(self, method: str, params: Dict) -> Optional[Dict]:
        """Make a request to Last.fm API with error handling and rate limiting."""
        params.update({
            'api_key': self.api_key,
            'method': method,
            'format': 'json'
        })
        
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for Last.fm API errors
            if 'error' in data:
                print(f"Last.fm API Error: {data['message']}")
                return None
                
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
    
    def get_similar_artists(self, artist: str, limit: int = 10) -> List[Dict]:
        """Get similar artists based on a given artist."""
        data = self._make_request('artist.getsimilar', {
            'artist': artist,
            'limit': limit
        })
        
        if not data or 'similarartists' not in data:
            return []
        
        similar_artists = data['similarartists'].get('artist', [])
        
        recommendations = []
        for artist_data in similar_artists:
            recommendations.append({
                'type': 'artist',
                'name': artist_data['name'],
                'match_score': float(artist_data.get('match', 0)),
                'url': artist_data.get('url', ''),
                'image': artist_data.get('image', [{}])[-1].get('#text', '') if artist_data.get('image') else ''
            })
        
        return recommendations
    
    def get_similar_tracks(self, artist: str, track: str, limit: int = 10) -> List[Dict]:
        """Get similar tracks based on a given track."""
        data = self._make_request('track.getsimilar', {
            'artist': artist,
            'track': track,
            'limit': limit
        })
        
        if not data or 'similartracks' not in data:
            return []
        
        similar_tracks = data['similartracks'].get('track', [])
        
        recommendations = []
        for track_data in similar_tracks:
            recommendations.append({
                'type': 'track',
                'name': track_data['name'],
                'artist': track_data['artist']['name'],
                'match_score': float(track_data.get('match', 0)),
                'url': track_data.get('url', ''),
                'duration': track_data.get('duration', 0)
            })
        
        print("Getting similar tracks to ", artist , track)
        print(recommendations)
        return recommendations
    
    def get_top_tracks_by_artist(self, artist: str, limit: int = 10) -> List[Dict]:
        """Get top tracks for a specific artist."""
        data = self._make_request('artist.gettoptracks', {
            'artist': artist,
            'limit': limit
        })
        
        if not data or 'toptracks' not in data:
            return []
        
        top_tracks = data['toptracks'].get('track', [])
        
        recommendations = []
        for track_data in top_tracks:
            recommendations.append({
                'type': 'track',
                'name': track_data['name'],
                'artist': track_data['artist']['name'],
                'playcount': int(track_data.get('playcount', 0)),
                'listeners': int(track_data.get('listeners', 0)),
                'url': track_data.get('url', '')
            })
        
        return recommendations
    
    def get_top_albums_by_artist(self, artist: str, limit: int = 10) -> List[Dict]:
        """Get top albums for a specific artist."""
        data = self._make_request('artist.gettopalbums', {
            'artist': artist,
            'limit': limit
        })
        
        if not data or 'topalbums' not in data:
            return []
        
        top_albums = data['topalbums'].get('album', [])
        
        recommendations = []
        for album_data in top_albums:
            recommendations.append({
                'type': 'album',
                'name': album_data['name'],
                'artist': album_data['artist']['name'],
                'playcount': int(album_data.get('playcount', 0)),
                'url': album_data.get('url', ''),
                'image': album_data.get('image', [{}])[-1].get('#text', '') if album_data.get('image') else ''
            })
        
        return recommendations
    
    def get_recommendations_by_tags(self, tags: List[str], limit: int = 10) -> List[Dict]:
        """Get artist recommendations based on genre tags."""
        all_recommendations = []
        
        for tag in tags[:3]:  # Limit to 3 tags to avoid too many API calls
            data = self._make_request('tag.gettopartists', {
                'tag': tag,
                'limit': limit
            })
            
            if data and 'topartists' in data:
                artists = data['topartists'].get('artist', [])
                
                for artist_data in artists:
                    all_recommendations.append({
                        'type': 'artist',
                        'name': artist_data['name'],
                        'tag': tag,
                        'rank': int(artist_data.get('@attr', {}).get('rank', 0)),
                        'url': artist_data.get('url', ''),
                        'image': artist_data.get('image', [{}])[-1].get('#text', '') if artist_data.get('image') else ''
                    })
            
            time.sleep(0.2)  # Rate limiting
        
        # Remove duplicates and sort by rank
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec['name'] not in seen:
                seen.add(rec['name'])
                unique_recommendations.append(rec)
        
        return sorted(unique_recommendations, key=lambda x: x['rank'])[:limit]
    
    def get_intersection_based_recommendations(self, 
                                            liked_artists: List[str] = None,
                                            liked_tracks: List[tuple] = None,  # [(artist, track), ...]
                                            genres: List[str] = None,
                                            limit: int = 10,
                                            min_similarity_threshold: float = 0.1) -> Dict[str, List[Dict]]:
        """
        Get recommendations that are similar to ALL of the user's preferences, not just individual ones.
        Uses intersection-based scoring to find artists/tracks that match the overall taste profile.
        
        Args:
            liked_artists: List of artist names
            liked_tracks: List of (artist, track) tuples
            genres: List of genre/tag names
            limit: Number of final recommendations to return
            min_similarity_threshold: Minimum similarity score to consider a match
        
        Returns:
            Dictionary with intersection-based recommendations
        """
        recommendations = {
            'intersection_artists': [],
            'profile_based_tracks': []
        }
        
        # Find artists similar to ALL liked artists
        if liked_artists and len(liked_artists) > 1:
            artist_similarities = self._find_intersection_artists(liked_artists, min_similarity_threshold)
            recommendations['intersection_artists'] = artist_similarities[:limit]
        
        # Find tracks that match the overall profile (both artists and individual tracks)
        if liked_artists or liked_tracks:
            profile_tracks = self._find_profile_matching_tracks(liked_artists, liked_tracks, limit)
            recommendations['profile_based_tracks'] = profile_tracks
        
        return recommendations
    
    def _find_intersection_artists(self, liked_artists: List[str], min_similarity: float) -> List[Dict]:
        """Find artists that are similar to ALL the liked artists."""
        all_similar_artists = {}  # artist_name -> {similarity_scores: [...], data: {...}}
        
        for artist in liked_artists:
            similar_artists = self.get_similar_artists(artist, limit=50)  # Get more to find intersections
            
            for similar_artist in similar_artists:
                name = similar_artist['name']
                score = similar_artist['match_score']
                
                if name not in all_similar_artists:
                    all_similar_artists[name] = {
                        'similarity_scores': [],
                        'data': similar_artist,
                        'appears_in': []
                    }
                
                all_similar_artists[name]['similarity_scores'].append(score)
                all_similar_artists[name]['appears_in'].append(artist)
            
            time.sleep(0.2)
        
        # Calculate intersection score - only keep artists that appear for ALL liked artists
        intersection_artists = []
        required_appearances = len(liked_artists)
        
        for artist_name, artist_info in all_similar_artists.items():
            # Only consider artists that are similar to ALL input artists
            if len(artist_info['similarity_scores']) == required_appearances:
                # Calculate aggregate similarity score
                avg_similarity = sum(artist_info['similarity_scores']) / len(artist_info['similarity_scores'])
                min_similarity_score = min(artist_info['similarity_scores'])
                
                # Use minimum similarity as the final score (most conservative)
                # This ensures the artist is reasonably similar to ALL preferences
                if min_similarity_score >= min_similarity:
                    artist_data = artist_info['data'].copy()
                    artist_data['intersection_score'] = min_similarity_score
                    artist_data['avg_similarity'] = avg_similarity
                    artist_data['similar_to'] = artist_info['appears_in']
                    
                    intersection_artists.append(artist_data)
        
        # Sort by intersection score (minimum similarity to all artists)
        return sorted(intersection_artists, key=lambda x: x['intersection_score'], reverse=True)
    
    def _find_intersection_tracks(self, liked_tracks: List[tuple], min_similarity: float) -> List[Dict]:
        """Find tracks that are similar to ALL the liked tracks."""
        all_similar_tracks = {}  # track_key -> {similarity_scores: [...], data: {...}}
        
        for artist, track in liked_tracks:
            similar_tracks = self.get_similar_tracks(artist, track, limit=50)
            
            for similar_track in similar_tracks:
                track_key = f"{similar_track['artist']}-{similar_track['name']}"
                score = similar_track['match_score']
                
                if track_key not in all_similar_tracks:
                    all_similar_tracks[track_key] = {
                        'similarity_scores': [],
                        'data': similar_track,
                        'appears_in': []
                    }
                
                all_similar_tracks[track_key]['similarity_scores'].append(score)
                all_similar_tracks[track_key]['appears_in'].append(f"{artist} - {track}")
            
            time.sleep(0.2)
        
        # Calculate intersection score
        intersection_tracks = []
        required_appearances = len(liked_tracks)
        
        for track_key, track_info in all_similar_tracks.items():
            if len(track_info['similarity_scores']) == required_appearances:
                min_similarity_score = min(track_info['similarity_scores'])
                avg_similarity = sum(track_info['similarity_scores']) / len(track_info['similarity_scores'])
                
                if min_similarity_score >= min_similarity:
                    track_data = track_info['data'].copy()
                    track_data['intersection_score'] = min_similarity_score
                    track_data['avg_similarity'] = avg_similarity
                    track_data['similar_to'] = track_info['appears_in']
                    
                    intersection_tracks.append(track_data)
        
        return sorted(intersection_tracks, key=lambda x: x['intersection_score'], reverse=True)
    
    def _find_profile_matching_tracks(self, liked_artists: List[str] = None, 
                                    liked_tracks: List[tuple] = None, 
                                    limit: int = 10,
                                    max_tracks_per_artist: int = 2) -> List[Dict]:
        """
        Find tracks that match the overall profile by looking at tracks from similar artists
        and tracks similar to the user's preferences. Ensures variety by limiting tracks per artist.
        """
        all_candidate_tracks = []
        
        # Strategy 1: Get tracks from artists similar to the user's liked artists
        if liked_artists:
            intersection_artists = self._find_intersection_artists(liked_artists, min_similarity=0.1)
            
            for artist_rec in intersection_artists[:8]:  # More artists to get variety
                artist_name = artist_rec['name']
                top_tracks = self.get_top_tracks_by_artist(artist_name, limit=5)
                
                for track in top_tracks:
                    track['profile_score'] = artist_rec['intersection_score']
                    track['source'] = 'intersection_artist'
                    track['source_artist'] = artist_name
                    track['reason'] = f"From {artist_name} (similar to your taste)"
                    all_candidate_tracks.append(track)
                
                time.sleep(0.2)
        
        # Strategy 2: Get tracks similar to each of the liked tracks
        if liked_tracks:
            print("Liked Tracks executed.")
            for artist, track_name in liked_tracks:
                print(artist,track_name)
                similar_tracks = self.get_similar_tracks(artist, track_name, limit=10)
                print(similar_tracks)
                for track in similar_tracks:
                    # Weight the score based on how many similar tracks this appears in
                    print(track['match_score'])
                    track['profile_score'] = track['match_score'] * 0.6  # Slightly lower than intersection artists
                    track['source'] = 'similar_track'
                    track['source_track'] = f"{artist} - {track_name}"
                    track['reason'] = f"Similar to {track_name} by {artist}"
                    all_candidate_tracks.append(track)
                
                time.sleep(0.2)
        
        # Strategy 3: Diversified selection to ensure variety
        return self._diversify_track_selection(all_candidate_tracks, limit, max_tracks_per_artist)
    
    def _diversify_track_selection(self, candidate_tracks: List[Dict], 
                                 limit: int, 
                                 max_tracks_per_artist: int) -> List[Dict]:
        """
        Select tracks ensuring variety - limit tracks per artist and balance different sources.
        """
        # Group tracks by artist
        tracks_by_artist = {}
        for track in candidate_tracks:
            artist = track['artist']
            if artist not in tracks_by_artist:
                tracks_by_artist[artist] = []
            tracks_by_artist[artist].append(track)
        
        # Sort tracks within each artist by profile score
        for artist in tracks_by_artist:
            tracks_by_artist[artist].sort(key=lambda x: x['profile_score'], reverse=True)
        
        # Select tracks with variety
        selected_tracks = []
        artist_track_counts = {}
        
        # Create a sorted list of all tracks
        all_tracks_sorted = sorted(candidate_tracks, key=lambda x: x['profile_score'], reverse=True)
        
        for track in all_tracks_sorted:
            artist = track['artist']
            
            # Check if we can add more tracks from this artist
            current_count = artist_track_counts.get(artist, 0)
            
            if current_count < max_tracks_per_artist:
                # Avoid duplicates
                track_key = f"{track['artist']}-{track['name']}"
                if not any(f"{t['artist']}-{t['name']}" == track_key for t in selected_tracks):
                    selected_tracks.append(track)
                    artist_track_counts[artist] = current_count + 1
                    
                    if len(selected_tracks) >= limit:
                        break
        
        # If we still need more tracks and were too restrictive, add more with higher limits
        if len(selected_tracks) < limit:
            remaining_needed = limit - len(selected_tracks)
            used_tracks = {f"{t['artist']}-{t['name']}" for t in selected_tracks}
            
            for track in all_tracks_sorted:
                track_key = f"{track['artist']}-{track['name']}"
                if track_key not in used_tracks:
                    selected_tracks.append(track)
                    used_tracks.add(track_key)
                    remaining_needed -= 1
                    
                    if remaining_needed <= 0:
                        break
        
        return selected_tracks
    
    def format_intersection_recommendations(self, recommendations: Dict[str, List[Dict]]) -> str:
        """Format intersection-based recommendations in a readable format for the agent system."""
        output = []
        
        if recommendations['intersection_artists']:
            output.append("\n## Artists Similar to Your Overall Taste")
            for i, rec in enumerate(recommendations['intersection_artists'][:5], 1):
                output.append(f"{i}. **{rec['name']}**")
                output.append(f"   - Intersection Score: {rec['intersection_score']:.3f}")
                output.append(f"   - Similar to: {', '.join(rec['similar_to'])}")
                if rec.get('avg_similarity'):
                    output.append(f"   - Average Similarity: {rec['avg_similarity']:.3f}")
        
        if recommendations['profile_based_tracks']:
            output.append("\n## Tracks Matching Your Profile")
            
            # Group by source for better presentation
            artist_tracks = []
            similar_tracks = []
            
            for rec in recommendations['profile_based_tracks'][:10]:
                if rec.get('source') == 'intersection_artist':
                    artist_tracks.append(rec)
                else:
                    similar_tracks.append(rec)
            
            if artist_tracks:
                output.append("\n### From Artists Similar to Your Taste:")
                for i, rec in enumerate(artist_tracks[:5], 1):
                    output.append(f"{i}. **{rec['name']}** by {rec['artist']}")
                    output.append(f"   - Score: {rec.get('profile_score', 0):.3f}")
                    output.append(f"   - {rec.get('reason', 'Matches your taste profile')}")
            
            if similar_tracks:
                output.append("\n### Similar to Your Liked Tracks:")
                for i, rec in enumerate(similar_tracks[:5], 1):
                    output.append(f"{i}. **{rec['name']}** by {rec['artist']}")
                    output.append(f"   - Score: {rec.get('profile_score', 0):.3f}")
                    output.append(f"   - {rec.get('reason', 'Similar to your preferences')}")
        
        if not any(recommendations.values()):
            output.append("\n## No Intersection Found")
            output.append("The artists/tracks you mentioned don't have enough similar artists/tracks in common.")
            output.append("Try providing more diverse preferences or lowering the similarity threshold.")
        
        return '\n'.join(output)


# Example usage for your multi-agent system
class MusicRecommendationAgent:
    """Agent wrapper for the recommendation tool."""
    
    def __init__(self, lastfm_api_key: str):
        self.recommender = LastFMRecommendationTool(lastfm_api_key)
    
    def process_user_input(self, user_message: str, 
                          liked_artists: List[str] = None,
                          liked_tracks: List[tuple] = None) -> str:
        """
        Process user input and return intersection-based music recommendations.
        This is where you'd integrate with your multi-agent system.
        """
        # Get intersection-based recommendations with variety
        recs = self.recommender.get_intersection_based_recommendations(
            liked_artists=liked_artists,
            liked_tracks=liked_tracks,
            limit=12,  # Get more to ensure variety after filtering
            min_similarity_threshold=0.15  # Adjust based on how strict you want to be
        )
        
        return self.recommender.format_intersection_recommendations(recs)


# Usage example:
if __name__ == "__main__":
    # Initialize with your Last.fm API key
    load_dotenv()
    API_KEY = os.environ["LastFM_API_KEY"]
    
    # Create the recommendation tool
    recommender = LastFMRecommendationTool(API_KEY)
    
    # Example 1: Find artists similar to BOTH Kendrick Lamar AND J. Cole
    print("=== Artists Similar to Both Kendrick Lamar and J. Cole ===")
    recommendations = recommender.get_intersection_based_recommendations(
        liked_artists=["Kendrick Lamar", "J. Cole"],
        liked_tracks=[("J Cole","No Role Modelz"), ("The Weeknd", "Blinding Lights")],
        limit=5,
        min_similarity_threshold=0.1
    )
    
    formatted_output = recommender.format_intersection_recommendations(recommendations)
    print(formatted_output)

  
    
    