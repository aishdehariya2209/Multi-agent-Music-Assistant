from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_tavily import TavilySearch
from lyricsgenius import Genius
from pydantic import Field, BaseModel
from dotenv import load_dotenv
from typing import Annotated
from langgraph.types import Command, Send
from langgraph.prebuilt import InjectedState
from sentiment_analysis import RedditSentimentAnalyzer
from music_recommendation import LastFMRecommendationTool
import asyncio
import os

# Environment setup
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
)

# Enhanced pretty print functions
from langchain_core.messages import convert_to_messages

def pretty_print_message(message, indent=False):
    if message is None:
        return
    
    # Handle different message types
    if hasattr(message, 'content'):
        content = message.content
        if isinstance(content, list):
            # Handle content that might be a list (tool calls, etc.)
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    print(f"{'  ' if indent else ''}{item['text']}")
                elif hasattr(item, 'text'):
                    print(f"{'  ' if indent else ''}{item.text}")
                else:
                    print(f"{'  ' if indent else ''}{str(item)}")
        else:
            print(f"{'  ' if indent else ''}{content}")
    else:
        pretty_message = message.pretty_repr(html=True) if hasattr(message, 'pretty_repr') else str(message)
        if indent:
            indented = "\n".join("  " + line for line in pretty_message.split("\n"))
            print(indented)
        else:
            print(pretty_message)

def pretty_print_messages(update, last_message=False):
    # Handle None or empty updates
    if update is None:
        return
    
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return
        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print()
        is_subgraph = True

    # Handle case where update is None after tuple unpacking
    if update is None:
        return

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "  " + update_label
        print(update_label)
        print()

        # Check if node_update has messages and they're not None
        if node_update is None:
            print("  No messages to display" if is_subgraph else "No messages to display")
            print()
            continue
            
        # Handle different types of node updates
        messages = None
        if isinstance(node_update, dict) and "messages" in node_update:
            messages = node_update["messages"]
        elif hasattr(node_update, 'messages'):
            messages = node_update.messages
        else:
            # If it's not a dict with messages, treat the whole thing as a message
            messages = [node_update] if node_update else None

        if messages is None:
            print("  No messages to display" if is_subgraph else "No messages to display")
            print()
            continue

        # Convert to messages if needed
        try:
            messages = convert_to_messages(messages) if not isinstance(messages, list) else messages
        except:
            # If conversion fails, work with what we have
            pass
            
        if messages is None:
            print("  No messages to display" if is_subgraph else "No messages to display")
            print()
            continue
            
        if last_message and len(messages) > 0:
            messages = messages[-1:]

        for m in messages:
            if m is not None:
                pretty_print_message(m, indent=is_subgraph)
        print()

# TOOLS
# Web search Tool
web_search = TavilySearch(max_results=3)

# Wikipedia search Tool
class WikiInputs(BaseModel):
    """Inputs to the wikipedia tool."""
    query: str = Field(
        description="query to look up in Wikipedia, should be 3 or less words"
    )

api_wrapper = WikipediaAPIWrapper(top_k_results=3)
wikipedia_search = WikipediaQueryRun(
    name="wiki-tool",
    description="look up things in wikipedia",
    args_schema=WikiInputs,
    api_wrapper=api_wrapper,
    return_direct=False,
)

# Tool for finding lyrics on songs from genius
class GeniusInput(BaseModel):
    """Inputs to the song search tool"""
    song_name: str = Field(description="Enter the exact title of the song you want to look up")

@tool("song-search-tool", args_schema=GeniusInput, return_direct=False)
def search_song(song_name):
    """Search for lyrics and other info about songs"""
    GENIUS_API_KEY = os.environ["GENIUS_API_KEY"]
    genius = Genius(GENIUS_API_KEY)
    song = genius.search_song(song_name)
    if song:
        print("-----------Song Found---------")
        song_id = song.id
        lyrics = song.lyrics
        title = song.title_with_featured
        artist = song.artist
        search_output = {"lyrics": lyrics, "artist": artist, "title": title, "song_id": song_id}
        return search_output
    else:
        return "song name not found on site, try using a different song name"
#Tool for sentiment analysis agent
class SentimentInput(BaseModel):
    """Inputs to the sentiment analysis tool"""
    artist_name: str = Field(description="Enter the name of the artist who you would like insights on")

@tool("sentiment-analysis-tool",args_schema=SentimentInput,return_direct = False)
def analyse_sentiments(artist_name):
    print("Tool called by sentiment Agent")
    REDDIT_CLIENT = os.eviron["REDDIT_CLIENT_ID"]
    REDDIT_SECRET = os.environ["REDDIT_SECRET"]
    reddit_config = {
        'client_id': REDDIT_CLIENT,
        'client_secret': REDDIT_SECRET,
        'user_agent': 'SentimentAnalysisBot/1.0'
    }
    
    sentiment_llm = llm
    
    async def execute_sentiment_analysis(artist_name,reddit_config,llm):
        analyzer = RedditSentimentAnalyzer(reddit_config, llm)
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
    
    result = asyncio.run(execute_sentiment_analysis(artist_name, reddit_config, sentiment_llm))
    return result 
class RecommendationsInput(BaseModel):
    pass

@tool("get-recommendations-tool",args_schema = RecommendationsInput , return_direct = False)
def get_recommendations(liked_songs , liked_artists , limit):
    "This tool gives song recommendations"
    # Initialize with your Last.fm API key
    API_KEY = os.environ["LastFM_API_KEY"]
    
    # Create the recommendation tool
    recommender = LastFMRecommendationTool(API_KEY)
    recommendations = recommender.get_intersection_based_recommendations(
        liked_artists=liked_artists,
        liked_tracks=liked_tracks,
        limit=max(3,min(10,limit)),
        min_similarity_threshold=0.1
    )

    formatted_output = recommender.format_intersection_recommendations(recommendations)

    return formatted_output



# AGENTS
musicinfo_agent = create_react_agent(
    model=llm,
    tools=[web_search, wikipedia_search],
    prompt=(
        "You are Music Info agent.\n\n"
        "INSTRUCTIONS:\n"
        "Assist with tasks that require researching biographical or historical data about an artist or songs\n"
        "If you want to answer a more general query related to an artists discography or life story, you can use your wikipedia search tool\n"
        "If your query includes multiple questions, break it into parts and pass each of them individually while using the wikipedia search tool for better results.\n"
        "If you have to answer queries which require more information, feel free to use the web search tool.\n"
        "Provide comprehensive and detailed information based on your research.\n"
        "Format your response clearly and include all relevant details you found."
    ),
    name="musicinfo_agent"
)

lyrics_agent = create_react_agent(
    model=llm,
    tools=[web_search, search_song],
    prompt=(
        "You are a song lyrics agent.\n\n"
        "INSTRUCTIONS:\n"
        "ONLY Assist with tasks that require retrieving lyrics and other information of a song.\n"
        "If you are not confident about the song_name provided by the user, use the web search tool to figure out the correct name of the song and then use the search song_tool.\n"
        "Use the search_song tool to search for a song and retrieve lyrics of that song and other relevant data.\n"
        "If the search song tool doesn't result a song then try searching the web for the correct name of the song, THEN USE THE SEARCH SONG TOOL AGAIN WITHOUT FAIL.\n"
        "If you still can't find the song, try searching the web for it specifically.\n"
        "Return the lyrics, title, artist of the song in a clear format.\n"
        "If specific sections are requested (like chorus, verse, bridge), extract and return only those sections.\n"
        "Always format the lyrics clearly with proper line breaks and structure."
    ),
    name="lyrics_agent"
)

sentiment_agent = create_react_agent(
   model = llm,
   tools = [analyse_sentiments],
   prompt = (
       "You are sentiment analysis agent.\n\n"
       "INSTRUCTIONS:\n"
       "When asked for analysis / insights about an artist use the analyse_sentiments tool.\n\n"
       "After getting the sentiment data from the tool , craft an engaging analysis into the sentiment data and try to find interesting insights"
       "Focus on what makes THIS artist's sentiment profile unique - don't follow a rigid template. Consider highlighting:\n"
       "- Surprising sentiment patterns across platforms\n"
       "- What audiences actually talk about most regarding this artist\n" 
       "- Notable positive/negative comment examples that capture the essence of fan opinions\n"
       "- Emerging trends or shifts in perception\n"
       "- Platform-specific audience differences that tell a story\n\n"
       "Always include key data points (total comments analyzed, data sources, confidence level) but weave them naturally into your narrative. Your goal is to paint a vivid picture of how the public sees this artist, emphasizing the most compelling insights from the data rather than covering every metric.\n\n"
       "Be analytical but conversational, and let the data guide you toward the most interesting story to tell about this artist's public perception."
   ),
   name = "sentiment_agent"
)

recommendation_agent = create_react_agent(
    model = llm,
    tools = [get_recommendations],
    prompt = (
        "You are music recommendation agent.\n\n"
        "INSTRUCTIONS:\n"
        "When asked for a recommendation"
    ),
    name = "recommendation_agent"   
)


# HANDOFF TOOLS
def create_task_handoff_tool(*, agent_name: str, description: str | None = None):
    """Creates a handoff tool that passes specific curated tasks to agents"""
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        task_description: Annotated[
            str,
            "Description of what the next agent should do, including all of the relevant context.",
        ],
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_description_message = {"role": "user", "content": task_description}
        agent_input = {**state, "messages": [task_description_message]}
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool

# Create handoff tools
transfer_to_musicinfo = create_task_handoff_tool(
    agent_name="musicinfo_agent",
    description="Transfer biographical, historical, or general information tasks about artists or songs to the music info agent."
)

transfer_to_lyrics = create_task_handoff_tool(
    agent_name="lyrics_agent", 
    description="Transfer tasks that require finding song lyrics or specific song information to the lyrics agent."
)

transfer_to_sentiments = create_task_handoff_tool(
    agent_name = "sentiment_agent",
    description = "Transfer tasks that require analysing sentiments towards an artist and giving insights to the sentiment agent"
)

# FIXED: Proper answer query tool that works as both a normal tool and with Command/Send
@tool("answer_query", return_direct=True)
def transfer_to_synthesizer(
    query_summary: Annotated[str, "Brief summary of what information has been gathered"],
    state: Annotated[MessagesState, InjectedState],
) -> Command:
    """
    This tool will synthesize and answer the users query based on the information gathered, Feel free to call this when you think you have gathered enough information to answer the query.
    """
    print("*"*100)
    print("Query is about to be answered!")
    
    # Extract the original user query
    original_query = ""
    if state.get("messages") and len(state["messages"]) > 0:
        first_message = state["messages"][0]
        if hasattr(first_message, 'content'):
            original_query = first_message.content
        elif isinstance(first_message, dict) and 'content' in first_message:
            original_query = first_message['content']
        else:
            original_query = str(first_message)
    
    # Collect all the information gathered from the conversation
    conversation_context = ""
    for message in state.get("messages", []):
        if hasattr(message, 'content'):
            content = message.content
        elif isinstance(message, dict) and 'content' in message:
            content = message['content']
        else:
            content = str(message)
        
        # Skip system messages and tool calls, focus on actual information
        if isinstance(message, dict):
            role = message.get('role', '')
            if role in ['assistant', 'tool'] and content:
                conversation_context += content + "\n\n"
        elif hasattr(message, 'content') and message.content:
            conversation_context += content + "\n\n"
    
    # Create the synthesis prompt
    synthesis_prompt = f"""
You are tasked with providing a comprehensive final answer to the user's query using ONLY the information that has been gathered.

ORIGINAL USER QUERY: {original_query}

GATHERED INFORMATION:
{conversation_context}

QUERY SUMMARY: {query_summary}

CRITICAL INSTRUCTIONS:
1. Use ONLY the information provided above from the conversation context
2. If lyrics were found, reproduce them EXACTLY as provided - do not truncate or summarize
3. If artist information was found, include all the biographical/historical details that were gathered
4. If sentiment analysis report was found , then include it in a near identical way in your answer.
5. Format your response clearly with proper structure and headings
6.. Do NOT add any information that wasn't explicitly provided in the conversation
7. If multiple questions were asked, address each one using the gathered information
8. Provide a complete, comprehensive answer - don't cut anything short

FORMAT GUIDELINES:
- Use clear headings (## Artist Name - Song Title)
- For lyrics: Include ALL verses, choruses, bridges, outros with proper formatting
- Separate different sections clearly: [Intro], [Verse 1], [Chorus], etc.
- Maintain original line breaks and structure
- Include all metadata (artist, title, featured artists, etc.)

Please provide your final comprehensive answer:
"""
    
    # Use the LLM to synthesize the final answer
    try:
        response = llm.invoke([{"role": "user", "content": synthesis_prompt}])
        
        # Extract content from response
        if hasattr(response, 'content'):
            final_answer = response.content
        else:
            final_answer = str(response)
        
        print(final_answer)
        
        # Return a Command that goes to END with the final answer
        return Command(
            goto=END,
            update={"messages": [{"role": "assistant", "content": final_answer}]}
        )
        
    except Exception as e:
        error_message = f"Error generating final response: {str(e)}\n\nRaw information gathered:\n{conversation_context}"
        print(error_message)
        
# SUPERVISOR AGENT - Enhanced with better decision making
supervisor_agent = create_react_agent(
    model=llm,
    tools=[transfer_to_musicinfo, transfer_to_lyrics,transfer_to_sentiments, transfer_to_synthesizer],
    prompt=(
        "You are a supervisor. Follow these EXACT steps:\n\n"
        "AVAILABLE AGENTS:\n"
        "1. MUSIC INFO AGENT: Handles biographical, historical, and general music information\n"
        "   - Artist biographies and careers\n"
        "   - Music history and context\n"
        "   - Discographies and albums\n"
        "   - Industry information\n\n"
        "2. LYRICS AGENT: Handles song-specific information and complete lyrical content\n"
        "   - Song identification and metadata\n"
        "   - COMPLETE FULL LYRICS retrieval\n"
        "   - Song structure information\n\n"
        "3. SENTIMENT AGENT: Handles providing insights and analysing sentiments of audience towards an artist\n"
        "   -Sentiment analysis into artists\n"
        "   -Interesting insights and temporal trends\n" 
        "4.RECOMMENDATION AGENT: Handles providing recommendations or suggestions on which songs/artists to listen to based on user preferences"
        "   -Suggests Artists and songs"
        "   -Tailored to user preferences"
        "CO-ORDINATION STRATEGY"
        "1. Look at the user's original query\n"
        "2. If it's about lyrics and no lyrics have been provided yet → use transfer_to_lyrics\n"
        "3. If it's about artist info and no info has been provided yet → use transfer_to_musicinfo\n"
        "4. If its about insights/analysis into an artist and no analysis has been provided yet → use transfer_to_sentiments\n"
        "5. If you are asked for information that would require help from more than one agent then use the agents in order of percieved context of the query.\n\n"
        "CRITICAL : Once you have gathered enough information to answer the query use the transfer_to_synthesizer tool"
        "Simple decision tree:\n"
        "- User wants lyrics + no lyrics provided = transfer_to_lyrics\n"
        "- User wants artist info + no info provided = transfer_to_musicinfo  \n"
        "- User wants insights or sentiment analysis + no analysis provided = transfer_to_sentiments\n"
        "- Agent has provided information = transfer_to_synthesizer\n\n"
        "IMPORTANT:"
        "- If you feel like stopping , do not STOP call the trasfer_to_synthesizer tool instead"
        "Always explain your decision briefly before taking action."
    ),
    name="supervisor_agent"
)

# GRAPH CONSTRUCTION
def create_music_assistant():
    """Creates music assistant with proper END handling"""
    workflow = StateGraph(MessagesState)
    
    # Add all nodes
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("musicinfo_agent", musicinfo_agent) 
    workflow.add_node("lyrics_agent", lyrics_agent)
    workflow.add_node("sentiment_agent",sentiment_agent)
    
    # Define the flow - simple edges back to supervisor
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("musicinfo_agent", "supervisor")
    workflow.add_edge("lyrics_agent", "supervisor")
    workflow.add_edge("sentiment_agent","supervisor")
    
    # The answer_query tool will handle routing to END via Command
    
    return workflow.compile()

# Create the compiled graph
music_assistant_graph = create_music_assistant()

# MAIN EXECUTION
user_input = input("Enter your music query: ")

print(f"\n🎵 Processing your query: {user_input}\n")
print("="*60)

# Stream from the compiled graph
try:
    for chunk in music_assistant_graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        subgraphs = True
    ):
        print(f"Chunk type: {type(chunk)}")
        print(f"Chunk content: {chunk}")
        pretty_print_messages(chunk)
except Exception as e:
    print(f"Error during execution: {e}")
    import traceback
    traceback.print_exc()
