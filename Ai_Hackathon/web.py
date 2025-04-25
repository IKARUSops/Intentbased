import os
from langchain_community.tools.tavily_search import TavilySearchResults
tavily_api_key = os.getenv("TAVILY_API_KEY")
web_search_tool = TavilySearchResults(k=3,tavily_api_key=tavily_api_key)