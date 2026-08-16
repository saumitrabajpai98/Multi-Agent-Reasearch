from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from rich import print
load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query:str) -> str:
    """
    Search the web for recent and reliable information on a given topic. Returns Titles, URLs and snippets.
    """
    tavily_results = tavily.search(query=query, max_results=5)

    out = []

    # print("=====================================================")
    # print(f"Output is: {tavily_results}")
    # print("=====================================================")

    for r in tavily_results['results']:
        out.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n")

    # print("=====================================================")
    # print(f"Output is: {out}")
    # print("=====================================================")

    return "\n-----\n".join(out)

# print(web_search.invoke("What are recent news about war in Israel?"))

@tool
def web_scrape(url:str) -> str:
    """Scrape and return the clean text content from a given URL for a deeper reading."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"}) #If beautiful soup send the reuqest without headers, it will be blocked by some websites. So we add a user-agent header to mimic a real browser request.
        soup = BeautifulSoup(resp.content, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape the URL: {str(e)}"