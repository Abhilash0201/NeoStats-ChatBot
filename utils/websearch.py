from typing import List
from tavily import TavilyClient
from config.config import TAVILY_API_KEY

def web_search(query: str, max_results: int = 5) -> List[dict]:
    """Return a list of {'title','url','content'} from Tavily."""
    client = TavilyClient(api_key=TAVILY_API_KEY or None)
    res = client.search(query=query, max_results=max_results)
    # Normalize
    out = []
    for item in res.get("results", []):
        out.append({
            "title": item.get("title",""),
            "url": item.get("url",""),
            "content": item.get("content","")
        })
    return out

def summarized_snippets(query: str, max_results: int = 5) -> str:
    items = web_search(query, max_results=max_results)
    if not items:
        return ""
    return "\n\n".join([f"- {it['title']} — {it['url']}\n  {it['content'][:300]}..." for it in items])
