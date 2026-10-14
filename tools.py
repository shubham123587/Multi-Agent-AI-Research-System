
from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# =========================================================
# TAVILY API KEY CONFIGURATION
# =========================================================

try:
    TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")
except Exception:
    TAVILY_API_KEY = None

# Fall back to the local .env file
if not TAVILY_API_KEY:
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError(
        "TAVILY_API_KEY is missing. Add it to your local .env "
        "file or configure it in Streamlit Cloud Secrets."
    )

tavily = TavilyClient(api_key=TAVILY_API_KEY)


# =========================================================
# WEB SEARCH TOOL
# =========================================================

@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information
    about a given topic.

    Returns titles, URLs and snippets.
    """
    try:
        results = tavily.search(
            query=query,
            max_results=5
        )

        output = []

        for result in results.get("results", []):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            content = result.get("content", "")

            output.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Snippet: {content[:500]}\n"
            )

        if not output:
            return "No search results were found."

        return "\n---\n".join(output)

    except Exception as e:
        return f"Web search failed: {str(e)}"


# =========================================================
# URL SCRAPER TOOL
# =========================================================

@tool
def scrape_url(url: str) -> str:
    """
    Scrape a webpage and return clean text content
    for deeper research.
    """
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/154.0 Safari/537.36"
                )
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unnecessary HTML elements
        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]):
            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        # Clean excessive spaces
        text = " ".join(text.split())

        return text[:8000]

    except requests.exceptions.RequestException as e:
        return f"Could not access URL: {str(e)}"

    except Exception as e:
        return f"Could not scrape URL: {str(e)}"
