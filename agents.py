from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()


# ==========================================
# MODEL SETUP - GOOGLE GEMINI
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)


# ==========================================
# 1st AGENT - SEARCH AGENT
# ==========================================

def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )


# ==========================================
# 2nd AGENT - READER AGENT
# ==========================================

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )


# ==========================================
# WRITER CHAIN
# ==========================================

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert research writer.

Your job is to create clear, structured, factual,
and insightful research reports.

Use only the information provided in the research.
Do not invent facts or sources."""
    ),

    (
        "human",
        """Write a detailed research report on the topic below.

Topic:
{topic}

Research Gathered:
{research}

Structure the report as:

# Introduction

Provide a brief introduction to the topic.

# Key Findings

Provide at least 3 well-explained findings.

# Conclusion

Summarize the most important insights.

# Sources

List all URLs found in the research.

Be detailed, factual, professional, and easy to understand."""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# ==========================================
# CRITIC CHAIN
# ==========================================

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a sharp and constructive research critic.

Evaluate the report objectively.
Be honest, specific, and helpful.
Focus on factual quality, completeness,
clarity, structure, and sources."""
    ),

    (
        "human",
        """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in exactly this format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()