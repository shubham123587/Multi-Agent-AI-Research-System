from agents import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain
)


def run_research_pipeline(topic: str) -> dict:

    state = {}

    # =========================================================
    # STEP 1 - SEARCH AGENT
    # =========================================================

    print("\n" + "=" * 60)
    print("STEP 1 - Search Agent is working...")
    print("=" * 60)

    search_agent = build_search_agent()

    search_result = search_agent.invoke({
        "messages": [
            (
                "user",
                f"""Find recent, reliable and detailed information about:

{topic}

Use the available web search tool to find relevant sources.
Return the important findings along with the URLs."""
            )
        ]
    })

    state["search_results"] = search_result["messages"][-1].content

    print("\nSEARCH RESULTS:\n")
    print(state["search_results"])


    # =========================================================
    # STEP 2 - READER AGENT
    # =========================================================

    print("\n" + "=" * 60)
    print("STEP 2 - Reader Agent is scraping top resources...")
    print("=" * 60)

    reader_agent = build_reader_agent()

    reader_result = reader_agent.invoke({
        "messages": [
            (
                "user",
                f"""Based on the following search results about:

{topic}

Select the most relevant and reliable URL and scrape it
using the available scraping tool.

Search Results:
{state["search_results"][:8000]}

Return the useful detailed information extracted from
the selected source."""
            )
        ]
    })

    state["scraped_content"] = reader_result["messages"][-1].content

    print("\nSCRAPED CONTENT:\n")
    print(state["scraped_content"])


    # =========================================================
    # STEP 3 - WRITER
    # =========================================================

    print("\n" + "=" * 60)
    print("STEP 3 - Writer is drafting the report...")
    print("=" * 60)

    research_combined = (
        f"SEARCH RESULTS:\n"
        f"{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n"
        f"{state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    print("\nFINAL REPORT:\n")
    print(state["report"])


    # =========================================================
    # STEP 4 - CRITIC
    # =========================================================

    print("\n" + "=" * 60)
    print("STEP 4 - Critic is reviewing the report...")
    print("=" * 60)

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })

    print("\nCRITIC REPORT:\n")
    print(state["feedback"])


    # =========================================================
    # RETURN COMPLETE PIPELINE STATE
    # =========================================================

    return state


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    topic = input("\nEnter a research topic: ")

    if topic.strip():

        result = run_research_pipeline(topic)

    else:

        print("\nPlease enter a valid research topic.")