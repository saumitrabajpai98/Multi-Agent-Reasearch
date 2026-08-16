from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

def run_research_pipeline(topic: str) -> dict:
    # since we are using state to store the o/p of agents, here also we'll use the state
    
    state = {}

    #Step 1 Search agent
    print("\n"+ " ="*50)
    print("Step 1 - Search Agent is working...")
    print("\n"+ " ="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information on the topic: {topic}")],
    })

    state["search_result"] = search_result['messages'][-1].content

    print("\n Search Result: ", state['search_result'])

    #Step 2 Reader agent
    print("\n"+ " ="*50)
    print("Step 2 - Reader Agent is working...")
    print("\n"+ " ="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user", 
                      f"Based on the following search results about '{topic}',"
                      f"pick the most relevant URL and scrape it for deeper content \n\n"
                      f"Search Results:\n{state['search_result'][:800]}"
                      )]
    })

    state["scraped_content"] = reader_result['messages'][-1].content

    print("\nScraped Content:\n", state['scraped_content'])

    #Step 3 Writer Chain
    print("\n"+ " ="*50)
    print("Step 1 - Writer Agent is drafting response...")
    print("\n"+ " ="*50)

    research_combined = (f"Search Results: \n {state['scraped_content']} \n\n"
                         f"Detailed Scraped Content: \n {state['scraped_content']}")

    state['report'] = writer_chain.invoke({
        "topic": topic,
        "research":research_combined
    })

    print("\nFinal Report:\n", state['report'])

    #critic report
    #Step 2 Reader agent
    print("\n"+ " ="*50)
    print("Step 4 - Critic is reviewing the report...")
    print("\n"+ " ="*50)

    state['feedback'] = critic_chain.invoke({
        "report": state['report']})

    print("\nCritic Report \n", state["feedback"])

    return state


if __name__ == "__main__":
    topic = input("\nEnter a topic for research: ")
    run_research_pipeline(topic)
