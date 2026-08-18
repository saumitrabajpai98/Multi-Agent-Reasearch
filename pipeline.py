from agents import build_reader_agent, build_search_agent
from initialise import initialise_critic_chain,initialise_writer_chain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import model_selection

load_dotenv()

def run_research_pipeline(topic: str) -> dict:
    # since we are using state to store the o/p of agents, here also we'll use the state
    model_selection.response_llm = ChatOpenAI(model_name=model_selection.selected_model, temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    model_selection.critic_llm = ChatOpenAI(model_name=model_selection.not_selected_model, temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
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
    print("Step 3 - Writer Agent is drafting response...")
    print("\n"+ " ="*50)

    research_combined = (f"Search Results: \n {state['scraped_content']} \n\n"
                         f"Detailed Scraped Content: \n {state['scraped_content']}")

    writer_chain = initialise_writer_chain()

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

    critic_chain = initialise_critic_chain()

    state['feedback'] = critic_chain.invoke({
        "report": state['report']})

    print("\nCritic Report \n", state["feedback"])

    return state


if __name__ == "__main__":
    topic = input("\nEnter a topic for research: ")
    response_model = int(input("Choose the model to create the report: [1]: Write 1 for `gpt-5.4-mini` , [2]: Write 2 for `gpt-5.4-nano`: "))
    for i in range(len(model_selection.llm_model_list)):
        if i+1 == response_model:
            model_selection.selected_model = model_selection.llm_model_list[i]["model_name"]
            model_selection.llm_model_list[i]["selected"] = True
            break
    model_selection.not_selected_model = list(filter(lambda x:x["selected"] == False,model_selection.llm_model_list))[0]["model_name"]
    print("Selected Model: ",model_selection.selected_model)
    print("Not Selected Model: ",model_selection.not_selected_model)
    run_research_pipeline(topic)
