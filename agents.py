from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from tools import web_scrape, web_search
import model_selection


# 1st agent with create_agent() instead of create_react_agent() (legacy)
# with create_react_agent(), it was a bit more complex to set up the prompt template and output parser, but with create_agent(), it is simpler and more straightforward. The agent will use the tools provided to perform web search and web scraping tasks. Parsing the output also becomes easier with this approach, as we can directly use the StrOutputParser to handle the output from the tools. This makes it easier to extract relevant information and present it in a user-friendly format.
def build_search_agent():
    print(f"{model_selection.selected_model} is the openai model used for creating the final report.")
    return create_agent(
        model=model_selection.response_llm,
        tools=[web_search]
    )

def build_reader_agent():
    return create_agent(
        model=model_selection.response_llm,
        tools=[web_scrape]
    )

# writer chain
# I didn't write the prompts above because I want my agents to do research freely and not be constrained by a specific prompt. I want them to be able to search for information and read content from the web without being limited by a predefined prompt. This allows for more flexibility and adaptability in their research capabilities and then I will use the output from the agents to feed into a writer chain that will generate a final report or summary based on the information gathered by the agents. This way, the agents can focus on gathering and processing information, while the writer chain can focus on synthesizing and presenting that information in a coherent and structured manner.

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. With clear structure and insightful reports."),
    ("human", """ Write a detailed research on the topic below.

    Topic: {topic}

    Research gathered
    {research}

    Structure the report as:
    - Introduction
    - Key findings (minimum 3 well-explained points)
    - Conclusion
    - Sources (list the sources used in the research, with URLs)

    Be detailed, factual, and professional. Use the research provided to support your points, and ensure that the report is well-organized and easy to read. Avoid plagiarism and ensure that all sources are properly cited.
    """)
])

critic_prompt = ChatPromptTemplate.from_messages([
    "system", "You are a sharp and constructive critic. Be honest and specific.",
    "human", """Review the research report below and eavluate it stritcly.
    Report: {report}

    Respond in this exact format:

    Score: X/10

    Strengths:
    - ...
    - ...

    Areas to Improve:
    - ...
    - ...

    One line verdict:
    ...
    """
])