from langchain_core.output_parsers import StrOutputParser
from agents import writer_prompt,critic_prompt
import model_selection

def initialise_writer_chain():
    print("Initialising the writer chain method: ",model_selection.selected_model) 
    writer_chain = writer_prompt | model_selection.response_llm | StrOutputParser()
    return writer_chain

def initialise_critic_chain():
    print("Initialising the critic chain method: ",model_selection.not_selected_model)
    critic_chain = critic_prompt | model_selection.critic_llm | StrOutputParser()
    return critic_chain