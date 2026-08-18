llm_model_list = []


llm_model1 = {"model_name":"gpt-5.4-mini","selected":False}
llm_model2 = {"model_name":"gpt-5.4-mini","selected":False} # use 'gpt-5.4-nano' or any other openai if the credits are present, for free tier users, the 'gpt-5.4-mini' is the only model available for use.


llm_model_list.append(llm_model1)
llm_model_list.append(llm_model2)

selected_model = ""
not_selected_model = ""

response_llm = ""
critic_llm = ""