import openai
import random
import numpy as np
import pickle
import string
import re
import os
import google.generativeai as genai


def generate_response_from_gemini(system_prompt=None, user_prompt=None, messages_history=None, deterministic=False):
	generation_config = {
	"temperature": 1,
	"top_p": 0.95,
	"top_k": 64,
	"max_output_tokens": 8192,
	"response_mime_type": "text/plain",
	}
	
	model = genai.GenerativeModel(
	model_name = "gemini-1.5-pro",
	generation_config=generation_config,
	system_instruction=system_prompt,
	)

	if messages_history == None:
		messages_history = [{
						"role": "user",
						"parts": user_prompt
					},]
		chat_session = model.start_chat(
		history=[])
	else:
		chat_session = model.start_chat(
		history=messages_history)

	print(f"\t\tModel: gemini-1.5-pro || Deterministic?: {deterministic}")
	response = chat_session.send_message(f"{user_prompt}")
	actual_response = response.text.replace("Output: ", "")
	messages_history.append({
								"role": "model",
						  		"parts":actual_response})
	
	# print(f"chat_session:\n{chat_session}")
	# print(f"chat_session history:\n")
	# for message in chat_session.history:
	# 	print(f"{message}")

	return actual_response, messages_history

def generate_response_from_gpt(system_prompt=None, user_prompt=None, messages_history=None, deterministic=False):
	gpt_model = "gpt-4" # gpt-4 or gpt-4o or gemini
	if messages_history == None:
		messages_history = [
					{
						"role": "system",
						"content": system_prompt
					},
					{
						"role": "user",
						"content":  user_prompt
					}
					]

	print(f"\t\tModel: {gpt_model} || Deterministic?: {deterministic}")
	if deterministic == True:
		completion = openai.ChatCompletion.create(
			model=gpt_model,
			messages=messages_history,
			temperature=0,  # use deterministic greedy token decoding during evaluation experiments however due deterministic decoding is not guaranteed, hence peform multiple runs and get mode of the responses
			max_tokens=2000,
		)
	else:
		completion = openai.ChatCompletion.create(
			model=gpt_model,
			messages=messages_history,
			temperature=0.3,
			top_p=1,
			max_tokens=2000,
		)
	
	response = completion.choices[0].message.content
	messages_history.append({
								"role": "assistant",
								"content":response})
	return response, messages_history

def generate_response_from_llm(llm_api, system_prompt=None, user_prompt=None, messages_history=None, deterministic=False):
	# openai or googleai
	print(f"\t\tUsing: {llm_api} API")

	if llm_api == "openai":
		response, messages_history = generate_response_from_gpt(system_prompt, user_prompt, messages_history, deterministic)

	elif llm_api == "googleai":
		response, messages_history = generate_response_from_gemini(system_prompt, user_prompt, messages_history)

	return response, messages_history

def generate_incontext_learning_prompt(in_context_learning_tasks,k, strategy=None ,query=None,lang2embedding=None,lang2ltl=None, enable_prints=True):
	ret = ''

	if 'random_embedding' in strategy : #select random examples to put into prompt
		if enable_prints==True: print("Randomly selecting in-context learning examples")
		chosen_indices = random.sample(range(0, len(in_context_learning_tasks)), k)
		for idx in chosen_indices:
			ret += f"Input: {in_context_learning_tasks[idx][0]}\nOutput: {in_context_learning_tasks[idx][1]}\n"

	elif 'similar_embedding' in strategy:
		if enable_prints==True: print("Selecting in-context learning examples based on embedding similarity")

		#generate embedding for query
		query_embedding = openai.Embedding.create(input=query, model="text-embedding-ada-002")["data"][0]["embedding"]

		#get the thop k most similar embeddings in lang2embedding along with their corresponding languages
		lang2similarity = {}
		for lang, embedding in lang2embedding.items():
			lang2similarity[lang] = np.dot(query_embedding, embedding)
		#get the top k most similar languages and their associated similarities
		top_k_langs = sorted(lang2similarity, key=lang2similarity.get, reverse=True)[:k]
		top_k_similarities = [lang2similarity[lang] for lang in top_k_langs]

		#generate the prompt
		for lang in top_k_langs:
			ret += f"Input: {lang}\nOutput: {lang2ltl[lang]}\n"
	return(ret)


def query_llm(llm_api, query_task, in_context_learning_tasks, k_num=None, strategy=None, lang2embedding_path=None, lang2ltl_path=None, enable_prints=True):
	# Get the directory of the current file
	current_file_dir = os.path.dirname(os.path.abspath(__file__))

	# Construct the path to the prompts folder
	prompts_folder = os.path.join(current_file_dir, 'prompts')

	if enable_prints==True: print(f"Strategy: {strategy}\n")

	if "similar_embedding" in strategy:
		#set up similar embedding incontext learning
		lang2embedding = pickle.load(open(lang2embedding_path, "rb"))
		lang2ltl = pickle.load(open(lang2ltl_path, "rb"))
	elif "random_embedding" in strategy:
		#set up random incontext learning
		lang2embedding = None
		lang2ltl = None

	if strategy == "single_stage":
		spatial_signature_systemprompt = open(os.path.join(prompts_folder, 'singlestage_limp_prompt.txt'), "r").read()
		spatial_signature_prompt_meat = f'''
		Input_instruction: {query_task}
		Output:'''
		spatial_signature_userprompt = spatial_signature_systemprompt + spatial_signature_prompt_meat
		if enable_prints==True: print("Spatial Predicate Lifted LTL Prompt")
		if enable_prints==True: print(f"Spatial prompt: \n {spatial_signature_userprompt}")
		response, llm_history = generate_response_from_llm(llm_api, spatial_signature_systemprompt,spatial_signature_prompt_meat)
		if enable_prints==True: print(f"response:{response}")
		return response, llm_history

	elif "two_stage" in strategy:
		#preprompt w/ distance ignoring
		if enable_prints==True: print("\nStage 1")
		system_prompt = f"You are a LLM that understands operators involved with Linear Temporal Logic (LTL), such as F, G, U, &, |, ~ , etc. Your goal is translate language input to LTL output."
		in_context_learning = generate_incontext_learning_prompt(in_context_learning_tasks,k=k_num,strategy=strategy,query=query_task,lang2embedding=lang2embedding,lang2ltl=lang2ltl,enable_prints=enable_prints)
		prompt_meat = f"Input: {query_task}\n"
		postprompt = "Output: "
		user_prompt =  in_context_learning + prompt_meat + postprompt
		if enable_prints==True: print(f"prompt: {system_prompt+user_prompt}")
		response,llm_history = generate_response_from_llm(llm_api, system_prompt,user_prompt)
		if enable_prints==True: print(f"response:{response}")

		if enable_prints==True: print("\nStage 2")
		spatial_signature_systemprompt = spatial_signature_systemprompt = open(os.path.join(prompts_folder, 'twostage_limp_prompt.txt'), "r").read()
		spatial_signature_prompt_meat = f'''
		Input_instruction: {query_task}
		Input_ltl: {response}
		Output:'''
		spatial_signature_userprompt = spatial_signature_systemprompt + spatial_signature_prompt_meat
		if enable_prints==True: print("Spatial Predicate Lifted LTL Prompt")
		if enable_prints==True: print(f"Spatial prompt: \n {spatial_signature_userprompt}")
		response,llm_history= generate_response_from_llm(llm_api, spatial_signature_systemprompt, spatial_signature_prompt_meat)
		if enable_prints==True: print(f"response:{response}")
		return response, llm_history


def parse_spatial_lifted_ltl(response):
	# Define the regular expression pattern to find the lifted predicates
    pattern = re.compile(r'(near|pick|release)\[([^\]]+?)\]')
    
    # Find all occurrences of the pattern in the input string
    matches = re.findall(pattern, response)
    
    # Format the matches back into the desired string format and return as a list
    predicates = [f'{robot_predicate}[{location_predicate}]' for robot_predicate, location_predicate in matches]

    reserved_characters = {'F', 'G', 'X', 'R', 'M', 'W', 'U'}
	# Create a filtered string of uppercase letters
    available_characters = ''.join(ch for ch in string.ascii_uppercase if ch not in reserved_characters)

	# Check if there are enough available characters
    if len(predicates) > len(available_characters):
        raise ValueError("Not enough available characters to encode all predicates")
    
	# Create the dictionary that maps predicates to letters
    encoded_predicates = {available_characters[i]:p for i, p in enumerate(predicates)}

    for predicate_key in encoded_predicates:
       response = response.replace(encoded_predicates[predicate_key], predicate_key)

    return encoded_predicates, response


def generate_ltl(llm_api, language_query, training_data_folder, lang2embedding_path, lang2ltl_path, k, strategy=None, enable_prints=None):
	lang_txt = 'eng.txt'
	ltl_txt = 'ltl.txt'
	example_lang2ltl = []

	#load in context dataset from file
	english_list = []
	ltl_list = []
	with open(f"{training_data_folder}/{lang_txt}") as f:
		contents = f.read()
		english_list = contents.split("\n")

	with open(f"{training_data_folder}/{ltl_txt}") as f:
		contents = f.read()
		ltl_list = contents.split("\n")

	example_lang2ltl = list(zip(english_list,ltl_list))

	random.shuffle(example_lang2ltl)

	#query the LLM
	raw_response,llm_history = query_llm(llm_api, language_query, example_lang2ltl, k_num=k, strategy=strategy, lang2embedding_path=lang2embedding_path, lang2ltl_path=lang2ltl_path, enable_prints=enable_prints)

	encoded_predicates, parsed_response = parse_spatial_lifted_ltl(raw_response)

	return([parsed_response,encoded_predicates], llm_history)