import re
import spot
import copy
import string
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import limp.language.temporal_logic.dfa as dfa
from limp.planner.multi_level_planner import max_possible_stepwise_dfa_paths
from limp.language.llm4tl import generate_ltl, generate_response_from_llm, parse_spatial_lifted_ltl


def get_spatial_referents(encoding_map):
    referent_spatial_predicates = extract_instruction_referent_details(encoding_map)
    referent_spatial_details = parse_spatial_relations(referent_spatial_predicates)
    return referent_spatial_details


# Function to extract referents and spatial location predicates from the instruction for grounding
def extract_instruction_referent_details(encoding_map):
    plan_predicates = list(encoding_map.values())
    pattern = re.compile(r'(near|pick|release)\[([^\]]+?)\]') 
    referent_spatial_predicates = []

    # Extract spatial location predicates from robot action predicates (near,pick,release) with regex
    for pred in plan_predicates:
        matches = re.findall(pattern, pred)
        if matches:
            if matches[0][0] == "release":
                referent_spatial_predicates.extend(matches[0][1].split(",",1)) # Splitting referent to release and location to drop at and adding to referent_spatial_predicates individually
            else:
                referent_spatial_predicates.append(matches[0][1])
    referent_spatial_predicates = list(set(referent_spatial_predicates))
    return referent_spatial_predicates

# Modified function to recursively get unique referents as keys and spatial details as values
def parse_spatial_relations(input_list):
    result = {}

    def recursive_parse(item, parent=None):
        if '::' in item:
            item_components = item.split('::')
            object_name = item_components[0]
            details = item_components[1:]
            detail_str = '::'.join(details)
            open_parentheses = 0
            current_detail = ''
            
            for char in detail_str:
                if char == '(':
                    open_parentheses += 1
                elif char == ')':
                    open_parentheses -= 1

                current_detail += char

                if open_parentheses == 0 and char == ')':
                    current_detail = current_detail.strip()
                    # Remove leading '::' from the detail
                    if current_detail.startswith('::'):
                        current_detail = current_detail[2:]

                    if object_name not in result:
                        result[object_name] = []
                    if current_detail not in result[object_name]:
                        result[object_name].append(current_detail)
                    inside_parentheses = current_detail[current_detail.find("(")+1:current_detail.rfind(")")]
                    for part in inside_parentheses.split(','):
                        recursive_parse(part.strip(), object_name)
                    current_detail = ''

        else:
            if parent and item not in result:
                result[item] = []
            elif not parent and item not in result:
                result[item] = []

    for entry in input_list:
        recursive_parse(entry)
    
    return result

def translate_english(decoded_transition_expression):
    logical_replacements = decoded_transition_expression.replace("&","and").replace("!","not")
    english_replacements = logical_replacements.replace("near","be near the ").replace("pick","have picked up the ").replace("release","have released the ").replace(",","] at the [")
    return english_replacements


def find_transition_expression(dfa_edges, from_state, to_state):
    for t in dfa_edges:
        if t[0] == from_state and t[1] == to_state:
            logical_expression = t[2]
            return logical_expression
        
def decode_transition_expression(encoded_expression,encoding_map):
    logical_expression = ' '+' '.join(encoded_expression)+' ' #adding spaces to make replacement explicit and not wrongly replace internal same alphabets

    # Replace variables in the expression using the encoding map
    for key, value in encoding_map.items():
        logical_expression = logical_expression.replace(f" {key} ", f" {value} ")
    return logical_expression


def verification_reprompting(verification_type, error_clarification, llm_response_history, strategy, llm_api):
    using_parts_keyword = False
    try:
        last_response = llm_response_history[-1]['content'] #gpt api uses content keyword in history
    except:
        last_response = llm_response_history[-1]['parts'] #gemini api uses parts keyword in history
        using_parts_keyword = True
    message_to_llm = copy.copy(llm_response_history)
    
    referent_reprompt_text = f'''There was a mistake with your output LTL formula: Error with referent verification! Consider the clarification feedback and regenerate the correct output for the Input_instruction. Make sure to adhere to all rules and instructions in your system prompt!
    previous output:{last_response}
    error clarifcation: {error_clarification}
    correct output:'''
    
    task_structure_reprompt_text = f'''There was a mistake with your output LTL formula: Error with task structure verification! Consider the clarification feedback and regenerate the correct output for the Input_instruction. Make sure to adhere to all rules and instructions in your system prompt!
    previous output: {last_response}
    error clarifcation: {error_clarification}
    correct output:'''

    if verification_type=="referent_verification":
        reprompt_text = referent_reprompt_text
    elif verification_type=="task_structure_verification":
        reprompt_text = task_structure_reprompt_text

    if using_parts_keyword==False:
        message_to_llm.append({'role': 'user','content': reprompt_text})
    elif using_parts_keyword==True:
        message_to_llm.append({'role': 'user','parts': reprompt_text})
    raw_response, llm_history = generate_response_from_llm(llm_api, messages_history = message_to_llm)
    raw_response = raw_response.replace(" ", "")

    #parse response
    if strategy=="two_stage_similar_embedding" or strategy=="two_stage_random_embedding" or strategy=="single_stage":
        encoded_predicates, parsed_response = parse_spatial_lifted_ltl(raw_response)
        spot_ltl, encoded_map  = spotify(parsed_response,encoded_predicates)
        return encoded_map, parsed_response, spot_ltl, llm_history


def referents_to_english(encoding_map, ui_mode=False, baseline_mode=False):
    referent_spatial_details=get_spatial_referents(encoding_map,) ## Extract spatial information
    referent_details_english = ""
    verification_referent_list = []
    for referent_name in referent_spatial_details:
        referent_details = referent_spatial_details[referent_name]
        if (len(referent_details)!=0):
            for detail in referent_details:
                verification_referent_list.append(f"{referent_name}::{detail}")
        else:
            verification_referent_list.append(f"{referent_name}")

    ## remove duplicates
    verification_referent_list = set(verification_referent_list)
    if baseline_mode == False: 
        referent_details_english+="I extracted this list of relevant objects based on your instruction:\n"

    for idx, refname in enumerate(verification_referent_list):
        referent_details_english+=f"\t* {refname}\n"

    if baseline_mode == True: return referent_details_english

    referent_details_english+="Does this match your intention?"
    if ui_mode==False:
        referent_details_english+=" (y/n)"
    return referent_details_english


def referent_verification(input_lang_instruction,encoding_map, response_ltl, spot_ltl, llm_response_history, strategy_choice, llm_api):
    user_referent_verification_response = "n"

    while user_referent_verification_response.lower() != "y":
        print("Original instruction: ",input_lang_instruction)
        try:
            print("Last llm ltl response: ", llm_response_history[-1]['content'],"\n")  #gpt api uses content keyword in history
        except:
            print("Last llm ltl response: ", llm_response_history[-1]['parts'],"\n") #gemini api uses parts keyword in history

        print("*****************************\nReferent Verification\n*****************************")
        referents_english=referents_to_english(encoding_map)
        print(referents_english)

        user_referent_verification_response = input()

        if user_referent_verification_response.lower() == "y":
            return encoding_map, response_ltl, spot_ltl, llm_response_history
        
        elif user_referent_verification_response.lower() == "n":
            print("Clarify what went wrong and what you intended")
            user_clarification_response = input()
            print("\t",user_clarification_response)
            # Reprompt with history and user_clarification_response
            encoding_map, response_ltl, spot_ltl, llm_response_history = verification_reprompting("referent_verification", user_clarification_response, llm_response_history, strategy_choice, llm_api)
        else:  
            return encoding_map, response_ltl, spot_ltl, llm_response_history
        print("")


def task_structure_to_english(encoding_map,response_ltl,ui_mode=False,baseline_mode=False):
    task_details_english = ""
    #constructing task dfa from ltl formula to extract subgoal objectives (Transition to progressive DFA states)
    task_dfa, dfa_graph = ltl2dfa(encoding_map,response_ltl,visualize_details=False,show_diagram=False)
    initial_state, accepting_states, ltl2state, edges = task_dfa.dfa_details()

    #obtain path through dfa that leads to goal state
    infeasible_paths = []
    selected_dfa_path, original_paths = max_possible_stepwise_dfa_paths(dfa_graph,initial_state,accepting_states,infeasible_paths,print_flag=False)

    if baseline_mode==False: 
        task_details_english+="Based on my understanding here is the sequence of subgoal objectives needed to satisfy the task:\n"
    for i in range(len(selected_dfa_path) - 1):
        # translate and print transition logical expressions for path through dfa
        from_state = selected_dfa_path[i]
        to_state = selected_dfa_path[i + 1]
        logical_transition_expression = find_transition_expression(edges, from_state, to_state).strip()
        decoded_transition_expression = decode_transition_expression(logical_transition_expression, encoding_map).strip()
        english_expression=translate_english(decoded_transition_expression)

        if ui_mode :
            task_details_english+=f"* Subgoal_{i + 1}: I should {english_expression}\n"
        elif baseline_mode:
            task_details_english+=f"   Subgoal_{i + 1}: {decoded_transition_expression}\n"
        else:
            task_details_english+=f"Subgoal_{i + 1}:\n"
            task_details_english+=f"\t Logical Expression: {logical_transition_expression}\n"
            task_details_english+=f"\t Decoded Expression: {decoded_transition_expression}\n"
            task_details_english+=f"\t English translation: I should {english_expression}\n"
            
    if baseline_mode==True: return task_details_english
        
    task_details_english+="Does this match your intention?"
    if ui_mode==False:
         task_details_english+=" (y/n)"
    
    return task_details_english, selected_dfa_path, original_paths


def task_structure_verification(input_lang_instruction,encoding_map, response_ltl, spot_ltl, llm_response_history, strategy_choice, llm_api):
    user_task_verification_response = "n"

    while user_task_verification_response.lower() != "y":
        
        task_details_english, selected_dfa_path, original_paths = task_structure_to_english(encoding_map,spot_ltl)
        #task structure verification
        print("Plausible DFA paths to goal: ",original_paths,"Selected_path: ",selected_dfa_path,"\n")
        print("Original instruction: ",input_lang_instruction)
        try:
            print("Last llm ltl response: ", llm_response_history[-1]['content'])
        except:
            print("Last llm ltl response: ", llm_response_history[-1]['parts'])
        print("Encoded form: ",spot_ltl,"\n")
        print("*****************************\nTask Structure Verification\n*****************************")
        
        # print("Here are the sequence of subgoal objective states needed to satisfy the task:"
        print(task_details_english)


        user_task_verification_response = input()

        if user_task_verification_response.lower() == "y":
            return encoding_map, spot_ltl, spot_ltl, llm_response_history,selected_dfa_path
        elif user_task_verification_response.lower() == "n":
            print("Clarify what went wrong and what you intended")
            user_clarification_response = input()
            print("\t",user_clarification_response)
            # Reprompt with history and user_clarification_response
            encoding_map, spot_ltl, spot_ltl, llm_response_history = verification_reprompting("task_structure_verification", user_clarification_response, llm_response_history, strategy_choice, llm_api)  
        else:  
            return encoding_map, spot_ltl, spot_ltl, llm_response_history,selected_dfa_path
        print("")


def visualize_dfa_details(task_dfa,ltl_spot,encoding_map):
    initial_state, accepting_states, ltl2state, edges = task_dfa.dfa_details()

    print(f"********************************\nTask: {ltl_spot}")
    # display(ltl_spot)
    print(f"Symbolic maping: {encoding_map}")
    print(f"********************************\nTask Dfa Details\n********************************")
    print(f"Initial_state    : {initial_state}")
    print(f"Accepting_states :{accepting_states}")
    print(f"Ltl2state        :{ltl2state}")
    print(f"DFA Transitions:")
    for edge in edges:
        print(f"\t Edge: {edge}")

def build_dfa_graph(task_dfa, visualize, show_labels, path=None, use_cmap=True):
    initial_state, accepting_states, ltl2state, edges = task_dfa.dfa_details()
    dfa_graph = nx.DiGraph()
    dfa_graph.add_nodes_from(ltl2state.values())

    edge_labels = {}
    for edge in edges:
        dfa_graph.add_edge(edge[0], edge[1], action=edge[2])
        edge_labels[(edge[0], edge[1])] = edge[2]

    # Generate colors for the path if provided
    if path:
        path_edges = list(zip(path, path[1:]))
        cmap = plt.cm.get_cmap('winter', len(path_edges))
        path_colors = [cmap(i) for i in range(len(path_edges))]
    
    if visualize:
        pos = nx.circular_layout(dfa_graph)
        fig, ax = plt.subplots(figsize=(10, 10))
        # ax.set_facecolor('#1D1F21')
        # nx.draw_networkx(dfa_graph, pos, node_size=1000, node_color="grey", linewidths=2, ax=ax, edge_color="white")
        nx.draw_networkx(dfa_graph, pos, node_size=1000, node_color="grey", linewidths=2, ax=ax)
        nx.draw_networkx_nodes(dfa_graph, pos, node_size=1000, nodelist=accepting_states,  node_color="white", edgecolors="tab:green", linewidths=2)
        nx.draw_networkx_nodes(dfa_graph, pos, node_size=1000, nodelist=[initial_state], node_color="white", edgecolors="tab:blue", linewidths=2)
        if path:
            for edge, color in zip(path_edges, path_colors):
                if use_cmap:
                    nx.draw_networkx_edges(dfa_graph, pos, edgelist=[edge], edge_color=[color], width=2.0, arrows=True)
                else:
                    nx.draw_networkx_edges(dfa_graph, pos, edgelist=[edge], edge_color="red", width=2.0)
        if show_labels:
            nx.draw_networkx_edge_labels(dfa_graph, pos, edge_labels=edge_labels, label_pos=0.4, font_size=10)
        
        # Add legend for edge colors
        if path and use_cmap:
            legend_elements = [
                Line2D([0], [0], color=cmap(i), lw=2, label=f'Subgoal_{i+1}: {edge_labels[edge]}') for i,edge in enumerate(path_edges)
                # Show decoded transition expressions
                #  Line2D([0], [0], color=cmap(i), lw=2, label=f'Subgoal_{i+1}: {decode_transition_expression(edge_labels[edge] , ENCODING_MAP).strip()}') for i,edge in enumerate(path_edges)
            ]
            plt.legend(handles=legend_elements, loc='lower right')

        plt.show()
    return dfa_graph


def _get_spot_format(ltl_formula,lpopl_format=False):
    if lpopl_format:
        ltl_formula = str(ltl_formula).replace("(","").replace(")","").replace(",","")
    ltl_spot = ltl_formula.replace("'until'","U").replace("'not'","!").replace("'or'","|").replace("'and'","&")
    ltl_spot = ltl_spot.replace("'next'","X").replace("'eventually'","F").replace("'always'","G").replace("'True'","t").replace("'False'","f").replace("\'","\"")
    return ltl_spot


def spotify(ltl_formula, encoding_map, multiple_occurrences=True):
    encoding_map = copy.deepcopy(encoding_map)
    reserved_characters = {'F', 'G', 'X', 'R', 'M', 'W', 'U'}
    
    if multiple_occurrences:
        # print("Resolving multiple proposition occurrences")
        available_characters = ''.join(ch for ch in string.ascii_uppercase if ch not in reserved_characters)
        seen_chars = set()
        available_characters_iter = iter(ch for ch in available_characters if ch not in seen_chars)
        new_formula = []
        
        # for when the formula says sth happens multiple times hence the same character reoccurs we should not override it
        for char in ltl_formula:
            if char.isalpha() and char.upper() not in reserved_characters:
                if char not in seen_chars:
                    seen_chars.add(char)
                    new_formula.append(char)
                else:
                    # Assign a new character for each occurrence
                    new_char = next(available_characters_iter)
                    new_formula.append(new_char)
                    encoding_map[new_char] =  encoding_map[char]
            else:
                new_formula.append(char)
        ltl_formula = ''.join(new_formula)

    if  isinstance(ltl_formula,tuple):
        ltl_spot = _get_spot_format(ltl_formula,lpopl_format=True)
    else:
        ltl_spot = _get_spot_format(ltl_formula)

    try:
        f = spot.formula(ltl_spot)
    except Exception as e:
            # print(f"Error in spotify: {e}")
            if "missing closing parenthesis" in str(e):
                # print("Adding missing closing parenthesis")
                ltl_spot = ltl_spot + ")"
                f = spot.formula(ltl_spot)
            else:
                print(f"Error in spotify: {e}")

    f = spot.simplify(f)
    ltl_spot = f.__format__("l")
    # return ltl_spot
    # return f#.to_str('latex')
    return f, encoding_map

def get_std_format(ltl_spot):

    s = ltl_spot[0]
    r = ltl_spot[1:]

    if s in ["X","U","&","|"]:
        v1,r1 = get_std_format(r)
        v2,r2 = get_std_format(r1)
        if s == "X": op = 'next'
        if s == "U": op = 'until'
        if s == "&": op = 'and'
        if s == "|": op = 'or'
        return (op,v1,v2),r2

    if s in ["F","G","!"]:
        v1,r1 = get_std_format(r)
        if s == "F": op = 'eventually'
        if s == "G": op = 'always'
        if s == "!": op = 'not'
        return (op,v1),r1

    if s == "f":
        return 'False', r

    if s == "t":
        return 'True', r

    if s[0] == '"':
        return s.replace('"',''), r

    assert False, "Format error in spot2std"

#function to compute rotation matrix from quaternion
def rotation_matrix_from_quaternion(quaternion):
   # Step 1: Normalize the quaternion
   quaternion = quaternion / np.linalg.norm(quaternion)

   # Step 2: Extract quaternion components
   w, x, y, z = quaternion

   # Step 3: Construct rotation matrix
   R = np.array([[1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
               [2 * x * y + 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * w * x],
               [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2]])
   return R

def ltl2dfa(encoding_map,ltl_spot,visualize_details=False,show_diagram=False,show_labels=True,path=None):
    """
    This function converts LTL formula to DFA.

    Parameters:
        ltl_formula (str): The LTL formula of the task.

    """
    ltl_std,_ = get_std_format(ltl_spot.__format__('l').split(' ')) #prefix notation with operator words
    ltl_lpopl= str(ltl_std).replace("'eventually'","\'until\',\'True\'") #to work with DFA construction replace eventually with "until True"

    encoded_ltl = eval(ltl_lpopl) #already encoded in spatial approach

    prop_list = list(encoding_map.keys())

    task_dfa = dfa.DFA(encoded_ltl,prop_list) 
    dfa_graph = build_dfa_graph(task_dfa,show_diagram,show_labels,path)

    if visualize_details==True:
        visualize_dfa_details(task_dfa,ltl_spot,encoding_map)

    return task_dfa, dfa_graph


def llm4tl(llm_api, input_lang,in_context_examples_file,lang2embedding_path,lang2ltl_path,in_context_count, enable_prints=None, strategy=None):
    result, llm_history = generate_ltl(llm_api, input_lang, training_data_folder=in_context_examples_file, lang2embedding_path=lang2embedding_path, lang2ltl_path=lang2ltl_path, k=in_context_count, strategy=strategy, enable_prints=enable_prints)
    
    ltl_formula = result[0]
    orig_encoding_map = result[1]

    spot_ltl, spotify_encoding_map = spotify(ltl_formula,orig_encoding_map)

    return spotify_encoding_map, ltl_formula, spot_ltl, llm_history