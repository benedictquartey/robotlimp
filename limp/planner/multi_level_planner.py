import networkx as nx
import numpy as np
from osg.spatial_relationships import all_element_with_label
import open3d as o3d
import copy
import limp.planner.fmt as fmt
import limp.utils.fmt_utils as fmt_utils
import time
import os
import matplotlib.colors
import matplotlib.pyplot as plt
import traceback

class NoDFAGoalNodes(Exception):
    pass


def longest_progressive_chain(list_of_lists):
    def progressive_chain_length(sublist):
        if not sublist:  # Handle empty sublists
            return 0
        # Start from the first element
        current_length = 1
        for i in range(1, len(sublist)):
            # Check if current element is exactly one more than the previous
            if sublist[i] == sublist[i - 1] + 1:
                current_length += 1
            else:
                break  # Break if the sequence is interrupted
        return current_length

    max_chain_length = 0
    best_sublist = None

    # Find the sublist with the longest progressive chain
    for sublist in list_of_lists:
        current_length = progressive_chain_length(sublist)
        if current_length > max_chain_length:
            max_chain_length = current_length
            best_sublist = sublist

    return best_sublist

def get_max_length_list(input_list):
    # Convert list of lists into an array of lengths
    lengths = np.array([len(sublist) for sublist in input_list])
    
    # Find the maximum length
    max_length = lengths.max()
    
    # Filter and return all sublists with the maximum length
    return [sublist for i, sublist in enumerate(input_list) if lengths[i] == max_length]


def dfa_level_planning(dfa_paths):
    #selecting min length path
    path_lens = [len(path) for path in dfa_paths]
    min_path_idx = np.argmax(path_lens)
    selected_path = dfa_paths[min_path_idx]
    return selected_path

def get_next_dfa_path(start_dfa_state,accepting_states,original_paths,infeasible_paths):
    dfa_paths = []
    for path in original_paths:
        if path not in infeasible_paths:
            dfa_paths.append(path)
    if len(dfa_paths)!=0:
        print(f"\nAll feasible paths through DFA state from start task dfa state:{start_dfa_state} to accepting states:{accepting_states}")
        print(dfa_paths)
 
    if len(dfa_paths)==0:
        return None

    selected_path = dfa_level_planning(dfa_paths)

    return selected_path


def max_possible_stepwise_dfa_paths(dfa_graph,current_dfa_state,accepting_states,infeasible_paths,print_flag=True):
    dfa_paths = []
    all_paths = list(nx.all_simple_paths(dfa_graph, source=current_dfa_state, target=accepting_states))

    for path in all_paths:
        if path not in infeasible_paths:
            dfa_paths.append(path)
    if len(dfa_paths)!=0:
        if print_flag: print(f"\nAll paths through from current task dfa state:{current_dfa_state} to accepting states:{accepting_states}")
        if print_flag: print(dfa_paths)
 
    if len(dfa_paths)==0:
        return None,None

    max_len_paths = get_max_length_list(dfa_paths)
    
    #select the version that flows progressively for intermediate states save for accepting state. ie choose 0,1,2,3,[5] as opposed to 0,1,2,4,[5]
    selected_path = longest_progressive_chain(max_len_paths)
    return selected_path, max_len_paths


def get_stepwise_paths_to_accepting_states(current_dfa_state,accepting_states,infeasible_paths,print_flag=True):
    dfa_paths = []

    for accepting_state in accepting_states:
        # Get path from current dfa state through all subgoal states to each accepting states
        path = list(range(current_dfa_state,accepting_state+1))
        if path not in infeasible_paths:
            dfa_paths.append(path)
    if len(dfa_paths)!=0:
        if print_flag: print(f"\nAll feasible paths through DFA state from current task dfa state:{current_dfa_state} to accepting states:{accepting_states}")
        if print_flag: print(dfa_paths)

    if len(dfa_paths)==0:
        return None,None
    
    selected_path = dfa_level_planning(dfa_paths)
    return selected_path, dfa_paths

def progressive_motion_planner(current_start_point, task_dfa, dfa_graph, pointcloud, filtered_relevant_element_details, encoding_map, nearness_value, map_resolution=0.01, h_min=-3,h_max=-0.15,robot_motion_type="3D",height_2d=0, stepfactor=30, use_heuristic=True,visualize = True,tmp_fldr=".", goal_sample_percentage=100,show_color_bars=True):
    
    fmt_planner = fmt.FMTPlanner(n_samples=2500, r_n=100, path_resolution=0.1, rr=1.0, max_search_iter=10000)
    complete_motion_route = []
    complete_motion_route_cost = 0
    planning_complete = False
    failed_planning = None
    infeasible_paths = []
    motion_stack = [] 
    saved_plan={"grid_plan":{},
                "world_plan":{}}

    
    original_start = current_start_point 
    initial_state, accepting_states, ltl2state, edges = task_dfa.dfa_details()
    # current_dfa_state = task_dfa.state
    
    print(f"*************************************************\nStarting Bi-level Planner\n-------------------------------------------------")
    print(f"Obtaining predicate satisfying referent positions ...\n-------------------------------------------------")
    predicate_satisfying_positions_navigation_only, all_predicate_satisfying_positions, predicate_satisfying_positions_navigation_only_print, all_predicate_satisfying_positions_print = get_predicate_satisfying_positions(filtered_relevant_element_details, encoding_map)
    print(f"\nAll predicate satisfying referent positions: {all_predicate_satisfying_positions_print}")
    # print(f"\nNavigation Predicate satisfying referent positions: {predicate_satisfying_positions_navigation_only_print}")

    print(f"\n*************************************************\nBegin Planning\n-------------------------------------------------")
    
    selected_dfa_path, original_paths = max_possible_stepwise_dfa_paths(dfa_graph,initial_state,accepting_states,infeasible_paths,print_flag=False) #get max length path this translates to possible stepwise paths and alternatives
    #TODO: revert to planning with next route if first one fails

    #begin bi-level planning
    while not planning_complete:
        if selected_dfa_path == None:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nNo feasible paths through DFA states\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            failed_planning = True
            planning_complete = True
            break
    
        print(f"*************************************************\nPlanning with selected DFA Path: {selected_dfa_path}\n-------------------------------------------------")
        plan_step = 1
        # current_dfa_state = initial_state

        while task_dfa.state != selected_dfa_path[-1]:
            if selected_dfa_path == None:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nNo feasible paths through DFA states\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                failed_planning = True
                planning_complete = True
                break

            print(f"\nProgressive planning step {plan_step} of {len(selected_dfa_path)-1}\n-------------------------------------------------")
            allowed_next_dfa_state = selected_dfa_path[task_dfa.state+1]
            required_transitions = get_dfa_required_transition(task_dfa.state, allowed_next_dfa_state, edges)

            print(f"Current position: {current_start_point}") 
            print(f"Current dfa state: {task_dfa.state}") 
            print(f"Allowed next dfa state: {allowed_next_dfa_state}") 
            print(f"Transition required: {required_transitions}")

            #assign values to predicates based on required transitions
            values = assign_values_to_predicates(required_transitions,list(all_predicate_satisfying_positions.keys()))
            
            # Check_predicate_action type and continue accordingly
            print(f"\nChecking predicate action type...")
            step_action_type, actual_predicate, encoded_predicate = get_step_action_type(values, encoding_map)
            if step_action_type == "skill":
                action = actual_predicate.split("[")[0]
                print(f"Next step in plan is '{action}' skill || Executing skill and updating dfa state... ")
                object_label = actual_predicate.split("[")[1].split("]")[0]
                object_positions = all_predicate_satisfying_positions_print[encoded_predicate]
                try:
                    previous_step = saved_plan["grid_plan"][f"step_{plan_step-1}"]
                    if previous_step["action"] == "navigation":
                        chosen_goal_referent = previous_step["chosen_goal_referent"]
                except:
                    print("No previous plan step")
                    chosen_goal_referent=None
                saved_plan["grid_plan"][f"step_{plan_step}"]= {"action":action,
                                                               "chosen_goal_referent": chosen_goal_referent,
                                                                  "action_parameter":object_label,
                                                                  "action_parameter_positions":object_positions
                                                                  }
                saved_plan["world_plan"][f"step_{plan_step}"]= {"action":action,
                                                                "chosen_goal_referent": chosen_goal_referent,
                                                                  "action_parameter":object_label,
                                                                  "action_parameter_positions":object_positions} #get index from goal_origin_predicate_idx and get id at object_positions[idx]
                ##update task dfa state
                task_dfa.state = allowed_next_dfa_state
                plan_step += 1
                continue #go to next step of planning
            elif step_action_type == "navigation":
                print(f"Next step in plan is '{step_action_type}' type || Continuing with progressive planning... ")

            # remove keys in values not also in predicate_satisfying_positions_navigation_only
            values = {key:values[key] for key in values.keys() if key in predicate_satisfying_positions_navigation_only.keys()}
            print(f"Valid Predicates (Navigation only): {values}")

            #augment pointcloud 
            relevant_predicate_satisfying_positions = {key:predicate_satisfying_positions_navigation_only[key] for key in values.keys()}
            augmented_point_cloud, value_map = generate_tpsm(pointcloud, relevant_predicate_satisfying_positions, values, nearness_threshold=nearness_value, motion_type=robot_motion_type,z_height_2d=height_2d, step_factor=stepfactor) #Task Progression Semantic Map
            
            #get all predicates that are also true for all points in value map
            print(f"\nGetting all predicates that are true for all points in value map...")
            transition_unallowed_predicates = [key for key in values.keys() if values[key] == -1]
            all_truth_valuemap = get_all_value_map_point_true_predicates(value_map,predicate_satisfying_positions_navigation_only,nearness_value,transition_unallowed_predicates)

            print(f"Getting goal points from value map...")
            goal_points, all_truth_valuemap = filter_goal_points_with_dfa(all_truth_valuemap, task_dfa, task_dfa.state, allowed_next_dfa_state)
            
            try:
                if len(goal_points) == 0: #no goal nodes means we are in a dead end, need to replan
                    raise NoDFAGoalNodes("NoDFAGoalNodes Exception occurred!")
            
                # The pointcloud to be transformed into obstacle map can simply only keep added points from the augmentation
                #that are goal points (value of 1) and those that are violating points (value of -1) we will treat those as obstacles as well
                #since points that are allowable (value of 0) we dont need to add them to the pointcloud

                #get obstacle pointcloud from augmented point cloud
                print(f"Creating obstacle map... ")
                obstacle_map_pointcloud = get_obstacle_map_pointcloud(augmented_point_cloud,all_truth_valuemap)

                #group goal points by predicate_index 
                world_goal_points_predicate_grouped = {}
                for point in goal_points:
                    goal_key = f"{point['origin_predicate']}_{point['origin_predicate_index']}"
                    if goal_key not in world_goal_points_predicate_grouped.keys():
                        world_goal_points_predicate_grouped[goal_key] = []
                    world_goal_points_predicate_grouped[goal_key].append(point['point_position'])

                obstacle_map, pred_grouped_goal_points,map_min_bound, map_max_bound = generate_obstacle_map(obstacle_map_pointcloud, world_goal_points_predicate_grouped, map_resolution, h_min,  h_max)
                print(f"TPSM obstacle map generated with min_bounds:{map_min_bound} || max_bound:{map_max_bound} || resolution:{map_resolution} || h_min_bottom:{h_min} || h_max_roof:{h_max} ")
                #group goals based on predicate_index
                map_grouped_goals = {}
                for key, value in pred_grouped_goal_points.items():
                    map_grouped_goals.setdefault(value, []).append(key)

                #sample goals from each group this is what we will plan to
                grouped_sampled_goals = fmt_utils.sample_goals(map_grouped_goals, goal_sample_percentage)
                
                #save progressive planning step artifacts to disk
                print(f"Saving progressive planning step artifacts to disk... ")
                obstacle_map_title = "TPSM Obstacle Map  ||  DFA Transition: " + str(required_transitions) + "  ||  Planning step " + str(plan_step) + " of " + str(len(selected_dfa_path)-1)
                sampled_obstacle_map_title = f"{goal_sample_percentage}% Sampled Goals Map ||  DFA Transition: " + str(required_transitions) + "  ||  Planning step " + str(plan_step) + " of " + str(len(selected_dfa_path)-1)
                progressive_plan_step_title = f"progressive_step_{plan_step}_of_{len(selected_dfa_path)-1}_artifacts"
                    
                save_progressive_planning_step_artifacts(progressive_plan_step_title, tmp_fldr, augmented_point_cloud, all_truth_valuemap, goal_points, obstacle_map, obstacle_map_title, sampled_obstacle_map_title, grouped_sampled_goals, pred_grouped_goal_points,show_color_bar=show_color_bars)
                
                if len(grouped_sampled_goals) == 0: #no goal nodes means we are in a dead end, need to replan         
                    raise NoDFAGoalNodes("NoDFAGoalNodes Exception occurred!")

                #low level motion planning with FMT* here
                motion_plan_info, next_start_point, plan_step, current_dfa_state = motion_level_planning(motion_stack,obstacle_map,grouped_sampled_goals,fmt_planner,current_start_point,use_heuristic,required_transitions,plan_step,selected_dfa_path,task_dfa)
                #update exisiting tracking variables with info in motion plan to make sure it is up to date and accounts for bactracking etc
                obstacle_map = motion_stack[motion_plan_info["id"]]["obstacle_map"]
                required_transitions = motion_stack[motion_plan_info["id"]]["required_transitions_before_plan"]
                current_start_point = motion_stack[motion_plan_info["id"]]["startpoint_before_plan"]
                current_dfa_state = motion_stack[motion_plan_info["id"]]["dfa_state_before_plan"]
                allowed_next_dfa_state = selected_dfa_path[current_dfa_state+1]

                #save progressive planning step artifacts to disk
                print(f"Saving updated motion plan info to disk... ")
                progressive_plan_step_title = f"progressive_step_{plan_step}_of_{len(selected_dfa_path)-1}_artifacts"
                
                save_progressive_planning_step_artifacts(progressive_plan_step_title, tmp_fldr, motion_plan_info=motion_plan_info,show_color_bar=show_color_bars)

                #visualize computed motion plan
                if visualize == True:
                    goal = [motion_plan_info['goal']]
                    viz_title = f"Computed Motion Plan  ||  DFA Transition: {required_transitions}  ||  Planning step {plan_step} of {len(selected_dfa_path)-1}"
                    fig = fmt_utils.get_result_visualization(viz_title, obstacle_map, fmt_planner, motion_plan_info, current_start_point, goal,show_color_bar=show_color_bars)
                    fig.savefig(tmp_fldr + progressive_plan_step_title + "/"+"computed_motion_plan.png")  # Save the figure as an image
                    

                if motion_plan_info['goal_flag'] == 1 and step_action_type == "navigation": 
                    # a plan was found
                    complete_motion_route.extend(motion_plan_info['path'])
                    complete_motion_route_cost += motion_plan_info['cost']

                    goal_chosen_referent_key, goal_chosen_referent_idx = motion_plan_info['goal_origin_predicate'].split("_")
                    goal_chosen_referent = all_predicate_satisfying_positions_print[goal_chosen_referent_key][int(goal_chosen_referent_idx)]
                    saved_plan["grid_plan"][f"step_{plan_step}"]= {"action":"navigation",
                                                                   "chosen_goal_referent":goal_chosen_referent,
                                                                   "path":motion_plan_info['path']}
                    saved_plan["world_plan"][f"step_{plan_step}"]= {"action":"navigation",
                                                                    "chosen_goal_referent":goal_chosen_referent,
                                                                    "path":grid_to_world(motion_plan_info['path'], map_min_bound, map_resolution)}
                    

                    # We now assume robot executes correctly and gets to the right waypoint and environment stayed the same, while planning || we can recapture image, ground and progress DFA to transition DFA state allowing for detection in environment change since mapping => Perturbation Invariant Planning(Future work)
                    ##update task dfa state
                    task_dfa.state = allowed_next_dfa_state

                    # update current startpoint
                    current_start_point = next_start_point

                    failed_planning = False

                elif motion_plan_info['goal_flag'] == 0:
                    print("Current motion plan with DFA path not feasible, trying a different DFA path... ")
                    failed_planning = True
                    task_dfa = task_dfa.reset()
                    current_observation_node = original_start
                    infeasible_paths.append(selected_dfa_path)

                    #clear low level motion plan tracker
                    complete_motion_route=[]
                    complete_motion_route_cost=0
                    motion_plan=[]
                    motion_stack=[]
                    saved_plan={"grid_plan":{},
                                "world_plan":{}}
                    
                    selected_dfa_path,_ = get_stepwise_paths_to_accepting_states(initial_state,accepting_states,infeasible_paths) #use stepwise path to accepting states
                    # selected_dfa_path = get_next_dfa_path(initial_state,accepting_states,original_paths,infeasible_paths)       #use all paths to accepting states
                    break
            
            except NoDFAGoalNodes:
                print("No areas in pointcloud leads to the next expected goal DFA state, trying a different DFA path... ")
                failed_planning = True
                task_dfa = task_dfa.reset()
                current_observation_node = original_start
                infeasible_paths.append(selected_dfa_path)

                #clear low level motion plan tracker
                complete_motion_route=[]
                complete_motion_route_cost=0
                motion_plan=[]
                motion_stack=[]
                saved_plan={"grid_plan":{},"world_plan":{}}
                selected_dfa_path,_ = get_stepwise_paths_to_accepting_states(initial_state,accepting_states,infeasible_paths) 
                # selected_dfa_path = get_next_dfa_path(initial_state,accepting_states,original_paths,infeasible_paths)         
                break

            #move to next step of planning
            plan_step += 1

            ## View motion stack
            print("Motion stack size:",len(motion_stack))
            for element in motion_stack:
                print(f"startpoint_before_plan: {element['startpoint_before_plan']} || endpoint_after_plan(goal): {element['motion_plan_info']['goal']} || goal_origin_predicate_idx: {element['chosen_goal_origin_predicate']} || dfa_state_before_plan: {element['dfa_state_before_plan']} || plan_step_before_plan: {element['plan_step_before_plan']} || required_transitions_before_plan: {element['required_transitions_before_plan']} || cost: {element['motion_plan_info']['cost']} || n_steps: {element['motion_plan_info']['n_steps']}")
 

        if failed_planning==False:
            planning_complete = True

            #execute motion plan
            # robot_waypoints=execute_motion_plan(complete_motion_route,complete_motion_route_cost)
            # saved_plan["complete_plan"]=robot_waypoints

            #save plan  to disk
            if tmp_fldr != None:
                np.save(tmp_fldr+f'saved_plan.npy', saved_plan)
            else:
                np.save(f'saved_plan.npy', saved_plan)

    print(f"\n*************************************************\nPlanning Complete\n*************************************************\n")
    return saved_plan

def get_step_action_type(values, encoding_map): 
    #we should only be executing one action type per step || 
    # if we have multiple predicates that are true and a skill is one of them we should execute the skill || 
    # if we have multiple predicates where they are all navigation is not a problem since allowed, unallowed etc are all used to demartcice space region so we can move directly to the goal
    step_action_type="navigation"
    allowed_predicates = [key for key in values.keys() if values[key] == 0]
    for allowed_predicate in allowed_predicates:
        key_list = [val for key, val in encoding_map.items() if key == allowed_predicate]
        if len(key_list) > 0:
            actual_predicate = key_list[0]
            predicate_action = actual_predicate.split("[")[0]
            print(f"Transition predicate: {allowed_predicate} || Action: {predicate_action} || Predicate: {actual_predicate}")
            if predicate_action in ["pick","release"]:
                step_action_type="skill"
                return step_action_type, actual_predicate, allowed_predicate
    return step_action_type,actual_predicate, allowed_predicate

def save_progressive_planning_step_artifacts(progressive_plan_step_title, tmp_fldr, augmented_point_cloud=None, all_truth_valuemap=None, goal_points_info=None, obstacle_map=[], obstacle_map_title=None, sampled_obstacle_map_title=None, grouped_sampled_goals=None, pred_grouped_goal_points =None,motion_plan_info=None,show_color_bar=True, minimal_save=True):
    #create folder for saving artifacts in tmp_fldr
    progressive_plan_step_fldr = tmp_fldr + progressive_plan_step_title + "/"
    if not os.path.exists(progressive_plan_step_fldr):
               os.makedirs(progressive_plan_step_fldr)
    
    #save augmented point cloud with colors
    if augmented_point_cloud!=None:
        color_augmented_pcd = color_augment_value_map(augmented_point_cloud, all_truth_valuemap, use_original_colors=True)
        o3d.io.write_point_cloud(progressive_plan_step_fldr+"tpsm.pcd", color_augmented_pcd)

    #save value map
    if all_truth_valuemap!=None and minimal_save==False:
        np.save(progressive_plan_step_fldr+"value_map.npy", all_truth_valuemap)

    #save goal points
    if goal_points_info!=None and minimal_save==False:
        np.save(progressive_plan_step_fldr+"goal_points_info.npy", goal_points_info)

    #save obstacle map 
    if len(obstacle_map)!=0:
        if minimal_save==False:
            np.save(progressive_plan_step_fldr+"obstacle_map.npy",obstacle_map)

        #  Visualize obstacle map with all goal points
        fig = visualize_obstacle_map(obstacle_map, obstacle_map_title,show_color_bar)
        fig.savefig(progressive_plan_step_fldr+"obstacle_map_img.png")  # Save the figure as an image

        if grouped_sampled_goals != None and minimal_save==False:
            # Also visualize obstacle map with just sampled goals
            fig2 = fmt_utils.sampled_visualize_obstacle_map(obstacle_map, grouped_sampled_goals, sampled_obstacle_map_title,show_color_bar)
            fig2.savefig(progressive_plan_step_fldr+"sampled_goals_obstacle_map_img.png")  # Save the figure as an image
    
    #save obstacle map grouped goal points
    if pred_grouped_goal_points!=None and minimal_save==False:
        np.save(progressive_plan_step_fldr+"pred_grouped_goal_points.npy",pred_grouped_goal_points)

    #save motion plan info
    if motion_plan_info != None:
        np.save(progressive_plan_step_fldr+"motion_plan_info.npy", motion_plan_info)
    

def visualize_obstacle_map(obstacle_map, title, show_colorbar=True):
    # Ensure the obstacle_map is a numpy array for processing
    obstacle_map = np.array(obstacle_map)

    # Create a color map: free space (white), obstacles (black), goal points (gold)
    cmap = matplotlib.colors.ListedColormap(['black', 'white', 'gold'])
    bounds = [-0.5, 0.5, 1.5, 2.5]  # Boundaries for the colormap
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust size as needed

    # Plotting
    cax = ax.imshow(obstacle_map, cmap=cmap, norm=norm, interpolation='nearest')
    if show_colorbar:
        fig.colorbar(cax, ticks=[0, 1, 2], format=plt.FuncFormatter(lambda x, _: {0: 'Obstacles', 1: 'Free Space', 2: 'Goal Points'}[x]))
    ax.set_title(title)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(False)

    return fig  # Return the figure object

def execute_motion_plan(complete_motion_route,complete_motion_route_cost):
    print(f"\n*************************************************\nExecuting Motion Plan\n-------------------------------------------------")
    robot_waypoints = complete_motion_route

    #remove successive duplicate waypoints from partial planning process
    # iterates over the robot_waypoints list starting from the second element, compares each element with its previous element, adds to resulting list if they are different
    robot_waypoints = [robot_waypoints[0]] + [x for i, x in enumerate(robot_waypoints[1:], start=1) if np.array_equal(robot_waypoints[i-1],x) != True]

    print(f"Complete Plan Cost:{complete_motion_route_cost}")
    print(f"\nPlan:")
    print(*robot_waypoints, sep =' ')
    print(f"\n-------------------------------------------------")
    #send waypoint routes to robot, nodes to key
    return robot_waypoints

def motion_planning(obstacle_map,grouped_sampled_goals,fmt_planner,start_point,use_heuristic):
    fmt_planner.add_map_design(obstacle_map)
    # map_goals = fmt_utils.get_goal_cordinates(obstacle_map)
    
    map_goals = grouped_sampled_goals #use sampled goals instead of all goals

    #maybe use timeit instead?
    print(f"\nPlanning with FMT*...")
    print(f"Goal candidate region count({len(grouped_sampled_goals)}) || Region identifiers: {list(grouped_sampled_goals.keys())}")
    start_time = time.time()
    if use_heuristic == True:
        path_info = fmt_planner.plan(start_point, map_goals, heuristic_weight=1.0)
    else:
        path_info = fmt_planner.plan(start_point, map_goals)
    end_time = time.time()
    print(f"Planning Time {(end_time - start_time) * 1000.0} ms")
    return path_info, map_goals

def motion_level_planning(motion_stack,obstacle_map,grouped_sampled_goals,fmt_planner,start_point,use_heuristic,required_transitions,plan_step,selected_dfa_path,task_dfa):
    motion_plan_info, grouped_goal_points = motion_planning(obstacle_map,grouped_sampled_goals,fmt_planner,start_point,use_heuristic)
    
    if motion_plan_info['goal_flag'] == 1:
        motion_plan_id = len(motion_stack)
        motion_plan_info['id']=motion_plan_id
        #add plan to stack
        print("Motion plan found || Placing on stack")
        chosen_goal_point = motion_plan_info['goal']
        chosen_goal_origin_predicate = motion_plan_info['goal_origin_predicate']
        motion_stack.append({"obstacle_map":obstacle_map,
                             "grouped_goal_points":grouped_goal_points,
                             "startpoint_before_plan":start_point,
                             "dfa_state_before_plan":task_dfa.state,
                             "plan_step_before_plan":plan_step,
                             "required_transitions_before_plan":required_transitions,
                             "chosen_goal_origin_predicate":chosen_goal_origin_predicate,
                             "motion_plan_info":motion_plan_info,
                             "motion_plan_id":motion_plan_id})
        
        print(f"\n*************************************************\nRecording Motion Plan\n-------------------------------------------------")
        print(f"Motion_plan: {motion_plan_info}\n-------------------------------------------------")
        

        next_start_point = motion_plan_info['goal']
        return motion_plan_info, next_start_point, plan_step, task_dfa.state

    elif motion_plan_info['goal_flag'] == 0:
        #backtrack to previous dfa state
        print(f"No feasible motion plan found, backtracking to previous DFA transition to replan ... || Motion stack size: {len(motion_stack)}")
        try:
            #pop the last entry from the stack
            stack_entry = motion_stack.pop()               
            chosen_goal_point = stack_entry["motion_plan_info"]["goal"]  #TODO: this should be goal region not individual node which may be in the same blocked goal region. 
                                                                        #if we do this we will have to backtrack multiple times to exhaust all goal candidates in a region before moving to next region
            chosen_goal_origin_predicate = stack_entry["chosen_goal_origin_predicate"] #Todo complete here: we track the predicate instance the goal came from and remove all sampled goal points from that originating predicate
            previous_grouped_goal_points = stack_entry["grouped_goal_points"]           #TODO: Explicitly test this new backtracking!!

            #if there are other zones for the predicate we want, remove all sampled goal points from that originating predicate we previously used so we get sampled points from a new zone. 
            # Else if there is just one zone keep as is and remove the goal point from the set of sampled goal points so we can get a new goal point
            if len(previous_grouped_goal_points) > 1:
                print("Multiple goal regions || removing old region to retry another goal region") #perhaps after cycling through all regions if still no path i should repeat, removing previously chosen goal points form each region
                previous_grouped_goal_points.pop(chosen_goal_origin_predicate,None) #remove all sampled goal points from that originating predicate
            else:
                print("Single goal region || removing old goal point to retry another goal point")
                previous_grouped_goal_points[chosen_goal_origin_predicate].remove(tuple(chosen_goal_point))
            previous_startpoint = stack_entry["startpoint_before_plan"]
            previous_dfa_state = stack_entry["dfa_state_before_plan"]
            planning_step_before_navigation = stack_entry["plan_step_before_plan"]
            previous_obstacle_map = stack_entry["obstacle_map"]
            previous_required_transitions = stack_entry["required_transitions_before_plan"]

            #set task dfa to previous dfa state
            task_dfa.state = previous_dfa_state

            #reset plan step to planning_step_before_navigation

            #TODO: figure out what happens with backtracking when a skill was in previous steps since it is not recorded in motion stack
            #we will only backtrack the last navigation section and we may not know to redo the skill version
            #Potential fix is to ensure that plan_step is checked against dfa state to make sure we will redo skill if we backtrack to a previous navigation motion dfa state

            #i think i have completed this todo by tracking and passing in the planning step in recursive call to motion_level_planning || validate
            return motion_level_planning(motion_stack,previous_obstacle_map,previous_grouped_goal_points,fmt_planner,previous_startpoint,use_heuristic,previous_required_transitions,planning_step_before_navigation,selected_dfa_path,task_dfa)
        except Exception as e:
            print(f"Cannot Backtrack || Error: {e}")
            traceback.print_exc()
            return None, None, None, None

def get_obstacle_map_pointcloud(augmented_point_cloud, all_truth_valuemap, remove_allowable=True, remove_goal=True):
    obstacle_map_pointcloud = copy.deepcopy(augmented_point_cloud)
    points = np.asarray(obstacle_map_pointcloud.points)
    colors = np.asarray(obstacle_map_pointcloud.colors)

    # Initialize a removal mask with False (indicating no points are to be removed initially)
    removal_mask = np.full(len(points), False, dtype=bool)

    # Iterate through all_truth_valuemap to update the removal_mask and color of violating points
    for point in all_truth_valuemap:
        idx = point['point_index']

        if point['point_value'] == 0:
            if remove_allowable: removal_mask[idx] = True # Mark for removal
        elif point['point_value'] == 1:
            if remove_goal: removal_mask[idx] = True
            else: colors[idx] = [1, 0.843, 0] # Color goal points gold
        elif point['point_value'] == -1:
            colors[idx] = [0, 0, 0]  # Color violating points black

    # Remove points marked in the removal_mask
    points = points[~removal_mask]
    colors = colors[~removal_mask]

    # Update the point cloud with filtered points and colors
    obstacle_map_pointcloud.points = o3d.utility.Vector3dVector(points)
    obstacle_map_pointcloud.colors = o3d.utility.Vector3dVector(colors)
    return obstacle_map_pointcloud

#Faster vectorized approach
def get_all_value_map_point_true_predicates(value_map, predicate_satisfying_positions, nearness_threshold, transition_unallowed_predicates):
    # Convert value_map to a structured numpy array for vectorized operations
    # dtype = [('point_index', int), ('point_position', float, 3), ('point_value', int), ('origin_predicate', 'U1'), ('all_true_predicates', 'O')]
    dtype = [('point_index', int), 
             ('point_position', float, 3), 
             ('point_value', int), 
             ('origin_predicate', 'U1'),  # Assuming predicate strings are not longer than 1 character
             ('origin_predicate_index', int), 
             ('origin_predicate_position', float, 3),
             ('all_true_predicates', 'O')]
    value_map_array = np.array([tuple(point.values()) + ([point['origin_predicate']],) for point in value_map], dtype=dtype)

    for pred, positions in predicate_satisfying_positions.items():
        # Convert positions to a numpy array for vectorized distance computation
        positions_array = np.array(positions)

        # Compute distances between each point in value_map and each position for the current predicate
        # Reshape arrays for broadcasting: (num_points, 1, 3) and (1, num_positions, 3)
        distances = np.linalg.norm(value_map_array['point_position'][:, np.newaxis, :] - positions_array[np.newaxis, :, :], axis=2)

        # Check if any distance is within the threshold, across all positions for each point
        within_threshold = np.any(distances <= nearness_threshold, axis=1)

        # Update 'all_true_predicates' for points that are within the threshold for the current predicate
        # Skip if the predicate is the same as the point's own predicate
        for point, is_within in zip(value_map_array, within_threshold):
            if is_within and point['origin_predicate'] != pred:
                point['all_true_predicates'].append(pred)

                #if we find out that a point has a predicate that is not allowed for that transition update its value to -1
                if pred in transition_unallowed_predicates:
                    point['point_value'] = -1

    # Convert the structured array back to the original list of dictionaries format
    optimized_value_map = [dict(zip(point.dtype.names, point)) for point in value_map_array]

    return optimized_value_map

def filter_goal_points_with_dfa(all_truth_valuemap, task_dfa, current_dfa_state, allowed_next_dfa_state):
    goal_points=[]
    all_truth_valuemap = copy.deepcopy(all_truth_valuemap)

    for point in all_truth_valuemap:
        point_true_propositions = point['all_true_predicates']
        dfa_nextstate = task_dfa.check_progression_state(current_dfa_state,point_true_propositions)

        #recording prior obtained current transition relevant information on position
        if point['point_value'] == 0:
            point['info']="transition_allowed_position || "
        elif point['point_value'] == -1:
            point['info']="transition_violating_position || "
        else:
            point['info']=""

        #checking dfa transition this position causes and recording information
        if dfa_nextstate == allowed_next_dfa_state:  
            point['point_value'] = 1 
            point['info']+="next_dfastate_position"
            goal_points.append(point)
        
        #should I add explicitly handling positions that lead to violating states and other states of the DFA  ?
        elif dfa_nextstate == -1: 
            point['point_value'] = -1 
            point['info']+="dfa_violating_position"

        elif dfa_nextstate == current_dfa_state: 
            point['info']+="same_dfastate_position" 

        elif dfa_nextstate != current_dfa_state and dfa_nextstate != -1 and dfa_nextstate != allowed_next_dfa_state:
            point['point_value'] = -1 
            point['info']+="other_dfastate_position"
    
    return goal_points, all_truth_valuemap

def assign_values_to_predicates(required_transition,encoded_predicate_keys):
    values = {}
   
    required_transition = required_transition.replace("&"," ").replace("|"," ").split(" ")  # parse required_transition string to get approved keys
    print("Required_transition after parsing: ",required_transition)

    for predicate_key in required_transition:
        if predicate_key in required_transition and "!" not in predicate_key:
            values[predicate_key] = 0   #attraction/allowed
        elif predicate_key in required_transition and "!" in predicate_key:
            predicate_key = predicate_key.replace("!","")
            values[predicate_key] = -1   #avoidance
        else:
            values[predicate_key] = None
    
    print(f"Assigned values to predicates: {values}")

    return values

def get_predicate_satisfying_positions(filtered_relevant_element_details, encoding_map):
    all_predicate_satisfying_positions = {}
    predicate_satisfying_positions_navigation_only = {}
    all_predicate_satisfying_positions_print = {}
    predicate_satisfying_positions_navigation_only_print = {}

    instruction_predicates = list(encoding_map.values()) 
    print(f"Instruction predicates: {instruction_predicates} \n")

    for idx,predicate_encoded_key in enumerate(encoding_map):
        predicate = encoding_map[predicate_encoded_key]
        predicate_action = predicate.split("[")[0]
        predicate_arguments = predicate.split("[")[1].split("]")[0]
        print(f"Instruction predicate {idx+1} of {len(instruction_predicates)} \n\tEncoded Key: {predicate_encoded_key} || Predicate: {predicate:50s}|| Predicate Action: {predicate_action:8s} || Arguments: {predicate_arguments}") #replace argument in predicate with actual correlating referents from filtered_relevant_element_details

        ##find elements in filtered_relevant_element_details that correspond to predicate arguments
        satisfying_elements = find_satisfying_predicate_arguments(predicate_action,predicate_arguments,filtered_relevant_element_details)
        print(f"\tSatisfying Elements: {[element['mask_id'] for element in satisfying_elements]}") 
        if satisfying_elements != []:
            all_predicate_satisfying_positions[predicate_encoded_key]= np.asarray([element['worldframe_3d_position'] for element in satisfying_elements])
            all_predicate_satisfying_positions_print[predicate_encoded_key]= np.asarray([{"id":element['mask_id'] ,"pos":element['worldframe_3d_position']} for element in satisfying_elements])
            # positions_for_valuemap.extend(np.asarray([element['worldframe_3d_position'] for element in satisfying_elements]))
            if predicate_action == "near":
                predicate_satisfying_positions_navigation_only[predicate_encoded_key]= np.asarray([element['worldframe_3d_position'] for element in satisfying_elements])
                predicate_satisfying_positions_navigation_only_print[predicate_encoded_key]= np.asarray([{"id":element['mask_id'] ,"pos":element['worldframe_3d_position']} for element in satisfying_elements])
        else:
            print(f"\t\tNo satisfying elements for predicate {predicate}")

    return predicate_satisfying_positions_navigation_only, all_predicate_satisfying_positions, predicate_satisfying_positions_navigation_only_print, all_predicate_satisfying_positions_print

def find_satisfying_predicate_arguments(predicate_action, predicate_arguments, filtered_relevant_element_details):
    satisfying_elements = []

    if "release" not in predicate_action:
        #handling near, pick predicates
        spatial_constraint = predicate_arguments
        
    elif "release" in predicate_action:
        #only second argument is relevant for release predicate
        spatial_constraint = predicate_arguments.split(",")[1]

    try:
        parts = spatial_constraint.split("::")
        element_label = parts[0] #extract label from spatial constraint
        actual_constraint = parts[1:]
    except:
        element_label = spatial_constraint #if no spatial constraint, just get label
        actual_constraint = None

    same_label_elements = all_element_with_label(element_label, filtered_relevant_element_details)
    print(f"\tThere are {len(same_label_elements)} element(s) with '{element_label}' label")

    print(f"\t\tChecking if any of these elements satisfy constraint: {actual_constraint}")
    if actual_constraint == None:
        satisfying_elements = same_label_elements
        
    for element_details in same_label_elements:
        constraint_satisfaction = []
        for recorded_constraint in element_details['spatial_constraints']:
            if recorded_constraint['constraint'] in actual_constraint: #representation will store all extracted spatial constraint to same class name so only get the constraint that is relevant to the current instruction predicate being considered to find satisfying elements
                    print(f"\t\tElement: {element_details['mask_id']} || Checked Constraint: {recorded_constraint['constraint']} || Satisfaction: {recorded_constraint['satisfies_spatial_constraint']}")
                    constraint_satisfaction.append(recorded_constraint['satisfies_spatial_constraint'])
        if all(constraint_satisfaction): #if all relevant constraints are satisfied
            satisfying_elements.append(element_details)
                    
    return satisfying_elements
          
def get_dfa_required_transition(current_dfa_state, allowed_next_dfa_state, edges):
    for edge in edges:
        if edge[0] == current_dfa_state and edge[1] == allowed_next_dfa_state:
            return edge[2]
    return None

#new methods to directly add new points into the point cloud to represent zones of interest for planning avoidance/attraction TPSM
def generate_tpsm(point_cloud, points_of_interest, values, nearness_threshold, motion_type="3D", z_height_2d=0, step_factor=40):
    point_cloud = copy.deepcopy(point_cloud)

    # Initialize arrays for updated points and colors
    original_points = np.asarray(point_cloud.points)
    new_point_index = len(original_points)

    # Generate grid points based on motion type
    if motion_type == "3D":
        grid_points = generate_unit_sphere_grid(nearness_threshold, step_factor)
    elif motion_type == "2D":
        grid_points = generate_circle_on_plane(nearness_threshold, z_height_2d, step_factor)

    # Process each group of points of interest
    new_points_list = []
    new_colors_list = []
    value_map = []
    for predicate, pois in points_of_interest.items():
        value = values.get(predicate, 0)
        pois = np.array(pois)

        # Broadcast to create new points for all pois at once
        all_new_points = pois[:, np.newaxis, :] + grid_points[np.newaxis, :, :]
        all_new_points = all_new_points.reshape(-1, 3)  # Reshape to 2D array

        new_points_list.append(all_new_points)
        new_colors_list.append(np.zeros((len(all_new_points), 3)))  # Black color for new points

        # Create value map entries
        for i, pt in enumerate(all_new_points, start=new_point_index):
            original_poi_index = (i - new_point_index) // len(grid_points)
            value_map_entry = {
                'point_index': i, 
                'point_position': pt.tolist(), 
                'point_value': value, 
                'origin_predicate': predicate,
                'origin_predicate_index': original_poi_index,  # Record the index of the original POI
                'origin_predicate_position': pois[original_poi_index].tolist()  # Record the original POI
            }
            value_map.append(value_map_entry)
        new_point_index += len(all_new_points)

    # Concatenate all new points and colors
    updated_points = np.vstack([original_points] + new_points_list)
    updated_colors = np.vstack([np.asarray(point_cloud.colors)] + new_colors_list)

    # Update the point cloud
    point_cloud.points = o3d.utility.Vector3dVector(updated_points)
    point_cloud.colors = o3d.utility.Vector3dVector(updated_colors)
    return point_cloud, value_map

#3d grid of points within a unit sphere for 3d moving robots like drones
def generate_unit_sphere_grid(radius, step_factor):
    # Generate a dense grid of points and filter out those that are outside the sphere
    step = radius / step_factor  # Adjust step size for denser or sparser points
    x, y, z = np.mgrid[-radius:radius:step, -radius:radius:step, -radius:radius:step]
    grid = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    sphere = grid[np.linalg.norm(grid, axis=1) <= radius]
    return sphere

#2d grid of points within a unit circle for 2d moving robots like rovers
def generate_circle_on_plane(radius, z_height, step_factor):
    # Determine the density of points based on the radius
    step = radius / step_factor

    # Create a grid of points
    x, y = np.mgrid[-radius:radius:step, -radius:radius:step]
    z = np.full_like(x, z_height)

    # Filter points to keep only those within the circle's radius
    mask = x**2 + y**2 <= radius**2
    circle_points = np.vstack([x[mask], y[mask], z[mask]]).T
    return circle_points

def color_augment_value_map(point_cloud, value_map, use_original_colors=False, h_min=-3, h_max=-0.15):
    point_cloud = copy.copy(point_cloud)

    # Define colors
    color_avoidance = [1, 0, 0]  # Red for value -1
    color_allowed = [0, 1, 0]  # Green for value 0
    color_goal = [1, 0.843, 0]  # Gold for goal color

    # Initialize color array
    if use_original_colors and point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)
    else:
        base_color = [0.5, 0.5, 0.5]  # Gray color for base
        colors = np.array([base_color for _ in range(len(point_cloud.points))])

    # Assign colors based on value_map
    for point_info in value_map:
        point_index = point_info['point_index']
        point_value = point_info['point_value']

        if point_value == 0:
            colors[point_index] = color_allowed
        elif point_value == -1:
            colors[point_index] = color_avoidance
        elif point_value == 1:
            colors[point_index] = color_goal

    # Update the point cloud colors
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

#function to convert grid coordinates back to point world coordinates
def grid_to_world(grid_coords, min_bound, resolution):
    return grid_coords * resolution + min_bound

#function to convert point world coordinates to grid coordinates
def world_to_grid(world_coords, min_bound, resolution):
    return np.floor((world_coords - min_bound) / resolution).astype(int)

def generate_obstacle_map(pcd, pred_grouped_goal_points, resolution=0.01, h_min=-3, h_max=-0.15):
    # Convert Open3D.PointCloud to numpy array
    points = np.asarray(pcd.points)

    # Filter out points that are outside the height limits
    height_mask = (points[:, 2] > h_min) & (points[:, 2] < h_max)
    points_filtered = points[height_mask]

    # Project the filtered points to a 2D plane
    points_2d = points_filtered[:, :2]

    # Compute the bounds for the grid and allocate space
    if pred_grouped_goal_points!= None:
        all_goal_points = np.concatenate([np.asarray(points) for points in pred_grouped_goal_points.values()])
        min_bound = np.minimum(points_2d.min(axis=0), all_goal_points[:, :2].min(axis=0))
        max_bound = np.maximum(points_2d.max(axis=0), all_goal_points[:, :2].max(axis=0))
    else:
        min_bound = points_2d.min(axis=0)
        max_bound = points_2d.max(axis=0)
    grid_size = np.ceil((max_bound - min_bound) / resolution).astype(int)
    obstacle_map = np.zeros(grid_size, dtype=np.uint8)

    # Convert the points to grid coordinates
    grid_coords = np.floor((points_2d - min_bound) / resolution).astype(int)
    np.add.at(obstacle_map, (grid_coords[:, 0], grid_coords[:, 1]), 1)

    # Convert obstacle cells to 1 and free cells to 0
    obstacle_map = (obstacle_map > 0).astype(np.uint8)

    # Create a dictionary to hold goal point information
    goal_point_info = {}

    # Project and add grouped goal points to the dictionary
    if pred_grouped_goal_points != None:
        for origin_key, goal_points in pred_grouped_goal_points.items():
            goal_points_2d = np.floor((np.asarray(goal_points)[:, :2] - min_bound) / resolution).astype(int)
            for point in goal_points_2d:
                if obstacle_map[point[0], point[1]] != 1:  # Check if not an obstacle
                    obstacle_map[point[0], point[1]] = 2   # Mark as goal point
                    goal_point_info[(point[0], point[1])] = origin_key  # Store origin key in separate dictionary

        # Invert the obstacle map to represent free space as 1 and obstacles as 0
        free_space_map = 1 - (obstacle_map == 1).astype(np.uint8)
        # Ensure goal points remain represented in the map
        free_space_map[obstacle_map == 2] = 2
    else:
        free_space_map = 1 - (obstacle_map == 1).astype(np.uint8)

    # Crop the map to only include the area with obstacles and goal points
    x_indices, y_indices = np.nonzero(free_space_map)
    rmin, rmax = x_indices.min(), x_indices.max()
    cmin, cmax = y_indices.min(), y_indices.max()
    cropped_obstacle_map = free_space_map[rmin:rmax+1, cmin:cmax+1]

    return cropped_obstacle_map, goal_point_info, min_bound, max_bound
