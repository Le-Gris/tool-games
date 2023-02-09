from pyGameWorld import PGWorld, ToolPicker, loadFromDict, loadToolPicker
from pyGameWorld.viewer import demonstrateTPPlacement
import json
import pygame as pg
from models.SSUP import SSUP
import numpy as np
import argparse
import os

# parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='environment/Trials/Original')
    parser.add_argument('--tnm', type=str, default='Catapult')
    args = parser.parse_args()
    return args


# helper function
def verifyDynamicObjects(objects, trial):
    # check if dict trial['world']['blocks'] is empty
    if not trial['world']['blocks']:
        return objects
    else:
        blocker_verts = trial['world']['blocks']['Blocker']['vertices']
    # remove objects that are blocked from list
    new_objs = []
    for obj in objects:
        obj_pos = obj.getPos()
        if obj_pos[0] < blocker_verts[0][0] or obj_pos[0] > blocker_verts[2][0] or obj_pos[1] < blocker_verts[0][1] or obj_pos[1] > blocker_verts[2][1]:
            new_objs.append(obj)
    return new_objs

# setup world
def setup_world(task_dir):
    with open(task_dir + ".json", 'r') as jfl:
        trial = json.load(jfl)
    pgw = loadFromDict(trial['world'])
    tp = loadToolPicker(task_dir + ".json")
    return trial, pgw, tp

# get SSUP model object
def init_ssup(trial, pgw, tp):
    objects = verifyDynamicObjects(pgw.getDynamicObjects(), trial=trial)
    print('There are {} movable objects in the world: '.format(len(objects)), [obj.name for obj in objects])
    tools = tp._tools
    goal_key = trial['world']['gcond']['goal']
    goal_pos = pgw.objects[goal_key].getPos()
    task_type = trial['world']['gcond']['type']
    if task_type == 'SpecificInGoal':
        objInGoalKeys = ['Ball']
    elif task_type == 'ManyInGoal':
        objInGoalKeys = trial['world']['gcond']['objlist']
    model = SSUP(objects=objects, tools=tools, toolpicker=tp, goal_pos=goal_pos, task_type=task_type, goal_key=goal_key, objInGoalKeys=objInGoalKeys, epsilon=0.05)
    return model

def run_model(model):
    # Sample an initial action for each tool (sample init in SSUP) and compute noisy rewards
    model.initialize()

    # outer loop variables
    successful = False
    n_iters = 5
    i = 0
    best_action = None
    last_reward = 0
    T = 0.5

    # run until task is solved
    while not successful:
        acting = False
        tool, pos, log_prob_tool, log_prob_pos, act_type = model.sample_action()
        toolname = model.idx_to_tool[tool]
        noisy_reward = model.simulate((toolname, pos))
        i += 1
        if noisy_reward > last_reward:
            best_action = (toolname, pos, act_type)
            last_reward = noisy_reward
        if noisy_reward > T:
            print("Action with a reward above threshold was found, trying action")
            acting = True
            path_dict, successful, _ = model.tp.observePlacementPath(toolname=toolname, position=pos)
            if path_dict is None:
                print("Action was not successful")
                continue
            demonstrateTPPlacement(model.tp, toolname, pos, hz=80.)
        elif i >= n_iters:
            print("No action with a reward above threshold was found, trying best action")
            acting = True
            toolname, pos, act_type = best_action
            i = 0
            last_reward = 0
            best_action = None #TODO verify that this should be reset (i.e., shouldn't be trying the same best action over and over again if it's not successful)
            print("Trying action: ", toolname, pos, act_type)
            log_prob_tool, log_prob_pos = model.getLogProbs((toolname, [pos[0], pos[1]]))
            path_dict, successful, _ = model.tp.observePlacementPath(toolname=toolname, position=pos)
            if path_dict is None:
                print("Action was not successful")
                continue
            demonstrateTPPlacement(model.tp, toolname, pos, hz=80.)
        if acting:
            if successful:
                print("Action was successful! ", successful)
                break
            reward = model.reward(path_dict)
            model.update((log_prob_tool, log_prob_pos), reward)
            for t in model.tools.keys():
                if t != toolname:
                    noisy_reward, log_prob_tool, log_prob_pos = model.counterfactual_simulation((t, pos))
                    model.update((log_prob_tool, log_prob_pos), noisy_reward)
                
        else:
            model.update((log_prob_tool, log_prob_pos), noisy_reward)


def main():
    # parse args
    args = parse_args()
    json_dir = args.dir
    tnm = args.tnm
    task_dir = os.path.join(json_dir, tnm)

    trial, pgw, tp = setup_world(task_dir=task_dir)

    model = init_ssup(trial=trial, pgw=pgw, tp=tp)



if __name__ == 'main':
    main()
