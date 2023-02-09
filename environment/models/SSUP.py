import numpy as np
import math
from scipy import stats
import torch
from torch import distributions as D
from collections import OrderedDict
torch.autograd.set_detect_anomaly(True)
from pyGameWorld.viewer import demonstrateTPPlacement

class SSUP:

    def __init__(
        self, 
        objects, 
        tools, 
        toolpicker, 
        task_type,
        goal_pos,
        goal_key,
        objInGoalKeys,
        std_y=200, 
        std_x=10, 
        n_init=3,
        n_sims=4, 
        epsilon=0.05,  
        collision_direction_std=0.1, 
        collision_elasticity_std=0.1) -> None:

        self.objects = objects # list of objects
        self.tools = OrderedDict(tools) # dict of tools with name as key and tool vertices as value
        self.task_type = task_type
        self.goal_key = goal_key
        self.objInGoalKeys = objInGoalKeys
        self.idx_to_tool = {i: tool for i, tool in enumerate(self.tools.keys())}
        self.tool_to_idx = {tool: i for i, tool in enumerate(self.tools.keys())}
        self.std_y = std_y
        self.std_x = std_x
        self.n_init = n_init
        self.n_sims = n_sims
        self.collision_direction_std = collision_direction_std
        self.collision_elasticity_std = collision_elasticity_std
        self.tp = toolpicker
        self.goal_pos = goal_pos
        self.prior = None
        self.epsilon = epsilon
        self.lr = 0.25
        self.gmm_var = 50
        self.K=3
        self.D=2

        # initialize policy parameters
        # locs are the means of the gaussian mixture which should be initialized at the center of the game world
        self.locs = torch.tile(torch.tensor([300.]), (self.K, self.D)).requires_grad_(True)

        # coord_scale is the standard deviation of the gaussian mixture, should be isotropic with variance of 50px
        coord_scale = torch.tile(torch.diag(torch.ones(2) * self.gmm_var), (self.K, 1)).reshape((self.K, self.D, self.D))
        self.coord_scale = coord_scale.requires_grad_(True)

        # component_logits is the probability of each component in the mixture
        # initialized to be uniform
        component_logits = torch.ones(self.K)
        self.component_logits = component_logits.requires_grad_(True)

        # initialize optimizer
        self.optimizer = torch.optim.SGD(params=[self.component_logits, self.locs, self.coord_scale], lr=self.lr)

    def object_oriented_prior(self, toolname):
        """
        Sample an action from the object oriented prior for the given tool
        """
        tool = self.tools[toolname]
        # TODO: learn p conditioned on tools P(object|tool) -- start with enumeration   
        object_idx = np.random.choice(np.arange(len(self.objects)), p=np.ones(len(self.objects))/len(self.objects))
        object = self.objects[object_idx]
        tool_BB = self.get_bounding_box(tool)
        height = tool_BB[3] - tool_BB[1]
        loc = object.getPos()[1]
        a, b = ((0+height/2) - loc) / self.std_y, ((600-height/2) - loc) / self.std_y

        # need to check that sampled position is not colliding with the object
        collides = True
        while collides:
            pos_y = stats.truncnorm.rvs(a, b, loc=loc, scale=self.std_y)
            if object.type == 'Ball':
                BB_left = object.getPos()[0] - object.radius
                BB_right = object.getPos()[0] + object.radius
            else:
                BB = self.get_bounding_box(object.toGeom())
                BB_left = object.getPos()[0] - np.abs(BB[0]) 
                BB_right = object.getPos()[0] + BB[2]

            v = stats.uniform.rvs(loc=BB_left-self.std_x, scale=BB_right+self.std_x)
            if v < BB_left:
                pos_x = stats.norm.rvs(loc=BB_left, scale=self.std_x)
            elif v > BB_right:
                pos_x = stats.norm.rvs(loc=BB_right, scale=self.std_x)
            else:
                pos_x = v
            collides = self.tp.checkPlacementCollide(toolname, [pos_x, pos_y])
        
        return object_idx, [pos_x, pos_y]

    def initialize(self):
        """
        For each tool, sample n_init initial actions, compute the noisy reward and update the policy parameters
        """
        prior = {}
        for idx, tool in enumerate(self.tools.keys()):
            prior[tool] = []
            for _ in range(self.n_init):
                obj, pos = self.object_oriented_prior(tool)
                noisy_reward = self.simulate((tool, pos))
                tool_dist = D.Categorical(logits=self.component_logits)
                log_prob_tool = tool_dist.log_prob(torch.tensor(idx))
                pos_dist = D.MultivariateNormal(loc=self.locs[idx], scale_tril=self.coord_scale[idx])
                log_prob_pos = pos_dist.log_prob(torch.tensor(pos))
                self.update((log_prob_tool, log_prob_pos), noisy_reward)
                prior[tool].append((tool, pos))
        self.prior = prior

    def sample_prior(self):
        """
        Sample an action from the prior
        """
        toolList = list(self.tools.keys())
        tool = np.random.choice(np.arange(len(self.tools)), p=np.ones(len(self.tools))/len(self.tools))
        toolname = toolList[tool]
        object_idx, pos = self.object_oriented_prior(toolname)
        tool_dist = D.Categorical(logits=self.component_logits)
        log_prob_tool = tool_dist.log_prob(torch.tensor(tool))
        pos_dist = D.MultivariateNormal(loc=self.locs[tool], scale_tril=self.coord_scale[tool])
        log_prob_pos = pos_dist.log_prob(torch.tensor(pos))

        return tool, pos, log_prob_tool, log_prob_pos, 'prior'
    
    def simulate(self, action):
        """
        Simulate the action and return the reward
        """
        reward = 0
        toolname, pos = action
        for _ in range(self.n_sims):
            path_dict, success, _ = self.tp.runNoisyPath(
            toolname=toolname, 
            position=pos, 
            maxtime=20., 
            noise_collision_direction=self.collision_direction_std, 
            noise_collision_elasticity=self.collision_elasticity_std
        )   
            if success:
                reward += 1
            else:
                reward = self.reward(path_dict)
        return reward/self.n_sims

    def update(self, log_probs, reward):
        """
        Update the policy using policy gradient
        """
        # compute the gradient of the reward with respect to the policy parameters
        self.optimizer.zero_grad()
        cost = -reward*(log_probs[0] + log_probs[1])
        cost.backward()
        self.optimizer.step()

    def reward(self, path_dict):
        """
        Compute the reward for the action. The reward is the minimum distance between the tool and the object reached during the trajectory 
        and normalized by the distance between the tool and the object at the start of the trajectory
        """
        reward = 0
        key = None
        for obj in self.objInGoalKeys:
            trajectory = np.array(path_dict[obj])
            distances = np.linalg.norm(trajectory[1:] - self.goal_pos, axis=1)
            min_dist_traj = np.min(distances)
            init_dist = distances[0]
            d = min_dist_traj/init_dist
            if 1-d > reward:
                reward = 1-d
        return reward
    
    def getLogProbs(self, action):
        """
        Compute the log probability of the action
        """
        toolname, pos = action
        tool_dist = D.Categorical(logits=self.component_logits)
        tool_idx = torch.tensor(self.tool_to_idx[toolname])
        log_prob_tool = tool_dist.log_prob(tool_idx)
        pos_dist = D.MultivariateNormal(loc=self.locs[tool_idx], scale_tril=self.coord_scale[tool_idx])
        log_prob_pos = pos_dist.log_prob(torch.tensor(pos))
        return log_prob_tool, log_prob_pos

    def counterfactual_simulation(self, action):
        """
        Simulate a counterfactual action and return the reward
        """
        reward = 0
        toolname, pos = action
        for _ in range(self.n_sims):
            path_dict, success, _ = self.tp.runNoisyPath(
            toolname=toolname, 
            position=pos, 
            maxtime=20., 
            noise_collision_direction=self.collision_direction_std, 
            noise_collision_elasticity=self.collision_elasticity_std
        )   
            if success:
                reward += 1
            else:
                reward = self.reward(path_dict)

        tool_dist = D.Categorical(logits=self.component_logits)
        tool_idx = torch.tensor(self.tool_to_idx[toolname])
        log_prob_tool = tool_dist.log_prob(tool_idx)
        pos_dist = D.MultivariateNormal(loc=self.locs[tool_idx], scale_tril=self.coord_scale[tool_idx])
        log_prob_pos = pos_dist.log_prob(torch.tensor(pos))
        return reward/self.n_sims, log_prob_tool, log_prob_pos
    
    def sample_policy(self):
        """
        Sample an action from the policy
        """
        tool_dist = D.Categorical(logits=self.component_logits)
        tool = tool_dist.sample()
        collides = True
        while collides:
            pos_dist = D.MultivariateNormal(loc=self.locs[tool], scale_tril=self.coord_scale[tool])
            pos = pos_dist.sample()
            cur_pos = pos.detach().numpy()
            collides = self.tp.checkPlacementCollide(toolname=self.idx_to_tool[tool.item()], position=(float(cur_pos[0]), float(cur_pos[1])))
        log_prob_tool = tool_dist.log_prob(tool)
        log_prob_pos = pos_dist.log_prob(pos)

        return tool.item(), (float(cur_pos[0]), float(cur_pos[1])), log_prob_tool, log_prob_pos, 'policy'

    def sample_action(self):
        """
        Sample an action from the policy with probability 1-epsilon, or from the prior with probability epsilon
        """
        if np.random.rand() < self.epsilon:
            return self.sample_prior()
        else:
            return self.sample_policy()

    def get_bounding_box(self, vertices):
        verts = np.array(vertices)
        x_min, x_max = np.min(verts[:,0]), np.max(verts[:,0])
        y_min, y_max = np.min(verts[:,1]), np.max(verts[:,1])
        return np.array([x_min, y_min, x_max, y_max])
