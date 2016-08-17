import numpy as np

from ..utils import rand_bounds
from .interest_model import InterestModel
from .random import RandomInterest

class RandomInterestDynamic(RandomInterest):
    """Like RandomInterest but only sample new goal after _interval_ steps

    for dynamic environments when you can't reach the goal in one step
    """
    def __init__(self, conf, expl_dims, interval = 100):
        RandomInterest.__init__(self, conf, expl_dims)

        self.cnt = 0
        self.interval = interval
        self.current_goal = rand_bounds(self.bounds).flatten()

    def sample(self):
        if self.cnt % self.interval == 0:
            self.current_goal = rand_bounds(self.bounds).flatten()
        self.cnt += 1
        return self.current_goal

class RandomInterestDynamicCopter(RandomInterest):
    def __init__(self, conf, expl_dims, interval = 100, goal_binary = False, lattime = 20):
        RandomInterest.__init__(self, conf, expl_dims)

        self.cnt = 0
        self.interval = interval

        self.goal_binary = goal_binary

        self.lattime = lattime

        self.more_context = True # HACK

        self.r_g     = [1, 1, 1] # at origin, 1m alt
        self.r_g_yaw = np.pi * 0.0
        self.s_g     = [0, 0, -0.3, 1, 0] #

    def sample_given_more_context(self, c, c_dims, x_):
        if self.cnt % self.interval == 0:
            self.r_g     = self.generate_real_goal()
            self.r_g_yaw = self.generate_real_goal_yaw() #
        self.cnt += 1
        self.s_g = self.compute_sensory_goal_from_real_goal(x_)
        return self.s_g

    def compute_sensory_goal_from_real_goal(self, x_):
        # z (altitude) component
        if self.goal_binary:
            if x_[2] > self.r_g[2]:
                self.s_g[2] = 0.3
            else:
                self.s_g[2] = -0.3
        else: # continuous goal
            self.s_g[2] = x_[2] - self.r_g[2]            

        # lateral, with scheduling
        if self.cnt > self.lattime:
            if self.goal_binary:
                if x_[0] > self.r_g[0]:
                    self.s_g[0] = 0.2
                else:
                    self.s_g[0] = -0.2
                if x_[1] > self.r_g[1]:
                    self.s_g[1] = 0.2
                else:
                    self.s_g[1] = -0.2
            else:
                self.s_g[0] = x_[0] - self.r_g[0]
                self.s_g[1] = x_[1] - self.r_g[1]
                
            # if x[3] > 0.70710678118654757:
            self.s_g[3] = np.cos(self.r_g_yaw)
            self.s_g[4] = np.sin(self.r_g_yaw) + 1e-6
            print("yaw goal", self.s_g[3], self.s_g[4])

            # if self.cnt % 100 == 0:
            #     r_g = generate_real_goal(r_g)
            #     ang = generate_real_goal_yaw() #
            #     print("new goal = %s" % r_g)

        # # scale goal to avoid overshoot?
        # for j in range(3):
        #     self.s_g[j] *= -0.8

        # Draw a goal given this context
        # self.s_g = list(im_model.sample_given_context(context, range(context_mode["context_n_dims"])))
        # TODO: if pos.z != setpoint, self.s_g = vel towards setpoint
        print("self.s_g", type(self.s_g), self.s_g, self.r_g)
        return self.s_g
        
    def generate_real_goal(self):
        x_goals = [1.0, -1.0, 0.0]
        y_goals = [1.0, -1.0, 0.0]
        z_goals = [1.0,  1.5, 2.0]
        r_g = [np.random.choice(x_goals), np.random.choice(y_goals), np.random.choice(z_goals)]
        return r_g

    def generate_real_goal_yaw(self):
        yaw_goals = [np.pi * f for f in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]] # [0.0, 0.5, 1.0, -0.5]]
        return np.random.choice(yaw_goals)
    
interest_models = {'random_dyn': (RandomInterestDynamic, {'default': {'interval': 100}}),
                   'random_dyn_copter': (RandomInterestDynamicCopter, {'default': {
                       'interval': 100,
                       'goal_binary': False,
                       'lattime': 20
                       }})}
