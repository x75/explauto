import numpy as np

import sys

from .sensorimotor_model import SensorimotorModel
from explauto.utils import bounds_min_max

sys.path.insert(0, "/home/src/QK/smp/imol")

from imol.models import LinearNetwork, ReservoirNetwork

class LinearNetworkFORCEModel(SensorimotorModel):
    """Use single hidden layer feedforward random (non-)liner expansion and learn linear combination readout with online regression

    (class is called LinearNetwork but it can even be nonlinear)"""
    def __init__(self, conf, modelsize = 100, sigma_explo_ratio=0.1, alpha = 1.0, eta = 1e-3, theta_state = 1e-2, input_scaling = 1.0,
                 g = 0.0, tau = 0.1, mtau = False, bias_scaling = 0.0, explo_noise_type = "gaussian"):
        SensorimotorModel.__init__(self, conf)
        for attr in ['m_ndims', 's_ndims', 'm_dims', 's_dims', 'bounds', 'm_mins', 'm_maxs']:
            setattr(self, attr, getattr(conf, attr))

        self.mode = 'explore'
        mfeats = tuple(range(self.m_ndims))
        sfeats = tuple(range(-self.s_ndims, 0))
        mbounds = tuple((self.bounds[0, d], self.bounds[1, d]) for d in range(self.m_ndims))
        print "mfeats", mfeats, "sfeats", sfeats

        self.modelsize = modelsize
        self.g = g
        
        if self.g > 0.0:
            # this is the forward model standard edition
            self.fmodel = ReservoirNetwork(modelsize = self.modelsize, idim = len(mfeats), odim = len(sfeats), alpha = alpha,
                                    eta = eta, theta_state = theta_state, input_scaling = input_scaling, g = g, tau = tau, mtau = mtau,
                                    bias_scaling = bias_scaling)
            # this is the forward model considering context
            self.fmodel_context = ReservoirNetwork(modelsize = self.modelsize, idim = len(sfeats)/2 + len(mfeats), odim = len(sfeats), alpha = alpha,
                                    eta = eta, theta_state = theta_state, input_scaling = input_scaling, g = g, tau = tau, mtau = mtau,
                                    bias_scaling = bias_scaling)
            # this is the inverse model, this works straightaway with context information
            self.imodel = ReservoirNetwork(modelsize = self.modelsize, idim = len(sfeats), odim = len(mfeats), alpha = alpha,
                                    eta = eta, theta_state = theta_state, input_scaling = input_scaling, g = g, tau = tau, mtau = mtau,
                                    bias_scaling = bias_scaling)
        else:
            # this is the forward model standard edition
            self.fmodel = LinearNetwork(modelsize = self.modelsize, idim = len(mfeats), odim = len(sfeats), alpha = alpha,
                                    eta = eta, theta_state = theta_state, input_scaling = input_scaling,
                                    bias_scaling = bias_scaling)
            # this is the forward model considering context
            self.fmodel_context = LinearNetwork(modelsize = self.modelsize, idim = len(sfeats)/2 + len(mfeats), odim = len(sfeats), alpha = alpha,
                                    eta = eta, theta_state = theta_state, input_scaling = input_scaling,
                                    bias_scaling = bias_scaling)
            # this is the inverse model, this works straightaway with context information
            self.imodel = LinearNetwork(modelsize = self.modelsize, idim = len(sfeats), odim = len(mfeats), alpha = alpha,
                                    eta = eta, theta_state = theta_state, input_scaling = input_scaling,
                                    bias_scaling = bias_scaling)

        print "models", self.g, self.fmodel, self.fmodel_context, self.imodel
            
        self.ffiterr = np.zeros((self.s_ndims,))        
        self.ffiterr_context = np.zeros((self.s_ndims,))
        self.ifiterr = np.zeros((self.m_ndims,))

        self.explo_noise_type = explo_noise_type
        self.sigma_expl = ((conf.m_maxs - conf.m_mins) * float(sigma_explo_ratio)).reshape((self.imodel.odim,1))
        print "self.sigma_expl", self.sigma_expl.shape

        # self.stats = (np.array([ 0.05330568]), np.array([ 0.32410747]), np.array([ 0.00865745]), np.array([ 0.56716436]))
        self.stats = (np.array([ 0.05330568]), np.array([ 1.0]), np.array([ 0.00865745]), np.array([ 1.0]))

    def infer(self, in_dims, out_dims, x):
        # if self.t < max(self.model.imodel.fmodel.k, self.model.imodel.k):
        #     raise ExplautoBootstrapError

        if in_dims == self.m_dims and out_dims == self.s_dims:  # forward
            # print "x", x
            x = np.array([x]).T
            # y, h = self.fmodel_context.predict(x)
            y, h = self.fmodel.predict(x)
            # print "y", y
            return y.reshape((self.fmodel.odim,))
            # return np.random.uniform(-1, 1, (len(out_dims),))
        elif in_dims == self.s_dims and out_dims == self.m_dims:  # inverse
            # print "x", x
            # print "inverse", len(in_dims), in_dims, out_dims
            x = np.array([x]).T / self.stats[1]
            # print "x.shape", x.shape#, "y", y
            y, h = self.imodel.predict(x)
            y *= self.stats[3]
            # y = np.clip(y, 0.2, 0.8)
            # print self.__class__.__name__, "infer: y pre", y
            if self.mode == "explore":
                # print self.__class__.__name__, "infer: explore: y.shape", y.shape, type(self.sigma_expl)
                # n = np.random.normal(y, 1) * self.sigma_expl
                # gaussian exploration noise
                if self.explo_noise_type == "gaussian":
                    n = np.random.normal(0, 1, y.shape) * self.sigma_expl
                    print "gaussain noise", n
                # pareto exploration noise
                elif self.explo_noise_type == "pareto":
                    # if  > 0:
                    n = ((np.random.binomial(1, 0.5) - 0.5) * 2) * np.random.pareto(1.5, y.shape) * self.sigma_expl
                    # else:
                    # n = -1.0 * np.random.pareto(1.5, y.shape) * self.sigma_expl

                # print self.__class__.__name__, "infer: explore: n.shape", y.shape, n.shape, self.imodel.odim
                y = y.copy() + n
                # print self.__class__.__name__, "infer: y post", y.shape, y
            y_ = bounds_min_max(y.T, self.m_mins, self.m_maxs)
            # print self.__class__.__name__, "infer: y bound", y_.shape, y_
            return y_.reshape((self.imodel.odim,))
                
        elif out_dims == self.m_dims[len(self.m_dims)/2:]:  # dm = i(M, S, dS)
            return np.random.uniform(-1, 1, (len(in_dims),))
        else:
            raise NotImplementedError

    def predict_given_context(self, x, c, c_dims):
        """predict sensory consequence (s_{t+1} given sensory / environmental context (s_t) and motor (m_t)"""
        x_ = np.array([c + x]).T
        # print "x, c, c_dims", x, c, c_dims, x_
        y, h = self.fmodel_context.predict(x_)
        # return np.zeros((self.fmodel_context.odim, ))
        return y.reshape((self.fmodel_context.odim,))
        
    def update(self, m, s):
        # print self.__class__.__name__, "update(m, s)", m, s, self.imodel.y
        s_t   = np.array([s[:self.fmodel_context.idim/2]]).T / self.stats[1] # context
        s_tp1 = np.array([s[self.fmodel_context.idim/2:]]).T / self.stats[1] # current sensory state
        m_    = np.array([m]).T / self.stats[3]
        # , self.fmodel_context.idim, self.fmodel_context.idim, 
        # print self.__class__.__name__, "update: prepared inputs", "s_t s_tp1, m_", s_t, s_tp1, m_
        X_fwd = m_.copy()
        Y_fwd = np.vstack((s_t, s_tp1))
                
        X_fwd_context = np.vstack((s_t, m_))
        Y_fwd_context = np.vstack((s_t, s_tp1)) # np.array([m]).T
        # X_inv_ = np.array([s]).T
        X_inv = np.vstack((s_t, s_tp1))
        # print "X_inv eq?", X_inv_ == X_inv
        Y_inv = m_
        # print "X_fwd", X_fwd, "Y_fwd", Y_fwd
        # print "X_inv", X_inv, "Y_inv", Y_inv, self.imodel.y
        self.ffiterr         = self.fmodel.fitFORCE(X_fwd, Y_fwd)
        self.ffiterr_context = self.fmodel_context.fitFORCE(X_fwd_context, Y_fwd_context)
        self.ifiterr         = self.imodel.fitFORCE(X_inv, Y_inv, reverse = False, activate = True)
        # self.ifiterr = self.imodel.fitRLS(X_inv, Y_inv)
        # self.ifiterr = self.imodel.fit(X_inv, Y_inv)
        # print("fiterr f, i", ffiterr, ifiterr, np.linalg.norm(self.imodel.W_o, 2))
        # return ffiterr, ifiterr
        
configurations = {
    # works: size 100, sigma 0.1, iscale 1e0, seed 2
    "default": {
        # "modelsize": 1000,
        "modelsize": 600,
        # "modelsize": 300,
        # "modelsize": 100,
        # "sigma_explo_ratio": 0.8, # 0.8 yields best results so far
        "sigma_explo_ratio": 0.1,   # point mass yeah!!! should also work with 0.3 or less, let's try
        # "sigma_explo_ratio": 0.2,   # morse coptershould also work with 0.3 or less, let's try
        "theta_state": 1e-4,
        # "theta_state": 1e-2,
        # "eta": 1e-2,
        # "input_scaling": 5e-2,
        # "input_scaling": 1e-1,
        "input_scaling": 1,
        "alpha": 1.0
    },
    "default_pm": {
        # "modelsize": 1000,
        "modelsize": 600,
        # "modelsize": 300,
        # "modelsize": 100,
        # "explo_noise_type": "gaussian",
        "explo_noise_type": "pareto",
        # "sigma_explo_ratio": 0.8, # 0.8 yields best results so far
        "sigma_explo_ratio": 0.1,   # point mass yeah!!! should also work with 0.3 or less, let's try
        # "sigma_explo_ratio": 0.2,   # morse coptershould also work with 0.3 or less, let's try
        "theta_state": 1e-4,
        # "theta_state": 1e-2,
        # "eta": 1e-2,
        # "input_scaling": 5e-2,
        # "input_scaling": 1e-1,
        "input_scaling": 1.0,
        "bias_scaling": 0.1,
        "alpha": 1.0
    },
    "recurrent_default_pm": {
        # "modelsize": 1000,
        "modelsize": 600,
        # "modelsize": 300,
        # "modelsize": 100,
        # "sigma_explo_ratio": 0.8, # 0.8 yields best results so far
        "explo_noise_type": "gaussian",
        # "explo_noise_type": "pareto",
        "sigma_explo_ratio": 0.1,   # point mass yeah!!! should also work with 0.3 or less, let's try
        # "sigma_explo_ratio": 0.2,   # morse coptershould also work with 0.3 or less, let's try
        "theta_state": 1e-2,
        "g": 0.99,
        "tau": 0.8,
        # "theta_state": 1e-2,
        # "eta": 1e-2,
        # "input_scaling": 5e-2,
        # "input_scaling": 1e-1,
        "input_scaling": 1.0,
        "alpha": 1.0
    },
    "default_morse_copter": {
        # "modelsize": 1000,
        "modelsize": 600,
        # "modelsize": 300,
        # "modelsize": 100,
        # "explo_noise_type": "gaussian",
        "explo_noise_type": "pareto",
        # "sigma_explo_ratio": 0.8, # 0.8 yields best results so far
        # "sigma_explo_ratio": 0.1,   # point mass yeah!!! should also work with 0.3 or less, let's try
        # "sigma_explo_ratio": np.array([0.1, 0.1, 0.1, 0.2]),
        "sigma_explo_ratio": 0.1,   # morse coptershould also work with 0.3 or less, let's try
        "theta_state": 1e-2,
        # "theta_state": 1e-2,
        # "eta": 1e-2,
        # "input_scaling": 5e-2,
        # "input_scaling": 1e-1,
        "input_scaling": 0.5,
        "bias_scaling": 0.2,
        "alpha": 1.0
    },
    "recurrent_morse_copter": {
        # "modelsize": 1000,
        "modelsize": 600,
        # "modelsize": 300,
        # "modelsize": 100,
        # "sigma_explo_ratio": 0.8, # 0.8 yields best results so far
        # "sigma_explo_ratio": 0.1,   # point mass yeah!!! should also work with 0.3 or less, let's try
        # "sigma_explo_ratio": np.array([0.1, 0.1, 0.1, 0.2]),
        "sigma_explo_ratio": 0.2,   # morse coptershould also work with 0.3 or less, let's try
        "theta_state": 1e-2,
        "g": 0.7,
        "tau": 0.5,
        # "theta_state": 1e-2,
        # "eta": 1e-2,
        # "input_scaling": 5e-2,
        # "input_scaling": 1e-1,
        "input_scaling": 0.5 * 5,
        "bias_scaling": 0.3,
        "alpha": 1.0
    },
    "medium": {
        "modelsize": 300,
        "sigma_explo_ratio": 0.01,
    }
}
sensorimotor_models = {'LN-FORCE': (LinearNetworkFORCEModel, configurations)}
