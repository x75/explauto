import numpy as np

import sys

from .sensorimotor_model import SensorimotorModel
from explauto.utils import bounds_min_max

sys.path.insert(0, "/home/src/QK/smp/imol")

from imol.models import LinearNetwork, ReservoirNetwork

class LinearNetworkEHModel(SensorimotorModel):
    """Use single hidden layer feedforward random (non-)liner expansion and learn linear combination readout with exploratory differential Hebbian learning"""
    def __init__(self, , conf, modelsize = 100, sigma_explo_ratio=0.1, alpha = 1.0, eta = 1e-3, sigma_state = 1e-2, input_scaling = 1.0,
                 g = 0.0, tau = 0.1, mtau = False, bias_scaling = 0.0, explo_noise_type = "gaussian")
        SensorimotorModel.__init__(self, conf)
        # aha
        for attr in ['m_ndims', 's_ndims', 'm_dims', 's_dims', 'bounds', 'm_mins', 'm_maxs']:
            setattr(self, attr, getattr(conf, attr))

        self.mode = "explore"
        mfeats = tuple(range(self.m_ndims))
        sfeats = tuple(range(-self.s_ndims, 0))
        mbounds = tuple((self.bounds[0, d], self.bounds[1, d]) for d in range(self.m_ndims))
        print "mfeats", mfeats, "sfeats", sfeats

        self.modelsize = modelsize
        self.g = g

        self.fmodel = Identity
        self.fmodel_context = Identity
        self.imodel = ReservoirNetwork
        # oder
        self.imodel = Reservoir

    def infer(self, in_dims, out_dims, x):
        if in_dims == self.m_dims and out_dims == self.s_dims:  # forward
            return np.random.uniform(-1, 1, (len(in_dims),))
        elif in_dims == self.s_dims and out_dims == self.m_dims:  # inverse
            return np.random.uniform(-1, 1, (len(in_dims),))
        elif out_dims == self.m_dims[len(self.m_dims)/2:]:  # dm = i(M, S, dS)
            return np.random.uniform(-1, 1, (len(in_dims),))
        else:
            raise NotImplementedError

    def predict_given_context(self, x, c, c_dims):
        return np.random.uniform(-1, 1, (self.fmodel_context.odim,))

    def update(self, m, s):
        # pass
        # if better than before
        # amplify state/output/reward tuple


configurations = {
    "default": {
        "modelsize": 500,
        "sigma_explo_ratio": 0.1,
        "theta_state": 1e-2,
        "input_scaling": 1.0, # check integer ;)
        "alpha": 1.0
        },
    }
