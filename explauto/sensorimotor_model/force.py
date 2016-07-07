import numpy as np

import sys

from .sensorimotor_model import SensorimotorModel
from explauto.utils import bounds_min_max

sys.path.insert(0, "/home/src/QK/smp/imol")

from imol.models import LinearNetwork

class LinearNetworkFORCEModel(SensorimotorModel):
    def __init__(self, conf, modelsize = 100, sigma_explo_ratio=0.1):
        SensorimotorModel.__init__(self, conf)
        for attr in ['m_ndims', 's_ndims', 'm_dims', 's_dims', 'bounds', 'm_mins', 'm_maxs']:
            setattr(self, attr, getattr(conf, attr))

        self.mode = 'explore'
        mfeats = tuple(range(self.m_ndims))
        sfeats = tuple(range(-self.s_ndims, 0))
        mbounds = tuple((self.bounds[0, d], self.bounds[1, d]) for d in range(self.m_ndims))
        print "mfeats", mfeats, "sfeats", sfeats

        self.sigma_expl = (conf.m_maxs - conf.m_mins) * float(sigma_explo_ratio)
        print "self.sigma_expl", self.sigma_expl
                
        self.modelsize = modelsize
        self.fmodel = LinearNetwork(modelsize = self.modelsize, idim = len(sfeats)/2 + len(mfeats), odim = len(sfeats)/2)
        self.imodel = LinearNetwork(modelsize = self.modelsize, idim = len(sfeats), odim = len(mfeats))

    def infer(self, in_dims, out_dims, x):
        # if self.t < max(self.model.imodel.fmodel.k, self.model.imodel.k):
        #     raise ExplautoBootstrapError

        if in_dims == self.m_dims and out_dims == self.s_dims:  # forward
            # print "x", x
            x = np.array([x]).T
            return self.fmodel.predict(x)
            # return np.random.uniform(-1, 1, (len(out_dims),))
        elif in_dims == self.s_dims and out_dims == self.m_dims:  # inverse
            # print "x", x
            # print "inverse", len(in_dims), in_dims, out_dims
            # r = np.random.uniform(-1, 1, (len(out_dims),))
            x = np.array([x]).T
            # print "r", r, "x", x#, "y", y
            y, h = self.imodel.predict(x)
            print self.__class__.__name__, "infer: y pre", y
            if self.mode == "explore":
                y = y.copy() + np.random.normal(y, self.sigma_expl)
            y_ = bounds_min_max(y, self.m_mins, self.m_maxs)
            print self.__class__.__name__, "infer: y post", y
            return y_.reshape((self.imodel.odim,))
        
        elif out_dims == self.m_dims[len(self.m_dims)/2:]:  # dm = i(M, S, dS)
            return np.random.uniform(-1, 1, (len(in_dims),))
        else:
            raise NotImplementedError


    def update(self, m, s):
        print self.__class__.__name__, "update(m, s)", m, s, self.imodel.y
        s_t   = np.array([s[:self.fmodel.idim/2]]).T
        s_tp1 = np.array([s[self.fmodel.idim/2:]]).T
        m_    = np.array([m]).T
        # , self.fmodel.idim, self.fmodel.idim, 
        print self.__class__.__name__, "update: prepared inputs", "s_t s_tp1, m_", s_t, s_tp1, m_
        X_fwd = np.vstack((s_t, m_))
        Y_fwd = s_tp1 # np.array([m]).T
        X_inv = np.array([s]).T
        Y_inv = m_
        print "X_fwd", X_fwd, "Y_fwd", Y_fwd
        print "X_inv", X_inv, "Y_inv", Y_inv, self.imodel.y
        ffiterr = self.fmodel.fitFORCE(X_fwd, Y_fwd)
        ifiterr = self.imodel.fitFORCE(X_inv, Y_inv)
        print("fiterr f, i", ffiterr, ifiterr, np.linalg.norm(self.imodel.W_o, 2))
        
configurations = {
    "default": {
        "modelsize": 100,
        "sigma_explo_ratio": 0.8,
    },
    "medium": {
        "modelsize": 300,
        "sigma_explo_ratio": 0.01,
    }
}
sensorimotor_models = {'LN-FORCE': (LinearNetworkFORCEModel, configurations)}
