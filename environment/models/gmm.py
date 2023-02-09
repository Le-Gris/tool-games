from pyro.distributions import MixtureOfDiagNormals 
from pyro.distributions.diag_normal_mixture import _MixDiagNormalSample
import torch
from torch import nn


class GMM(MixtureOfDiagNormals):

    def __init__(self, locs, coord_scale, component_logits):
        super().__init__(locs=locs, coord_scale=coord_scale, component_logits=component_logits)

    def rsample(self, sample_shape=torch.Size()):
        which = self.categorical.sample(sample_shape)
        return _MixDiagNormalSample.apply(
            self.locs,
            self.coord_scale,
            self.component_logits,
            self.categorical.probs,
            which,
            sample_shape + self.locs.shape[:-2] + (self.dim,),
        ), which



