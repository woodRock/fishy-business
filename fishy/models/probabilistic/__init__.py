# -*- coding: utf-8 -*-
from .gp import GaussianProcess
from .probabilistic_svc import ProbabilisticSVC
from .laplace_logistic import BayesianLogisticRegression

__all__ = ["GaussianProcess", "ProbabilisticSVC", "BayesianLogisticRegression"]
