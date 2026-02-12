# -*- coding: utf-8 -*-
import pytest
import numpy as np
from fishy.models.evolutionary.ga import GA
from fishy.models.evolutionary.eda import EDA
from fishy.models.evolutionary.es import ES
from fishy.models.evolutionary.pso import PSO
from fishy.models.evolutionary.gp import GP
from fishy.models.probabilistic.probabilistic_svc import ProbabilisticSVC
from fishy.models.probabilistic.gp import GaussianProcess

def test_evolutionary_models():
    X = np.random.randn(20, 10)
    y = np.random.randint(0, 2, 20)
    
    # GA
    ga = GA(generations=2, population_size=5)
    ga.fit(X, y)
    assert len(ga.predict(X)) == 20
    
    # EDA
    eda = EDA(generations=2, population_size=5)
    eda.fit(X, y)
    assert len(eda.predict(X)) == 20
    
    # ES
    es = ES(generations=2, mu=5, lambda_=10)
    es.fit(X, y)
    assert len(es.predict(X)) == 20
    
    # PSO
    pso = PSO(iterations=2, population_size=5)
    pso.fit(X, y)
    assert len(pso.predict(X)) == 20
    
    # GP
    gp = GP(generations=2, population_size=5)
    gp.fit(X, y)
    assert len(gp.predict(X)) == 20

def test_probabilistic_models():
    X = np.random.randn(20, 10)
    y = np.random.randint(0, 2, 20)
    
    # Probabilistic SVC
    svc = ProbabilisticSVC()
    svc.fit(X, y)
    assert svc.predict_proba(X).shape == (20, 2)
    
    # Gaussian Process
    gp = GaussianProcess()
    gp.fit(X, y)
    assert gp.predict_proba(X).shape == (20, 2)
