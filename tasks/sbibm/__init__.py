from sbibm.tasks.task import Task
from typing import Any

def get_task(name: str, *args: Any, **kwargs: Any) -> Task:
    if name == "slcp":
        from tasks.sbibm.slcp import SLCP
        return SLCP(*args, **kwargs)
    
    elif name == "lotka_volterra":
        from tasks.sbibm.lotka_volterra import LotkaVolterra
        return LotkaVolterra(*args, **kwargs)   
    
    elif name == "sir":
        from tasks.sbibm.sir import SIR
        return SIR(*args, **kwargs)
    
    elif name == "gaussian_linear":
        from tasks.sbibm.gaussian_linear import GaussianLinear
        return GaussianLinear(*args, **kwargs)
    
    elif name == "gaussian_mixture":
        from tasks.sbibm.gaussian_mixture import GaussianMixture
        return GaussianMixture(*args, **kwargs)
    elif name == "gaussian_mixture_uniform":
        from tasks.sbibm.gaussian_mixture import GaussianMixture
        return GaussianMixture(uniform=True, *args, **kwargs)
    
    elif name == "two_moons":
        from tasks.sbibm.two_moons import TwoMoons
        return TwoMoons(*args, **kwargs)
    elif name == "bernoulli_glm":
        from tasks.sbibm.bernoulli_glm import BernoulliGLM
        return BernoulliGLM(*args, **kwargs)
    elif name == "bernoulli_glm_raw":
        from tasks.sbibm.bernoulli_glm import BernoulliGLM
        return BernoulliGLM(summary="raw", *args, **kwargs)
    else:
        raise NotImplementedError(f"Task {name} not implemented.")
