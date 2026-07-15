__author__ = 'mhgou'
__version__ = '1.2.10'

from .graspnet import GraspNet
from .grasp import Grasp, GraspGroup, RectGrasp, RectGraspGroup


def __getattr__(name):
    """Load evaluation-only dependencies only when that API is requested."""
    if name == "GraspNetEval":
        from .graspnet_eval import GraspNetEval

        return GraspNetEval
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
