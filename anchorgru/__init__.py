"""anchorgru: a recurrent network that learns a bounded, penalised log-odds correction on top of a prior.

Start from an existing probability estimate and let a small recurrent network nudge it, only as far as the
recent history of each example justifies. The network starts as the prior exactly (zero-initialised head), its
offset is bounded (tanh cap) and shrunk toward zero in output space (``penalty``), so it does not drift away from
a good prior where the sequence carries no signal.
"""
from .calibrate import Calibrator, choose_calibration
from .diagnostics import residual_diagnostics
from .estimator import AnchoredGRUClassifier
from .io import load, save
from .tune.split import BlockedTimeSplit
from .windows import check_no_lookahead, point_in_time_windows

__version__ = "0.1.0"

__all__ = ["AnchoredGRUClassifier", "BlockedTimeSplit", "point_in_time_windows", "check_no_lookahead",
           "Calibrator", "choose_calibration", "residual_diagnostics", "save", "load", "__version__"]
