"""NFL Route Tracker - Phase 2: Synthetic Route Generation
===========================================================
Generates synthetic NFL route trajectories for training data augmentation.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import random
 
 
# =============================================================================
# Route type definitions
# =============================================================================
 
class RouteType(Enum):
    STREAK   = 'streak'
    SLANT    = 'slant'
    POST     = 'post'
    CORNER   = 'corner'
    DRAG     = 'drag'
    CURL     = 'curl'
    DIG      = 'dig'
    OUT      = 'out'
    COMEBACK = 'comeback'
    FLAT     = 'flat'
    WHEEL    = 'wheel'
 
 
ALL_ROUTE_TYPES = list(RouteType)
 
STREAK_ROUTES   = [RouteType.STREAK]
SLANT_ROUTES    = [RouteType.SLANT]
POST_ROUTES     = [RouteType.POST]
CORNER_ROUTES   = [RouteType.CORNER]
DRAG_ROUTES     = [RouteType.DRAG]
CURL_ROUTES     = [RouteType.CURL]
DIG_ROUTES      = [RouteType.DIG]
OUT_ROUTES      = [RouteType.OUT]
COMEBACK_ROUTES = [RouteType.COMEBACK]
FLAT_ROUTES     = [RouteType.FLAT]
WHEEL_ROUTES    = [RouteType.WHEEL]
 
 
# =============================================================================
# Parameter dataclass
# =============================================================================
 
@dataclass
class RouteParameters:
    """
    Parameters controlling a single synthetic route.
    All spatial values normalized to [0, 1].
      X = upfield direction
      Y = lateral / sideline direction
    """
    start_x:      float = 0.0
    start_y:      float = 0.5
    depth:        float = 0.25
    width:        float = 0.15
    cut_depth:    float = 0.50 
    noise_std:    float = 0.003
    smooth_sigma: float = 2.5
    num_points:   int   = 64
 
 
# =============================================================================
# Output dataclass
# =============================================================================
 
@dataclass
class SyntheticRoute:
    """A generated synthetic route trajectory."""
    route_type: RouteType
    x_coords:   np.ndarray
    y_coords:   np.ndarray
    params:     RouteParameters
    num_points: int = 64
 
    def to_normalized_dict(self) -> Dict:
        return {'route_type': self.route_type.value,
                'x_coords':   self.x_coords.tolist(),
                'y_coords':   self.y_coords.tolist(),
                'num_points': self.num_points,
                'params': {
                    'start_x': self.params.start_x,
                    'start_y': self.params.start_y,
                    'depth':   self.params.depth,
                    'width':   self.params.width,
                } if self.params is not None else None}
 
 
# =============================================================================
# Smoothing utility — linear interp + Gaussian blur
# =============================================================================
 
def _build_and_smooth(ctrl_x: List[float], ctrl_y: List[float], num_points: int, smooth_sigma: float,) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate through control points then apply Gaussian smoothing.
    """
    cx = np.array(ctrl_x, dtype=float)
    cy = np.array(ctrl_y, dtype=float)
 
    # Arc-length parameterisation — distributes samples evenly along the path
    t_ctrl = np.linspace(0.0, 1.0, len(cx))
 
    t_out  = np.linspace(0.0, 1.0, num_points)
    x_lin  = np.interp(t_out, t_ctrl, cx)
    y_lin  = np.interp(t_out, t_ctrl, cy)
 
    # mode='nearest' keeps endpoints pinned at their true positions
    x_smooth = gaussian_filter1d(x_lin, sigma=smooth_sigma, mode='nearest')
    y_smooth = gaussian_filter1d(y_lin, sigma=smooth_sigma, mode='nearest')
 
    return x_smooth, y_smooth
 
 
def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: a + t*(b-a). Used for all intermediate ctrl pts."""
    return a + t * (b - a)
 

# =============================================================================
# Y-flip utility for specific routes
# =============================================================================
 
def _maybe_flip_y(x: np.ndarray, y: np.ndarray, start_y: float, flip: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mirror the route in Y around start_y when flip=True.
    y_new = 2*start_y - y  (reflects around the horizontal line y=start_y).
    """
    if flip:
        return x, 2.0 * start_y - y
    return x, y
 
 
# =============================================================================
# Route generator
# =============================================================================
 
class SyntheticRouteGenerator:
    """
    Generates smooth, realistic synthetic NFL route trajectories.
    """
 
    def __init__(self, random_seed: Optional[int] = None):
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        self.rng = np.random.default_rng(random_seed)
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def generate(self, route_type: RouteType, params: Optional[RouteParameters] = None) -> SyntheticRoute:
        if params is None:
            params = self._random_params()

        generators = {
            RouteType.STREAK:   self._streak,
            RouteType.SLANT:    self._slant,
            RouteType.POST:     self._post,
            RouteType.CORNER:   self._corner,
            RouteType.DRAG:     self._drag,
            RouteType.CURL:     self._curl,
            RouteType.DIG:      self._dig,
            RouteType.OUT:      self._out,
            RouteType.COMEBACK: self._comeback,
            RouteType.FLAT:     self._flat,
            RouteType.WHEEL:    self._wheel,
        }

        x, y = generators[route_type](params)

        # Add noise
        if params.noise_std > 0:
            x += self.rng.normal(0, params.noise_std, len(x))
            y += self.rng.normal(0, params.noise_std, len(y))

        # Truncate BEFORE clipping
        x, y = _truncate_at_boundary(x, y)

        # Curl and comeback intentionally move backward in X — don't enforce monotonic
        allow_backward_x = route_type in {RouteType.CURL, RouteType.COMEBACK}
        x, y = _validate_route(x, y, enforce_monotonic_x=not allow_backward_x)

        return SyntheticRoute(route_type = route_type,
                                x_coords = x, y_coords = y,
                                params = params,
                                num_points = len(x))
 
    def generate_dataset(self, route_types: List[RouteType],
                        routes_per_type: int = 100, param_variations: bool = True) -> List[SyntheticRoute]:
        """Generate a labelled dataset of synthetic routes."""
        routes = []
        for route_type in route_types:
            for _ in range(routes_per_type):
                params = self._random_params() if param_variations else RouteParameters()
                routes.append(self.generate(route_type, params))
        return routes
 
    # ------------------------------------------------------------------
    # Parameter randomization
    # ------------------------------------------------------------------
 
    def _random_params(self) -> RouteParameters:
        """
        Sample realistic route parameters covering both sideline alignments.
        Always called by generate() so single-route preview plots also vary.
        """
        alignment = random.choice(['left_sideline', 'right_sideline', 'slot'])
        if alignment == 'left_sideline':
            start_y = random.uniform(0.05, 0.22)
        elif alignment == 'right_sideline':
            start_y = random.uniform(0.78, 0.95)
        else:
            start_y = random.uniform(0.35, 0.65)
 
        return RouteParameters(
            start_x      = random.uniform(0.0,  0.04),
            start_y      = start_y,
            depth        = random.uniform(0.25, 0.60),
            width        = random.uniform(0.08, 0.20),
            cut_depth    = random.uniform(0.35, 0.65),
            noise_std    = random.uniform(0.001, 0.004),
            smooth_sigma = random.uniform(2.0,  3.5),
            num_points   = 64,
        )
 
    # ------------------------------------------------------------------
    # Y-flip helper — 50/50 coin flip
    # ------------------------------------------------------------------
 
    def _flip(self) -> bool:
        return bool(self.rng.random() < 0.5)
 
    # ------------------------------------------------------------------
    # Perturbation helpers
    # ------------------------------------------------------------------
 
    def _small(self, lo: float = -0.05, hi: float = 0.05) -> float:
        """Uniform random value in [lo, hi]. Default ±0.05."""
        return random.uniform(lo, hi)
 
    def _pos_small(self, lo: float = 0.02, hi: float = 0.08) -> float:
        """Positive-only small perturbation. Used for magnitude randomness."""
        return random.uniform(lo, hi)
 
    # ------------------------------------------------------------------
    # Individual route generators
    # ------------------------------------------------------------------
 
    def _streak(self, p: RouteParameters):
        depth = p.depth * 1.6

        drift_dir = 1 if self._flip() else -1
        dy = drift_dir * random.uniform(0.02, 0.06)
        y_end = np.clip(p.start_y + dy, 0.0, 1.0)

        end_x = p.start_x + depth

        ctrl_x = [p.start_x, _lerp(p.start_x, end_x, 0.4), end_x]
        ctrl_y = [p.start_y, _lerp(p.start_y, y_end, 0.5), y_end]

        return _build_and_smooth(ctrl_x, ctrl_y, p.num_points, p.smooth_sigma)
  
    def _slant(self, p: RouteParameters):
        cut_frac = np.clip(p.cut_depth + self._small(-0.55, 0.05), 0.25, 0.50)

        cut_x = p.start_x + p.depth * cut_frac
        end_x = p.start_x + p.depth

        direction = _toward_midfield(p.start_y)

        # Enforce minimum slope
        dy = direction * random.uniform(0.3, 0.55)
        y_end = np.clip(p.start_y + dy, 0.0, 1.0)

        ctrl_x = [p.start_x, _lerp(p.start_x, cut_x, 0.55), cut_x, end_x]

        ctrl_y = [p.start_y, p.start_y, p.start_y, y_end]

        return _build_and_smooth(ctrl_x, ctrl_y, p.num_points, p.smooth_sigma)
 
    def _post(self, p: RouteParameters):
        cut_frac = np.clip(0.75 + self._small(-0.04, 0.04), 0.65, 0.85)

        cut_x = p.start_x + p.depth * cut_frac
        end_x = p.start_x + p.depth * 1.15

        direction = _toward_midfield(p.start_y)
        dy = direction * random.uniform(0.12, 0.30)
        y_end = np.clip(p.start_y + dy, 0.0, 1.0)

        ctrl_x = [p.start_x, _lerp(p.start_x, cut_x, 0.45), cut_x, end_x]
        ctrl_y = [p.start_y, p.start_y, p.start_y, y_end]

        return _build_and_smooth(ctrl_x, ctrl_y, p.num_points, p.smooth_sigma)
 
    def _corner(self, p: RouteParameters) -> Tuple[np.ndarray, np.ndarray]:
        cut_frac = np.clip(0.75 + self._small(-0.04, 0.04), 0.45, 0.65)
        cut_x    = p.start_x + p.depth * cut_frac
        end_x    = p.start_x + p.depth
        sideline = 0.0 if p.start_y < 0.5 else 1.0
 
        cut_reach = random.uniform(0.55, 0.80)
        y_end = p.start_y + (sideline - p.start_y) * cut_reach
 
        ctrl_x = [p.start_x, _lerp(p.start_x, cut_x, 0.45), cut_x, end_x]
        ctrl_y = [p.start_y, p.start_y, p.start_y, y_end]
 
        x, y = _build_and_smooth(ctrl_x, ctrl_y, p.num_points, p.smooth_sigma)
        return x, y
 
    def _drag(self, p: RouteParameters):
        depth_frac = np.clip(0.35 + self._small(-0.10, 0.10), 0.20, 0.60)
        end_x = p.start_x + p.depth * depth_frac

        y_end = float(np.clip(p.start_y + (0.5 - p.start_y) * 2.2, 0.0, 1.0))

        x_change = end_x - p.start_x
        y_change = abs(y_end - p.start_y)
        if y_change < 2.0 * x_change:
            direction = _toward_midfield(p.start_y)
            y_end = float(np.clip(p.start_y + direction * 2.0 * x_change, 0.0, 1.0))

        ctrl_x = [p.start_x, _lerp(p.start_x, end_x, 0.3), _lerp(p.start_x, end_x, 0.7), end_x]
        ctrl_y = [p.start_y, _lerp(p.start_y, y_end, 0.25), _lerp(p.start_y, y_end, 0.7), y_end]

        return _build_and_smooth(ctrl_x, ctrl_y, p.num_points, p.smooth_sigma)
 
    def _curl(self, p: RouteParameters):
        peak_x = p.start_x + p.depth * 0.45

        # Make backward segment LONGER
        return_dist = p.depth * random.uniform(0.03, 0.09)
        end_x = peak_x - return_dist

        direction = _toward_midfield(p.start_y)
        y_drift = direction * random.uniform(0.02, 0.05)

        # ADD TWO backward points (not one)
        mid_return_x = peak_x - return_dist * 0.5

        ctrl_x = [p.start_x, peak_x, mid_return_x, end_x]

        ctrl_y = [p.start_y, p.start_y, p.start_y + y_drift * 0.3, p.start_y + y_drift]

        x, y = _build_and_smooth(ctrl_x, ctrl_y, p.num_points, p.smooth_sigma)

        return np.clip(x, 0, 1), np.clip(y, 0, 1)
 
    def _dig(self, p: RouteParameters) -> Tuple[np.ndarray, np.ndarray]:
        cut_frac = np.clip(p.cut_depth + self._small(-0.05, 0.05), 0.35, 0.85)
        cut_x = p.start_x + p.depth * cut_frac
 
        # Slight slope after cut — mostly lateral but allows tiny ±X drift
        x_drift = self._small(-0.015, 0.03) * p.depth
        end_x = cut_x + x_drift
 
        # Stronger inside break toward midfield
        direction = _toward_midfield(p.start_y)
        dy = direction * random.uniform(0.18, 0.32)
        y_end = np.clip(p.start_y + dy, 0.0, 1.0)
 
        ctrl_x = [p.start_x, _lerp(p.start_x, cut_x, 0.5), cut_x, _lerp(cut_x, end_x, 0.35), end_x]
        ctrl_y = [p.start_y, p.start_y, p.start_y, _lerp(p.start_y, y_end, 0.55), y_end]
 
        x, y = _build_and_smooth(ctrl_x, ctrl_y, p.num_points, p.smooth_sigma)
        return x,y
 
    def _out(self, p: RouteParameters):
        cut_frac = np.clip(p.cut_depth + self._small(-0.05, 0.05), 0.30, 0.85)
        cut_x = p.start_x + p.depth * cut_frac

        direction = _toward_sideline(p.start_y)
        dy = direction * random.uniform(0.18, 0.32)
        y_end = np.clip(p.start_y + dy, 0.0, 1.0)

        end_x = cut_x + self._small(-0.01, 0.03) * p.depth

        ctrl_x = [p.start_x, _lerp(p.start_x, cut_x, 0.5), cut_x, _lerp(cut_x, end_x, 0.35), end_x]
        ctrl_y = [p.start_y, p.start_y, p.start_y, _lerp(p.start_y, y_end, 0.55), y_end]

        return _build_and_smooth(ctrl_x, ctrl_y, p.num_points, p.smooth_sigma)
 
    def _comeback(self, p: RouteParameters):
        depth = p.depth * random.uniform(1.20, 1.40)
        peak_x = p.start_x + depth * 0.68

        # Streak-like small lateral drift before cut
        drift_dir = 1 if self._flip() else -1
        y_peak = np.clip(p.start_y + drift_dir * random.uniform(0.015, 0.05), 0.0, 1.0)

        # MUCH stronger backward motion
        return_dist = depth * random.uniform(0.15, 0.25)
        end_x = peak_x - return_dist

        # Return slightly toward sideline / small drift on comeback
        y_drift = random.uniform(0.02, 0.06) * (1 if self._flip() else -1)
        y_return = np.clip(y_peak + y_drift, 0.0, 1.0)

        mid_return_x = peak_x - return_dist * 0.4
        late_return_x = peak_x - return_dist * 0.75

        ctrl_x = [p.start_x, _lerp(p.start_x, peak_x, 0.4), peak_x, mid_return_x, late_return_x, end_x]

        ctrl_y = [p.start_y, _lerp(p.start_y, y_peak, 0.5), y_peak, _lerp(y_peak, y_return, 0.3),  _lerp(y_peak, y_return, 0.7),  y_return]

        x, y = _build_and_smooth(ctrl_x, ctrl_y, p.num_points, p.smooth_sigma)

        return np.clip(x, 0, 1), np.clip(y, 0, 1)
 
    def _flat(self, p: RouteParameters):
        sideline = 0.0 if p.start_y < 0.5 else 1.0
        y_end = p.start_y + (sideline - p.start_y) * 0.50

        x_offset = self._small(-0.04, 0.10) * p.depth
        end_x = p.start_x + p.depth * 0.18 + x_offset

        # Enforce minimum slope — Y change must be at least 2x the X change
        x_change = abs(end_x - p.start_x)
        y_change = abs(y_end - p.start_y)
        if y_change < 2.0 * x_change:
            direction = 1.0 if sideline > p.start_y else -1.0
            y_end = float(np.clip(p.start_y + direction * 2.0 * x_change, 0.0, 1.0))

        ctrl_x = [p.start_x, _lerp(p.start_x, end_x, 0.40), end_x]
        ctrl_y = [p.start_y,  _lerp(p.start_y, y_end, 0.55), y_end]

        x, y = _build_and_smooth(ctrl_x, ctrl_y, p.num_points, p.smooth_sigma)
        return _maybe_flip_y(x, y, p.start_y, self._flip())
 
    def _wheel(self, p: RouteParameters) -> Tuple[np.ndarray, np.ndarray]:
        depth = p.depth * 1.55 * random.uniform(0.88, 1.12)
        sideline = 0.0 if p.start_y < 0.5 else 1.0
 
        # Perturb lateral release depth
        lateral_reach = np.clip(0.60 + self._small(-0.08, 0.10), 0.45, 0.85)
        mid_x = p.start_x + depth * 0.15
        mid_y = p.start_y + (sideline - p.start_y) * lateral_reach
 
        end_x = p.start_x + depth
        end_y = p.start_y + (sideline - p.start_y) * 0.62
 
        ctrl_x = [p.start_x, mid_x, _lerp(p.start_x, end_x, 0.50),  _lerp(p.start_x, end_x, 0.80), end_x]
        ctrl_y = [p.start_y, mid_y,  _lerp(mid_y, end_y, 0.30),  _lerp(mid_y, end_y, 0.70), end_y]
 
        x, y = _build_and_smooth(ctrl_x, ctrl_y, p.num_points, p.smooth_sigma)
        return x, y 
 
def _toward_midfield(start_y: float) -> float:
    s = np.sign(0.5 - start_y)
    return s if s != 0.0 else 1.0

def _toward_sideline(start_y: float) -> float:
    s = np.sign(0.5 - start_y)
    return -s if s != 0.0 else 1.0

def _truncate_at_boundary(x: np.ndarray, y: np.ndarray):
    for i in range(1, len(y)):
        if y[i] <= 0.0 or y[i] >= 1.0:
            return x[:i+1], y[:i+1]
    return x, y

def _validate_route(x: np.ndarray, y: np.ndarray, enforce_monotonic_x: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sanitize route coordinates.
    """
    if enforce_monotonic_x:
        x = np.maximum.accumulate(x)

    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    return x, y

def generate_synthetic_dataset(route_types: Optional[List[RouteType]] = None,
                               routes_per_type: int = 100,
                               random_seed: Optional[int] = 42,) -> List[SyntheticRoute]:
    """Convenience wrapper to generate a full labelled dataset."""
    if route_types is None:
        route_types = ALL_ROUTE_TYPES
    generator = SyntheticRouteGenerator(random_seed=random_seed)
    return generator.generate_dataset(route_types, routes_per_type)