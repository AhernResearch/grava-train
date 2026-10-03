"""Executable Planner encoding and deterministic trajectory decoding.

Planner parameters are ordinary text in a JSON object; no tokenizer changes are required.
Coordinates use the ego frame: x points right and y points forward.
"""

import json
import re
from typing import List, Tuple

import numpy as np
from numpy.linalg import lstsq
from scipy.special import comb


class TrajectoryFormatError(ValueError):
    """The generated planner answer does not match the supported schema."""


class _PlannerParameterCodec:
    """Shared parameter quantization and parsing for the four motion planners.

    Provides shared utilities:
        - Gear detection and token constants
        - Uniform binning helpers
        - Text extraction and parsing for decode
    """

    GEAR_FORWARD = "D"
    GEAR_REVERSE = "R"

    @staticmethod
    def _is_reverse(trajectory: List[List[float]]) -> bool:
        """Detect reverse: endpoint y < start y (net backward displacement)."""
        return trajectory[-1][1] < trajectory[0][1]

    @staticmethod
    def _val_to_bin(val: float, vmin: float, vmax: float, n_bins: int) -> int:
        """Map value to bin index (uniform binning)."""
        clamped = max(vmin, min(val, vmax))
        b = int((clamped - vmin) / (vmax - vmin) * n_bins)
        return min(b, n_bins - 1)

    @staticmethod
    def _bin_to_val(b: int, vmin: float, vmax: float, n_bins: int) -> float:
        """Map bin index to value (uniform binning, returns bin center)."""
        return vmin + (b + 0.5) * (vmax - vmin) / n_bins

    @staticmethod
    def _extract_content(text: str) -> str:
        """Extract content from <traj> or <answer> tags, or return text as-is."""
        for tag in ["traj", "answer"]:
            match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
            if match:
                return match.group(1)
        return text.split("</think>", 1)[-1]

    @classmethod
    def _parse_gear(cls, content: str) -> Tuple[bool, str]:
        """Parse gear token from content start.

        Returns:
            (is_reverse, remaining_content)
        """
        content = content.strip()
        gear_match = re.search(r"([DR])\s+", content)
        if not gear_match:
            raise ValueError("No gear token 'D' or 'R' found")
        reverse = gear_match.group(1) == cls.GEAR_REVERSE
        return reverse, content[gear_match.end() :]


class BezierCodec(_PlannerParameterCodec):
    """Encode/decode trajectories using gear + Bezier control points.

    Parameter text: [Gear] [p1x] [p1y] [p2x] [p2y] [p3x] [p3y]

    The trajectory is approximated by a cubic Bezier curve with:
        - P0 = (0, 0) fixed at origin
        - P1, P2, P3 = control points (encoded as tokens)

    Control point ranges (all 0.10m resolution):
        - p1_x: [-4, 4], 80 bins
        - p1_y: [-1, 25], 260 bins
        - p2_x: [-8, 8], 160 bins
        - p2_y: [-1, 42], 430 bins
        - p3_x: [-22, 22], 440 bins
        - p3_y: [-5, 65], 700 bins

    These parameter strings use the model's existing tokenizer.
    """

    # Control point bin configs: (vmin, vmax, n_bins)
    _P1_X = (-4.0, 4.0, 80)
    _P1_Y = (-1.0, 25.0, 260)
    _P2_X = (-8.0, 8.0, 160)
    _P2_Y = (-1.0, 42.0, 430)
    _P3_X = (-22.0, 22.0, 440)
    _P3_Y = (-5.0, 65.0, 700)
    DT = 0.5
    HORIZON_S = 4.0

    @staticmethod
    def _bernstein(n: int, k: int, t: float) -> float:
        """Bernstein basis polynomial B_{k,n}(t)."""

        return float(comb(n, k, exact=True) * (t**k) * ((1 - t) ** (n - k)))

    def _resample_to_8(self, trajectory: List[List[float]]) -> np.ndarray:
        """Resample trajectory to exactly 8 points.

        If len(trajectory) == 8, returns as-is.
        If len(trajectory) > 8, uniformly resamples to 8 points.
        """
        traj = np.array(trajectory, dtype=np.float64)
        n = len(traj)
        if n == 8:
            return traj
        if n < 8:
            raise ValueError(f"Trajectory must have at least 8 points, got {n}")

        # Uniform resampling to 8 points
        # Map [0, 1, ..., n-1] to indices [0, ..., 7] in target
        indices = np.linspace(0, n - 1, 8)
        resampled = np.zeros((8, 2), dtype=np.float64)
        for i, idx in enumerate(indices):
            idx_floor = int(np.floor(idx))
            idx_ceil = min(idx_floor + 1, n - 1)
            frac = idx - idx_floor
            resampled[i] = (1 - frac) * traj[idx_floor] + frac * traj[idx_ceil]
        return resampled

    def _fit_control_points(
        self,
        trajectory: List[List[float]],
    ) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        reverse = self._is_reverse(trajectory)
        gear = self.GEAR_REVERSE if reverse else self.GEAR_FORWARD

        t_vals = [(i + 1) / 8.0 for i in range(8)]
        b_matrix = np.array([[self._bernstein(3, j, t) for j in range(4)] for t in t_vals])

        waypoints = self._resample_to_8(trajectory)
        if reverse:
            waypoints = waypoints.copy()
            waypoints[:, 1] = -waypoints[:, 1]


        p0 = np.array([0.0, 0.0])
        rhs = waypoints - b_matrix[:, 0:1] @ p0.reshape(1, 2)
        p_control, _, _, _ = lstsq(b_matrix[:, 1:], rhs, rcond=None)
        return gear, p_control[0], p_control[1], p_control[2]

    def encode(self, trajectory: List[List[float]]) -> str:
        """Encode trajectory to gear + Bezier control point tokens.

        Args:
            trajectory: List of [x, y] pairs (8 or more waypoints, resampled to 8)

        Returns:
            Token string like 'D p1x_40 p1y_130 p2x_80 p2y_215 p3x_220 p3y_350'
        """
        gear, P1, P2, P3 = self._fit_control_points(trajectory)

        # Quantize control points
        p1x_b = self._val_to_bin(P1[0], *self._P1_X)
        p1y_b = self._val_to_bin(P1[1], *self._P1_Y)
        p2x_b = self._val_to_bin(P2[0], *self._P2_X)
        p2y_b = self._val_to_bin(P2[1], *self._P2_Y)
        p3x_b = self._val_to_bin(P3[0], *self._P3_X)
        p3y_b = self._val_to_bin(P3[1], *self._P3_Y)

        tokens = [
            gear,
            f"p1x_{p1x_b}",
            f"p1y_{p1y_b}",
            f"p2x_{p2x_b}",
            f"p2y_{p2y_b}",
            f"p3x_{p3x_b}",
            f"p3y_{p3y_b}",
        ]
        return " ".join(tokens)

    def decode(self, text: str) -> np.ndarray:
        """Decode gear + Bezier tokens to trajectory (8 waypoints).

        Args:
            text: String with gear and control point tokens

        Returns:
            np.ndarray of shape (8, 2)
        """
        content = self._extract_content(text)
        reverse, content = self._parse_gear(content)

        # Parse planner control point parameters.
        def parse(prefix: str, cfg: Tuple[float, float, int]) -> float:
            match = re.search(rf"{prefix}_(\d+)", content)
            if not match:
                raise ValueError(f"No {prefix}_N token found")
            return self._bin_to_val(int(match.group(1)), *cfg)

        P1 = np.array([parse("p1x", self._P1_X), parse("p1y", self._P1_Y)])
        P2 = np.array([parse("p2x", self._P2_X), parse("p2y", self._P2_Y)])
        P3 = np.array([parse("p3x", self._P3_X), parse("p3y", self._P3_Y)])
        P0 = np.array([0.0, 0.0])

        # Sample Bezier curve at t = 1/8, 2/8, ..., 8/8
        control = np.array([P0, P1, P2, P3])  # (4, 2)
        waypoints = []
        for i in range(8):
            t = (i + 1) / 8.0
            point = sum(self._bernstein(3, j, t) * control[j] for j in range(4))
            waypoints.append(point)
        waypoints = np.array(waypoints)

        if reverse:
            waypoints[:, 1] = -waypoints[:, 1]

        return waypoints


class AbsPointsCodec(_PlannerParameterCodec):
    """Encode/decode low-speed trajectories with absolute waypoint tokens.

    Planner parameter format: ``ABS_POINTS D afx1_N ay1_N ... afx8_N ay8_N``.
    This keeps stopped/crawling cases explicit instead of forcing a smooth curve.
    """

    X_RANGE = (-2.0, 2.0)
    X_STEP = 0.02
    X_BINS = 201
    Y_RANGE = (-5.0, 15.0)
    Y_STEP = 0.1
    Y_BINS = 201

    @staticmethod
    def _val_to_grid(val: float, vmin: float, vmax: float, step: float, n_bins: int) -> int:
        clamped = max(vmin, min(float(val), vmax))
        b = int(round((clamped - vmin) / step))
        return max(0, min(b, n_bins - 1))

    @staticmethod
    def _grid_to_val(b: int, vmin: float, step: float, n_bins: int) -> float:
        b = max(0, min(int(b), n_bins - 1))
        return round(vmin + b * step, 2)

    def encode(self, trajectory: List[List[float]]) -> str:
        if len(trajectory) != 8:
            raise ValueError(f"Expected 8 waypoints, got {len(trajectory)}")
        reverse = float(trajectory[-1][1]) < -0.5
        tokens = ["ABS_POINTS", self.GEAR_REVERSE if reverse else self.GEAR_FORWARD]
        for idx, point in enumerate(trajectory, start=1):
            x, y = float(point[0]), float(point[1])
            xb = self._val_to_grid(x, *self.X_RANGE, self.X_STEP, self.X_BINS)
            yb = self._val_to_grid(y, *self.Y_RANGE, self.Y_STEP, self.Y_BINS)
            tokens.extend([f"afx{idx}_{xb}", f"ay{idx}_{yb}"])
        return " ".join(tokens)

    def decode(self, text: str) -> np.ndarray:
        content = self._extract_content(text)
        if "ABS_POINTS" not in content:
            raise ValueError("No ABS_POINTS planner token found")
        coords = []
        for idx in range(1, 9):
            xm = re.search(rf"afx{idx}_(\d+)", content)
            ym = re.search(rf"ay{idx}_(\d+)", content)
            if not xm or not ym:
                raise ValueError(f"Missing afx{idx}_/ay{idx}_ token")
            x = self._grid_to_val(int(xm.group(1)), self.X_RANGE[0], self.X_STEP, self.X_BINS)
            y = self._grid_to_val(int(ym.group(1)), self.Y_RANGE[0], self.Y_STEP, self.Y_BINS)
            coords.append([x, y])
        return np.array(coords)


class StaticPointCodec(_PlannerParameterCodec):
    """Encode/decode final-stop trajectories with one endpoint.

    Planner parameter format: ``STATIC_POINT D sx_N sy_N``.
    Decoding expands the endpoint with a monotone stop profile. This keeps
    final-stop scenes compact while preserving the "move, then settle" temporal
    shape that linear interpolation loses.
    """

    X_RANGE = (-2.0, 2.0)
    X_STEP = 0.1
    X_BINS = 41
    Y_RANGE = (-2.0, 2.0)
    Y_BINS = 41
    Y_STEP = 0.1

    @classmethod
    def can_encode_endpoint(cls, x: float, y: float) -> bool:
        return (
            cls.X_RANGE[0] <= float(x) <= cls.X_RANGE[1]
            and cls.Y_RANGE[0] <= float(y) <= cls.Y_RANGE[1]
        )

    @staticmethod
    def _val_to_grid(val: float, vmin: float, vmax: float, step: float, n_bins: int) -> int:
        clamped = max(vmin, min(float(val), vmax))
        b = int(round((clamped - vmin) / step))
        return max(0, min(b, n_bins - 1))

    @staticmethod
    def _grid_to_val(b: int, vmin: float, step: float, n_bins: int) -> float:
        b = max(0, min(int(b), n_bins - 1))
        return round(vmin + b * step, 2)

    def encode(self, trajectory: List[List[float]]) -> str:
        if len(trajectory) != 8:
            raise ValueError(f"Expected 8 waypoints, got {len(trajectory)}")
        end_x, end_y = float(trajectory[-1][0]), float(trajectory[-1][1])
        reverse = end_y < -0.5
        sx = self._val_to_grid(end_x, *self.X_RANGE, self.X_STEP, self.X_BINS)
        sy = self._val_to_grid(end_y, *self.Y_RANGE, self.Y_STEP, self.Y_BINS)
        return " ".join(
            [
                "STATIC_POINT",
                self.GEAR_REVERSE if reverse else self.GEAR_FORWARD,
                f"sx_{sx}",
                f"sy_{sy}",
            ]
        )

    @staticmethod
    def _stop_profile(distance: float, ego_speed: float) -> np.ndarray:
        if distance <= 1e-6:
            return np.zeros(8, dtype=np.float64)
        u = np.linspace(0.5, 4.0, 8) / 4.0
        alpha = float(np.clip(1.2 + max(float(ego_speed), 0.0) / 8.0, 1.2, 2.0))
        front_loaded_u = 1.0 - (1.0 - u) ** alpha
        ratios = 6.0 * front_loaded_u**5 - 15.0 * front_loaded_u**4 + 10.0 * front_loaded_u**3
        if not np.all(np.isfinite(ratios)):
            raise ValueError("STATIC_POINT stop profile produced non-finite values")
        ratios = np.maximum.accumulate(np.clip(ratios, 0.0, 1.0))
        ratios[-1] = 1.0
        return ratios

    def decode_endpoint(self, end_x: float, end_y: float, ego_speed: float = 0.0) -> np.ndarray:
        distance = float(np.hypot(end_x, end_y))
        ratios = self._stop_profile(distance, ego_speed)
        coords = []
        for ratio in ratios:
            coords.append([round(float(end_x) * ratio, 2), round(float(end_y) * ratio, 2), 0.0])
        return np.array(coords)

    def decode(self, text: str, ego_speed: float = 0.0) -> np.ndarray:
        content = self._extract_content(text)
        if "STATIC_POINT" not in content:
            raise ValueError("No STATIC_POINT planner token found")
        xm = re.search(r"sx_(\d+)", content)
        ym = re.search(r"sy_(\d+)", content)
        if not xm or not ym:
            raise ValueError("STOP requires sx_/sy_ parameters")
        end_x = self._grid_to_val(int(xm.group(1)), self.X_RANGE[0], self.X_STEP, self.X_BINS)
        end_y = self._grid_to_val(int(ym.group(1)), self.Y_RANGE[0], self.Y_STEP, self.Y_BINS)
        return self.decode_endpoint(end_x, end_y, ego_speed=ego_speed)


class ControlParamCodec(_PlannerParameterCodec):
    """Encode/decode straight-driving trajectories with low-dimensional controls.

    Planner parameter format: ``CONTROL D cy_N cv_N clat_N``.

    ``cy`` is final longitudinal progress in meters, ``cv`` is final speed,
    and ``clat`` is endpoint lateral offset. Decoding deterministically expands
    those parameters into eight waypoints conditioned on observed ego speed using
    a monotone Hermite longitudinal profile. Legacy ``cyaw`` tokens are accepted
    during decode but ignored. This is a codec decoder, not a learned planner.
    """

    DT = 0.5
    Y_RANGE = (0.0, 60.0)
    Y_BINS = 600
    V_RANGE = (0.0, 16.0)
    V_BINS = 160
    YAW_RANGE = (-0.5, 0.5)
    YAW_BINS = 100
    LAT_RANGE = (-2.0, 2.0)
    LAT_BINS = 80
    HORIZON_S = 4.0

    @staticmethod
    def _segment_speeds(trajectory: List[List[float]]) -> np.ndarray:
        points = np.array([[point[0], point[1]] for point in trajectory], dtype=np.float64)
        prev = np.vstack([np.zeros((1, 2), dtype=np.float64), points[:-1]])
        return np.linalg.norm(points - prev, axis=1) / ControlParamCodec.DT

    def encode(self, trajectory: List[List[float]], ego_speed: float = 0.0) -> str:
        if len(trajectory) != 8:
            raise ValueError(f"Expected 8 waypoints, got {len(trajectory)}")
        reverse = float(trajectory[-1][1]) < -0.5
        speeds = self._segment_speeds(trajectory)
        progress_end = max(float(trajectory[-1][1]), 0.0)
        v_end = float(speeds[-1])
        lat_end = float(trajectory[-1][0])

        return " ".join(
            [
                "CONTROL",
                self.GEAR_REVERSE if reverse else self.GEAR_FORWARD,
                f"cy_{self._val_to_bin(progress_end, *self.Y_RANGE, self.Y_BINS)}",
                f"cv_{self._val_to_bin(v_end, *self.V_RANGE, self.V_BINS)}",
                f"clat_{self._val_to_bin(lat_end, *self.LAT_RANGE, self.LAT_BINS)}",
            ]
        )

    @classmethod
    def _decode_longitudinal(
        cls,
        progress_end: float,
        v_end: float,
        ego_speed: float,
        ego_accel: float,
    ) -> np.ndarray:
        """Reconstruct longitudinal progress without hard-constraining acceleration."""
        total_t = cls.HORIZON_S
        s_end = max(float(progress_end), 0.0)
        if s_end <= 1e-6:
            return np.zeros(8, dtype=np.float64)

        # Instantaneous acceleration is noisy and often conflicts with the next
        # four seconds of driving intent. Use ego speed and target end speed as
        # Hermite slopes, clipped to a monotone-feasible range.
        avg_speed = s_end / total_t
        slope_cap = max(0.0, 3.0 * avg_speed)
        v0 = min(max(float(ego_speed), 0.0), slope_cap)
        vt = min(max(float(v_end), 0.0), slope_cap)
        sample_t = np.linspace(cls.DT, total_t, 8)
        u = sample_t / total_t
        h10 = u**3 - 2 * u**2 + u
        h01 = -2 * u**3 + 3 * u**2
        h11 = u**3 - u**2
        y = h10 * total_t * v0 + h01 * s_end + h11 * total_t * vt
        if not np.all(np.isfinite(y)):
            raise ValueError("CONTROL longitudinal decode produced non-finite values")

        y = np.maximum.accumulate(np.maximum(y, 0.0))
        if y[-1] <= 1e-6:
            return np.zeros(8, dtype=np.float64)
        y = y * (s_end / y[-1])
        y[-1] = s_end
        return y

    def decode_params(
        self,
        progress_end: float,
        lat_end: float,
        v_end: float,
        ego_speed: float = 0.0,
        ego_accel: float = 0.0,
    ) -> np.ndarray:
        t = np.linspace(1 / 8, 1.0, 8)
        y = self._decode_longitudinal(progress_end, v_end, ego_speed, ego_accel)
        x = float(lat_end) * t
        return np.column_stack([x, y])

    def decode(self, text: str, ego_speed: float = 0.0, ego_accel: float = 0.0) -> np.ndarray:
        content = self._extract_content(text)
        if "CONTROL" not in content:
            raise ValueError("No CONTROL planner token found")
        progress_m = re.search(r"cy_(\d+)", content)
        vm = re.search(r"cv_(\d+)", content)
        lm = re.search(r"clat_(\d+)", content)
        if not (progress_m and vm and lm):
            raise ValueError("CRUISE requires cy_/cv_/clat_ parameters")
        progress_end = self._bin_to_val(int(progress_m.group(1)), *self.Y_RANGE, self.Y_BINS)
        v_end = self._bin_to_val(int(vm.group(1)), *self.V_RANGE, self.V_BINS)
        lat_end = self._bin_to_val(int(lm.group(1)), *self.LAT_RANGE, self.LAT_BINS)

        return self.decode_params(
            progress_end,
            lat_end,
            v_end,
            ego_speed=ego_speed,
            ego_accel=ego_accel,
        )


class ExecutablePlannerCodec:
    """Encode and decode the four Executable Planners."""

    STOP = "STOP"
    CRAWL = "CRAWL"
    CURVE = "CURVE"
    CRUISE = "CRUISE"

    def __init__(self) -> None:
        self.static_point = StaticPointCodec()
        self.abs_points = AbsPointsCodec()
        self.bezier = BezierCodec()
        self.control = ControlParamCodec()

    @staticmethod
    def _path_features(trajectory: List[List[float]]) -> dict:
        points = np.array(trajectory, dtype=np.float64)
        prev = np.vstack([np.zeros((1, 2), dtype=np.float64), points[:-1]])
        deltas = points - prev
        step = np.linalg.norm(deltas, axis=1)
        headings = np.unwrap(np.arctan2(deltas[:, 0], np.maximum(deltas[:, 1], 1e-6)))
        turn = np.abs(np.diff(headings)) if len(headings) > 1 else np.array([])
        return {
            "path_len": float(step.sum()),
            "mean_step": float(step.mean()),
            "final_x": float(points[-1, 0]),
            "final_y": float(points[-1, 1]),
            "max_abs_x": float(np.abs(points[:, 0]).max()),
            "lat_span": float(points[:, 0].max() - points[:, 0].min()),
            "max_turn": float(turn.max()) if len(turn) else 0.0,
            "sum_turn": float(turn.sum()) if len(turn) else 0.0,
            "late_step": float(step[-1]),
            "v_end": float(step[-1] / 0.5),
        }

    @classmethod
    def select_planner(
        cls,
        trajectory: List[List[float]],
        navigation: str = "",
        ego_speed: float = 0.0,
    ) -> str:
        f = cls._path_features(trajectory)
        nav = navigation.upper()
        if (
            f["v_end"] <= 0.2
            and f["max_abs_x"] <= 0.8
            and StaticPointCodec.can_encode_endpoint(f["final_x"], f["final_y"])
        ):
            return cls.STOP
        if nav in {"TURN_LEFT", "TURN RIGHT", "TURN_RIGHT", "TURN LEFT"}:
            return cls.CURVE
        if (
            abs(f["final_x"]) > 1.5
            or f["max_abs_x"] > 1.8
            or f["lat_span"] > 1.5
            or f["max_turn"] > 0.18
            or f["sum_turn"] > 0.45
        ):
            return cls.CURVE
        if (
            f["path_len"] < 6.0
            or f["mean_step"] < 0.75
            or (ego_speed < 2.0 and f["final_y"] < 8.0)
            or (f["late_step"] < 0.35 and f["path_len"] < 10.0)
        ):
            return cls.CRAWL
        return cls.CRUISE

    def encode_json(
        self,
        trajectory: List[List[float]],
        navigation: str = "",
        ego_speed: float = 0.0,
        planner_override: str | None = None,
    ) -> dict:
        """Serialize a trajectory using planner-specific text parameters."""
        xy = [[float(p[0]), float(p[1])] for p in trajectory]
        planner = planner_override or self.select_planner(xy, navigation, ego_speed)
        if planner == self.STOP:
            tokens = self.static_point.encode(xy).split()
            gear, params = tokens[1], {"endpoint": tokens[2:4]}
        elif planner == self.CRAWL:
            tokens = self.abs_points.encode(xy).split()
            gear, params = tokens[1], {"points": tokens[2:]}
        elif planner == self.CURVE:
            tokens = self.bezier.encode(xy).split()
            gear = tokens[0]
            params = {"p3": tokens[5:7], "p2": tokens[3:5], "p1": tokens[1:3]}
        elif planner == self.CRUISE:
            tokens = self.control.encode(xy, ego_speed=ego_speed).split()
            gear = tokens[1]
            params = {"progress_end": tokens[2], "lat_end": tokens[4], "v_end": tokens[3]}
        else:
            raise ValueError(f"Unknown Executable Planner: {planner!r}")
        return {"planner": planner, "gear": gear, "params": params}

    @staticmethod
    def _parameters(params: dict, key: str, prefixes: list[str]) -> list[str]:
        values = params.get(key)
        if len(prefixes) == 1:
            values = [values]
        if not isinstance(values, list) or len(values) != len(prefixes):
            raise TrajectoryFormatError(f"Invalid Executable Planner field: {key}")
        for value, prefix in zip(values, prefixes):
            if not isinstance(value, str) or not re.fullmatch(rf"{prefix}_\d+", value):
                raise TrajectoryFormatError(f"Expected {prefix}_ parameter in {key}")
        return values

    def decode(self, text: str, ego_speed: float = 0.0, ego_accel: float = 0.0) -> np.ndarray:
        """Decode a planner JSON answer into eight waypoints."""
        content = _PlannerParameterCodec._extract_content(text)
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match is None:
            raise TrajectoryFormatError("No Executable Planner JSON found")
        data = json.loads(match.group())
        gear, planner, params = data.get("gear"), data.get("planner"), data.get("params")
        if gear not in ("D", "R") or not isinstance(params, dict):
            raise TrajectoryFormatError("Executable Planner requires gear D/R and a params object")
        if planner == self.STOP:
            values = self._parameters(params, "endpoint", ["sx", "sy"])
            return self.static_point.decode(" ".join(["STATIC_POINT", gear, *values]), ego_speed)
        if planner == self.CRAWL:
            prefixes = [p for i in range(1, 9) for p in (f"afx{i}", f"ay{i}")]
            values = self._parameters(params, "points", prefixes)
            return self.abs_points.decode(" ".join(["ABS_POINTS", gear, *values]))
        if planner == self.CURVE:
            values = [
                v
                for i in range(1, 4)
                for v in self._parameters(params, f"p{i}", [f"p{i}x", f"p{i}y"])
            ]
            return self.bezier.decode(" ".join([gear, *values]))
        if planner == self.CRUISE:
            values = [
                v
                for key, prefix in (("progress_end", "cy"), ("v_end", "cv"), ("lat_end", "clat"))
                for v in self._parameters(params, key, [prefix])
            ]
            return self.control.decode(" ".join(["CONTROL", gear, *values]), ego_speed, ego_accel)
        raise TrajectoryFormatError(f"Unknown Executable Planner: {planner!r}")
