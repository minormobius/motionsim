# -*- coding: utf-8 -*-
"""
Jerk-limited motion profile simulator + multi-axis sequence composer
(sequential + parallel steps), with axis load model (gravity + Coulomb friction)
and motor sizing trade plots vs conversion (mm/rev).

Features
- 7-segment symmetric jerk-limited S-curve (rest-to-rest)
- Sequence runner with parallel "steps" (list of moves run concurrently)
- Per-axis time series: position, velocity, accel, jerk, force, moment
- Coulomb (constant) friction term opposing motion
- Drive configuration:
    - PulleyDrive(mm_per_rev, efficiency)
    - BallScrewDrive(lead_mm_per_rev, shaft_diameter_mm, efficiency)
  (Both map to an equivalent mm/rev conversion for motor torque/RPM sizing)
- Plot filtering: only plot axes that actually moved

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional, Any


# ============================================================
# 7-segment jerk-limited S-curve (rest-to-rest, symmetric)
# ============================================================

@dataclass
class Profile7Seg:
    Tj: float
    Ta: float
    Tv: float
    v_peak: float
    a_peak: float
    j_max: float
    L: float


def _s_accel(Tj: float, Ta: float, a_peak: float, j_max: float) -> float:
    """Distance during accelerating half (0 -> v_peak)."""
    if Ta <= 1e-15:
        # triangular accel (no constant-accel plateau)
        return j_max * Tj**3
    return a_peak * (Tj**2 + 1.5 * Ta * Tj + 0.5 * Ta**2)


def design_7seg(L: float, vmax: float, amax: float, jmax: float) -> Profile7Seg:
    """Design symmetric rest-to-rest jerk-limited profile for |L|."""
    L = float(abs(L))
    vmax, amax, jmax = float(vmax), float(amax), float(jmax)
    eps = 1e-12

    if L < eps:
        return Profile7Seg(0.0, 0.0, 0.0, 0.0, 0.0, jmax, 0.0)

    v_thr = amax**2 / jmax  # velocity threshold where Ta just becomes >=0

    # Try vmax-limited shape
    if vmax >= v_thr:
        Tj = amax / jmax
        Ta = vmax / amax - Tj
        a_peak = amax
    else:
        Ta = 0.0
        Tj = np.sqrt(vmax / jmax)
        a_peak = jmax * Tj

    s_acc = _s_accel(Tj, Ta, a_peak, jmax)

    # Cruise?
    if L >= 2.0 * s_acc - 1e-12:
        Tv = max(0.0, (L - 2.0 * s_acc) / vmax)
        return Profile7Seg(Tj, Ta, Tv, vmax, a_peak, jmax, L)

    # No cruise (Tv=0): solve for v_peak based on distance
    Tv = 0.0

    # Triangular accel candidate
    v_tri = (L * np.sqrt(jmax) / 2.0) ** (2.0 / 3.0)
    a_tri = np.sqrt(v_tri * jmax)
    if a_tri <= amax + 1e-9 and v_tri <= vmax + 1e-9:
        Tj = np.sqrt(v_tri / jmax)
        return Profile7Seg(Tj, 0.0, 0.0, v_tri, jmax * Tj, jmax, L)

    # amax-limited: solve Ta from quadratic with Tj fixed
    Tj = amax / jmax
    disc = Tj**2 + 4.0 * L / amax
    Ta = max(0.0, (-3.0 * Tj + np.sqrt(disc)) / 2.0)
    v_peak = amax * (Ta + Tj)
    return Profile7Seg(Tj, Ta, 0.0, v_peak, amax, jmax, L)


def sample_7seg(profile: Profile7Seg, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample x/v/a/j for the profile in a *relative positive* frame: x goes 0 -> L.
    Sign is handled by the caller.
    """
    Tj, Ta, Tv = profile.Tj, profile.Ta, profile.Tv
    jmax = profile.j_max

    segs = [
        (Tj, +jmax),
        (Ta,  0.0),
        (Tj, -jmax),
        (Tv,  0.0),
        (Tj, -jmax),
        (Ta,  0.0),
        (Tj, +jmax),
    ]

    t_all, x_all, v_all, a_all, j_all = [], [], [], [], []
    t0, x0, v0, a0 = 0.0, 0.0, 0.0, 0.0

    for dur, j in segs:
        if dur <= 1e-15:
            continue

        n = max(2, int(np.ceil(dur / dt)) + 1)
        tau = np.linspace(0.0, dur, n)

        a = a0 + j * tau
        v = v0 + a0 * tau + 0.5 * j * tau**2
        x = x0 + v0 * tau + 0.5 * a0 * tau**2 + (1.0 / 6.0) * j * tau**3
        jj = np.full_like(tau, j)

        if t_all:
            tau = tau[1:]; x = x[1:]; v = v[1:]; a = a[1:]; jj = jj[1:]

        t = t0 + tau

        t_all.append(t); x_all.append(x); v_all.append(v); a_all.append(a); j_all.append(jj)

        t0, x0, v0, a0 = float(t[-1]), float(x[-1]), float(v[-1]), float(a[-1])

    t = np.concatenate(t_all)
    x = np.concatenate(x_all)
    v = np.concatenate(v_all)
    a = np.concatenate(a_all)
    j = np.concatenate(j_all)

    # Normalize to hit exactly L (and keep v/a/j consistent)
    if profile.L > 0:
        scale = profile.L / x[-1]
        x *= scale
        v *= scale
        a *= scale
        j *= scale

    return t, x, v, a, j


# ============================================================
# Axis configuration: limits, drive types, load model
# ============================================================

@dataclass
class AxisLimits:
    vmax: float
    amax: float
    jmax: float


@dataclass
class PulleyDrive:
    """Belt/pulley/gear mapping expressed as linear travel per motor rev."""
    mm_per_rev: float
    efficiency: float = 0.95


@dataclass
class BallScrewDrive:
    """
    Ball screw mapping.
    lead_mm_per_rev: linear travel per screw rev (and motor rev if direct-coupled)
    shaft_diameter_mm: stored for later torsional / critical speed checks
    """
    lead_mm_per_rev: float
    shaft_diameter_mm: float
    efficiency: float = 0.90


DriveSpec = Union[PulleyDrive, BallScrewDrive]


def drive_mm_per_rev(drive: DriveSpec) -> float:
    if isinstance(drive, PulleyDrive):
        return float(drive.mm_per_rev)
    if isinstance(drive, BallScrewDrive):
        return float(drive.lead_mm_per_rev)
    raise TypeError(f"Unknown drive type: {type(drive)}")


def drive_efficiency(drive: DriveSpec) -> float:
    if isinstance(drive, PulleyDrive):
        return float(drive.efficiency)
    if isinstance(drive, BallScrewDrive):
        return float(drive.efficiency)
    raise TypeError(f"Unknown drive type: {type(drive)}")


@dataclass
class AxisLoadModel:
    """
    unit: axis direction in world coords (X,Y,Z), will be normalized
    r_offset_m: displacement vector from axis line to payload CG (m)
    mass_kg: payload mass acted on by this axis

    friction_N: Coulomb friction magnitude (N) opposing motion along axis.
    drive: PulleyDrive or BallScrewDrive (optional; used for motor sizing convenience).
    """
    limits: AxisLimits
    unit: Tuple[float, float, float]
    r_offset_m: Tuple[float, float, float]
    mass_kg: float

    friction_N: float = 0.0
    drive: Optional[DriveSpec] = None


@dataclass
class Move:
    axis: str
    dist: float  # signed
    dwell: float = 0.0


MoveLike = Union[Move, Tuple[str, float], Tuple[str, float, float]]
StepLike = Union[MoveLike, List[MoveLike]]  # either a single move or a parallel group


# ============================================================
# Multi-axis sequence simulator (supports parallel)
# ============================================================

def simulate_sequence(
    axes: Dict[str, AxisLoadModel],
    moves: List[StepLike],
    dt: float = 0.001,
    g: float = 9.80665,
    return_boundaries: bool = True,
):
    """
    Each top-level item in `moves` is a "step".
      - If it's a tuple/Move: one axis active that step.
      - If it's a list: multiple axes active in parallel that step.

    Step duration = max(duration of each axis command in the step).
    Axes that finish early hold (v=a=j=0) until the step ends.

    Returns:
      sim dict, boundaries list
    """

    def _to_move(obj: MoveLike) -> Move:
        if isinstance(obj, Move):
            return obj
        if isinstance(obj, tuple):
            if len(obj) == 2:
                return Move(axis=obj[0], dist=float(obj[1]), dwell=0.0)
            if len(obj) == 3:
                return Move(axis=obj[0], dist=float(obj[1]), dwell=float(obj[2]))
            raise ValueError("Tuple moves must be (axis, dist) or (axis, dist, dwell)")
        raise TypeError("Move must be Move or tuple")

    def _as_move_list(step_item: StepLike) -> List[Move]:
        if isinstance(step_item, list):
            return [_to_move(x) for x in step_item]
        return [_to_move(step_item)]

    axis_names = list(axes.keys())
    pos = {ax: 0.0 for ax in axis_names}

    # unit vectors, offsets, masses, friction
    u: Dict[str, np.ndarray] = {}
    r: Dict[str, np.ndarray] = {}
    m_kg: Dict[str, float] = {}
    fricN: Dict[str, float] = {}

    for ax, model in axes.items():
        uu = np.array(model.unit, dtype=float)
        n = np.linalg.norm(uu)
        if n <= 1e-12:
            raise ValueError(f"Axis {ax}: unit vector is zero.")
        u[ax] = uu / n
        r[ax] = np.array(model.r_offset_m, dtype=float)
        m_kg[ax] = float(model.mass_kg)
        fricN[ax] = float(getattr(model, "friction_N", 0.0))

    g_vec = np.array([0.0, 0.0, -float(g)])  # world gravity (Z up)

    # buffers
    t_chunks: List[np.ndarray] = []
    buf = {ax: {k: [] for k in ["x", "v", "a", "j", "F", "M"]} for ax in axis_names}

    t_cursor = [0.0]  # <-- mutable box; no nonlocal needed
    boundaries: List[Tuple[int, str, float, float]] = []
    move_id = 0

    def _force_moment_for_axis(ax_name: str,
                               a_scalar: np.ndarray,
                               v_scalar: np.ndarray,
                               v_eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          F_motor (N): force the motor must apply along +axis to achieve a and hold gravity
                       INCLUDING Coulomb friction effects with correct sign.
          M_mag (N*m): moment magnitude from r_offset x net (inertial+gravity) support force
                       (does not smear friction into CG moment).
        """
        # gravity projection along axis (+u)
        g_parallel = float(np.dot(g_vec, u[ax_name]))  # m/s^2

        # base motor force if no friction existed
        F_base = m_kg[ax_name] * (a_scalar - g_parallel)

        fric = fricN[ax_name]
        if fric > 0:
            # friction ON THE MASS opposes motion: F_fric_mass = -fric*sign(v)
            # near zero speed, oppose what you'd otherwise be trying to do (use F_base sign)
            dir_move = np.where(np.abs(v_scalar) > v_eps, np.sign(v_scalar), np.sign(F_base))
            F_fric_mass = -fric * dir_move
        else:
            F_fric_mass = np.zeros_like(F_base)

        # ✅ motor force required: F_motor + F_fric_mass = m(a - g_parallel)
        F_motor = F_base - F_fric_mass

        # Net support vector at CG from inertia+gravity (for your r_offset moment)
        a_vec = (a_scalar[:, None] * u[ax_name][None, :])  # Nx3
        F_support = m_kg[ax_name] * (a_vec - g_vec[None, :])  # Nx3

        M_vec = np.cross(r[ax_name][None, :], F_support)
        M_mag = np.linalg.norm(M_vec, axis=1)

        return F_motor, M_mag

    def append_chunk(t_rel: np.ndarray, per_axis: Dict[str, Dict[str, np.ndarray]]):
        t_abs = t_cursor[0] + t_rel

        # avoid duplicate boundary sample
        if t_chunks:
            t_abs = t_abs[1:]
            for ax in axis_names:
                for k in per_axis[ax].keys():
                    per_axis[ax][k] = per_axis[ax][k][1:]

        t_chunks.append(t_abs)
        for ax in axis_names:
            for k in ["x", "v", "a", "j", "F", "M"]:
                buf[ax][k].append(per_axis[ax][k])

        t_cursor[0] = float(t_abs[-1])

    # ---- main loop over steps ----
    for step in moves:
        step_moves = _as_move_list(step)
        cmd: Dict[str, Dict[str, Any]] = {}
        durations = []
        step_start = t_cursor[0]

        # build each axis command (motion + dwell), determine step duration
        for m in step_moves:
            if m.axis not in axes:
                raise KeyError(f"Move axis {m.axis!r} not in axes {axis_names}")

            lim = axes[m.axis].limits
            prof = design_7seg(m.dist, lim.vmax, lim.amax, lim.jmax)
            t_m, x_m, v_m, a_m, j_m = sample_7seg(prof, dt=dt)

            sgn = 1.0 if m.dist >= 0 else -1.0
            x_m = sgn * x_m
            v_m = sgn * v_m
            a_m = sgn * a_m
            j_m = sgn * j_m

            T_motion = float(t_m[-1]) if len(t_m) else 0.0

            # dwell arrays
            if m.dwell and m.dwell > 0:
                t_d = np.arange(0.0, m.dwell + dt / 2, dt)
                if len(t_m) > 0:
                    t_d = t_d[1:]
                x_d = np.full_like(t_d, x_m[-1] if len(x_m) else 0.0)
                v_d = np.zeros_like(t_d)
                a_d = np.zeros_like(t_d)
                j_d = np.zeros_like(t_d)

                t_rel = np.concatenate([t_m, T_motion + t_d])
                x_rel = np.concatenate([x_m, x_d])
                v_rel = np.concatenate([v_m, v_d])
                a_rel = np.concatenate([a_m, a_d])
                j_rel = np.concatenate([j_m, j_d])

                T_total = T_motion + float(m.dwell)
            else:
                t_rel, x_rel, v_rel, a_rel, j_rel = t_m, x_m, v_m, a_m, j_m
                T_total = T_motion

            cmd[m.axis] = {
                "t": t_rel, "x": x_rel, "v": v_rel, "a": a_rel, "j": j_rel,
                "dist": float(m.dist), "T_motion": T_motion, "dwell": float(m.dwell)
            }
            durations.append(T_total)

        # shared step timebase
        T_step = max(durations) if durations else 0.0
        t_step = np.arange(0.0, T_step + dt / 2, dt)
        if len(t_step) == 0:
            continue

        # assemble per-axis arrays
        per_axis: Dict[str, Dict[str, np.ndarray]] = {}
        for ax in axis_names:
            if ax in cmd:
                t_rel = cmd[ax]["t"]
                x_rel = cmd[ax]["x"]
                v_rel = cmd[ax]["v"]
                a_rel = cmd[ax]["a"]
                j_rel = cmd[ax]["j"]

                n = len(t_step)
                x = np.full(n, pos[ax] + (x_rel[-1] if len(x_rel) else 0.0), dtype=float)
                v = np.zeros(n, dtype=float)
                a = np.zeros(n, dtype=float)
                j = np.zeros(n, dtype=float)

                k = min(n, len(t_rel))
                x[:k] = pos[ax] + x_rel[:k]
                v[:k] = v_rel[:k]
                a[:k] = a_rel[:k]
                j[:k] = j_rel[:k]
            else:
                x = np.full_like(t_step, pos[ax], dtype=float)
                v = np.zeros_like(t_step, dtype=float)
                a = np.zeros_like(t_step, dtype=float)
                j = np.zeros_like(t_step, dtype=float)

            F_motor, M_mag = _force_moment_for_axis(ax, a, v)
            per_axis[ax] = {"x": x, "v": v, "a": a, "j": j, "F": F_motor, "M": M_mag}

        append_chunk(t_step, per_axis)

        # update positions + boundaries
        for ax, c in cmd.items():
            T_motion = c["T_motion"]
            dwell = c["dwell"]

            boundaries.append((move_id, ax, step_start, step_start + T_motion))
            if dwell and dwell > 0:
                boundaries.append((move_id, f"{ax}-dwell", step_start + T_motion, step_start + T_motion + dwell))

            pos[ax] += c["dist"]
            move_id += 1

    # finalize
    t = np.concatenate(t_chunks) if t_chunks else np.array([0.0])
    sim = {"t": t}
    for ax in axis_names:
        for k in ["x", "v", "a", "j", "F", "M"]:
            sim[f"{ax}_{k}"] = np.concatenate(buf[ax][k]) if buf[ax][k] else np.array([0.0])

    if len(t) > 2 and np.any(np.diff(t) <= 0):
        raise RuntimeError("Time is not strictly increasing. Decrease dt or check concatenation.")

    return (sim, boundaries) if return_boundaries else sim



# ============================================================
# Plotting (only active axes)
# ============================================================

def find_active_axes(sim: Dict[str, np.ndarray],
                     axes: List[str],
                     v_eps: float = 1e-6,
                     a_eps: float = 1e-6,
                     j_eps: float = 1e-6) -> List[str]:
    """Returns axes that appear to have moved (based on v/a/j thresholds)."""
    active = []
    for ax in axes:
        v = np.asarray(sim.get(f"{ax}_v", 0.0))
        a = np.asarray(sim.get(f"{ax}_a", 0.0))
        j = np.asarray(sim.get(f"{ax}_j", 0.0))
        if np.max(np.abs(v)) > v_eps or np.max(np.abs(a)) > a_eps or np.max(np.abs(j)) > j_eps:
            active.append(ax)
    return active


def plot_sequence(sim: Dict[str, np.ndarray],
                  axes: List[str],
                  show_legend: bool = True,
                  title: Optional[str] = None):
    t = sim["t"]
    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(12, 10))
    if title:
        fig.suptitle(title)

    def _plot_row(row, key_suffix, ylabel, ax_list):
        for ax_name in ax_list:
            axs[row].plot(t, sim[f"{ax_name}_{key_suffix}"], label=ax_name)
        axs[row].set_ylabel(ylabel)
        axs[row].grid(True, alpha=0.3)
        if show_legend and row == 0:
            axs[row].legend(ncols=min(6, len(ax_list)), fontsize=9)

    _plot_row(0, "x", "position", axes)
    _plot_row(1, "v", "velocity", axes)
    _plot_row(2, "a", "accel", axes)
    _plot_row(3, "j", "jerk", axes)
    _plot_row(4, "F", "force (N)", axes)
    _plot_row(5, "M", "moment (N·m)", axes)

    axs[-1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.show()


def plot_all(sim: Dict[str, np.ndarray],
             axes: List[str],
             show_legend_combined: bool = True,
             only_active: bool = True,
             v_eps: float = 1e-5,
             a_eps: float = 1e-6,
             j_eps: float = 1e-6):
    """Combined plot + per-axis plots; optionally filter to active axes only."""
    ax_list = axes
    if only_active:
        ax_list = find_active_axes(sim, axes, v_eps=v_eps, a_eps=a_eps, j_eps=j_eps)

    if len(ax_list) == 0:
        print("plot_all: no active axes detected; nothing to plot.")
        return

    plot_sequence(sim, axes=ax_list, show_legend=show_legend_combined, title=f"Combined: {', '.join(ax_list)}")
    for ax in ax_list:
        plot_sequence(sim, axes=[ax], show_legend=False, title=f"Axis: {ax}")


def print_boundaries(boundaries, digits: int = 3):
    fmt = f"{{:.{digits}f}}"
    print("Boundaries (move_index, label, t_start, t_end):")
    for i, label, ts, te in boundaries:
        print(f"({i}, {label!r}, {fmt.format(ts)}, {fmt.format(te)})")


# ============================================================
# Motor sizing vs conversion (mm/rev)
# ============================================================

OZIN_PER_NM = 141.6119323


def coupling_sweep_metrics(sim: Dict[str, np.ndarray],
                           axis: str,
                           mm_per_rev: np.ndarray,
                           eta: float = 0.9):
    """
    For a given axis and sweep of conversion (mm/rev), compute:
      - torque at peak |F|
      - torque at peak |v| (== peak RPM)
      - rpm at peak |F| time
      - peak rpm (at peak |v|)
    Assumes: tau = F * p / (2*pi*eta), rpm = 60 * v / p
    """
    mm_per_rev = np.asarray(mm_per_rev, dtype=float)
    p_m_per_rev = mm_per_rev / 1000.0

    F = np.asarray(sim[f"{axis}_F"], dtype=float)
    v = np.asarray(sim[f"{axis}_v"], dtype=float)

    i_F = int(np.argmax(np.abs(F)))
    i_v = int(np.argmax(np.abs(v)))

    F_pk = float(np.abs(F[i_F]))
    F_at_vpk = float(np.abs(F[i_v]))   # load at peak speed
    v_at_Fpk = float(np.abs(v[i_F]))
    v_pk = float(np.abs(v[i_v]))

    tau_at_Fpk = (F_pk * p_m_per_rev) / (2.0 * np.pi * eta)
    tau_at_vpk = (F_at_vpk * p_m_per_rev) / (2.0 * np.pi * eta)

    rpm_at_Fpk = 60.0 * (v_at_Fpk / p_m_per_rev)
    rpm_pk = 60.0 * (v_pk / p_m_per_rev)

    return tau_at_Fpk, tau_at_vpk, rpm_at_Fpk, rpm_pk


def plot_coupling_trade_twin_y(sim: Dict[str, np.ndarray],
                               axes: List[str],
                               mm_per_rev,
                               eta: float = 0.9,
                               only_active: bool = True,
                               v_eps: float = 1e-5):
    """
    One plot per axis, twin y-axes:
      - left: torque (oz·in) at peak |F| and at peak RPM (peak |v|)
      - right: rpm at peak |F| and peak rpm
    """
    ax_list = axes
    if only_active:
        ax_list = find_active_axes(sim, axes, v_eps=v_eps)

    if len(ax_list) == 0:
        print("plot_coupling_trade_twin_y: no active axes detected; nothing to plot.")
        return

    for ax in ax_list:
        tau_Fpk, tau_vpk, rpm_Fpk, rpm_pk = coupling_sweep_metrics(sim, ax, mm_per_rev, eta=eta)

        tau_Fpk_ozin = np.asarray(tau_Fpk) * OZIN_PER_NM
        tau_vpk_ozin = np.asarray(tau_vpk) * OZIN_PER_NM
        mm_per_rev_arr = np.asarray(mm_per_rev, dtype=float)

        fig, axL = plt.subplots(figsize=(10, 4.8))
        axR = axL.twinx()
        fig.suptitle(f"Motor sizing corners vs conversion: {ax} (eta={eta})")

        axL.plot(mm_per_rev_arr, tau_Fpk_ozin, label="Torque @ peak |F| (oz·in)")
        axL.plot(mm_per_rev_arr, tau_vpk_ozin, label="Torque @ peak RPM (oz·in)")
        axL.set_ylabel("Torque (oz·in)")

        axR.plot(mm_per_rev_arr, rpm_Fpk, label="RPM @ peak |F|")
        axR.plot(mm_per_rev_arr, rpm_pk, label="Peak RPM")
        axR.set_ylabel("RPM")

        axL.set_xlabel("Conversion (mm/rev)")
        axL.grid(True, alpha=0.3)

        h1, l1 = axL.get_legend_handles_labels()
        h2, l2 = axR.get_legend_handles_labels()
        axL.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)

        plt.tight_layout()
        plt.show()


# ============================================================
# Demo / entry point
# ============================================================

if __name__ == "__main__":
    # Your axes — full system — but plots will auto-filter to active motors
    axes = {
        "Xg": AxisLoadModel(
            limits=AxisLimits(vmax=0.5, amax=5, jmax=50),
            unit=(1.0, 0.0, 0.0),
            r_offset_m=(0.0, 0.0, 0.1),
            mass_kg=10.0,
            # friction_N=8.0,
            # drive=PulleyDrive(mm_per_rev=40.0, efficiency=0.96),
        ),
        "Yg": AxisLoadModel(
            limits=AxisLimits(vmax=0.5, amax=5, jmax=50),
            unit=(0.0, 1.0, 0.0),
            r_offset_m=(0.0, 0.0, 0.1),
            mass_kg=6.0,
            # friction_N=6.0,
            # drive=PulleyDrive(mm_per_rev=40.0, efficiency=0.96),
        ),
        "Zp1": AxisLoadModel(
            limits=AxisLimits(vmax=0.250, amax=2.500, jmax=25),
            unit=(0.0, 0.0, 1.0),
            r_offset_m=(0.03, 0.0, 0.0),
            mass_kg=1.0,
            # friction_N=10.0,
            # drive=BallScrewDrive(lead_mm_per_rev=5.0, shaft_diameter_mm=12.0, efficiency=0.90),
        ),
        "Zp2": AxisLoadModel(
            limits=AxisLimits(vmax=0.250, amax=2.500, jmax=25),
            unit=(0.0, 0.0, 1.0),
            r_offset_m=(0.03, 0.0, 0.0),
            mass_kg=1.0,
        ),
        "Zg": AxisLoadModel(
            limits=AxisLimits(vmax=0.250, amax=2.500, jmax=25),
            unit=(0.0, 0.0, 1.0),
            r_offset_m=(0.03, 0.0, 0.0),
            mass_kg=2.0,
        ),
        "Yb": AxisLoadModel(
            limits=AxisLimits(vmax=0.75, amax=7.5, jmax=75),
            unit=(0.0, 1.0, 0.0),
            r_offset_m=(0.0, 0.0, 0.1),
            mass_kg=6.0,
            friction_N=10.0,
            drive=PulleyDrive(mm_per_rev=40.0, efficiency=0.96),
        ),
        "Xd": AxisLoadModel(
            limits=AxisLimits(vmax=0.5, amax=5, jmax=50),
            unit=(1.0, 0.0, 0.0),
            r_offset_m=(0.0, 0.0, 0.1),
            mass_kg=6.0,
        ),
        "Xa": AxisLoadModel(
            limits=AxisLimits(vmax=0.5, amax=5, jmax=50),
            unit=(1.0, 0.0, 0.0),
            r_offset_m=(0.0, 0.0, 0.1),
            mass_kg=6.0,
            friction_N=10.0,
            drive=BallScrewDrive(lead_mm_per_rev=8.0, shaft_diameter_mm=12.0, efficiency=0.90),
        ),
        "Za": AxisLoadModel(
            limits=AxisLimits(vmax=0.250, amax=2.500, jmax=25),
            unit=(0.0, 0.0, 1.0),
            r_offset_m=(0.03, 0.0, 0.0),
            mass_kg=2.0,
            friction_N=2.0,
            drive=BallScrewDrive(lead_mm_per_rev=8.0, shaft_diameter_mm=12.0, efficiency=0.90),
        ),
    }

    moves_moveConsumable = [
        [("Xg", +0.500, 0.1), ("Yg", -0.10, 0.1)],
        ("Zg", -0.10, 1.0),
        ("Zg", +0.10, 0.1),
        ("Xg", -0.0500, 0.1),
        ("Zg", -0.10, 1.0),
        ("Zg", +0.10, 0.5),
        [("Xg", -0.450, 0.1), ("Yg", 0.10, 0.1)],
        ("Zg", -0.10, 1.0),
        ("Zg", +0.10, 0.5),
    ]

    moves_transferSample = [
        [("Xg", +0.2500, 0.1), ("Yg", -0.20, 0.1)],
        [("Zp1", +0.2500, 0.1), ("Zp2", -0.20, 0.1)],
        ("Zg", -0.10, 1.0),
        ("Zg", +0.10, 0.1),
        ("Xg", -0.0500, 0.1),
        ("Zg", -0.10, 1.0),
        ("Zg", +0.10, 0.5),
        [("Xg", -0.450, 0.1), ("Yg", 0.10, 0.1)],
        ("Zg", -0.10, 1.0),
        ("Zg", +0.10, 0.5),
    ]

    moves_Asp = [
        ("Yb", +0.04, 0.1),
        ("Za", -0.02, 0.1),
        ("Za", +0.005, 1.0),
        ("Za", +0.015, 0.1),
        [("Xa", -0.025, 0.1), ("Yb", 0.025, 0.1)],
        ("Za", -0.04, 0.1),
        ("Za", -0.05, 10.0),
        ("Za", +0.09, 0.1),
        [("Xa", +0.025, 0.1), ("Yb", -0.025, 0.1)],
        ("Za", -0.085, 1.0),
        ("Za", +0.095, 0.1),
    ]

    # pick one sequence
    sim, bounds = simulate_sequence(axes, moves_Asp, dt=0.0005, return_boundaries=True)

    mmrev = np.array([2, 4, 5, 8, 10, 16, 20], dtype=float)

    plot_coupling_trade_twin_y(sim, axes=list(axes.keys()), mm_per_rev=mmrev, eta=0.9,
                               only_active=True, v_eps=1e-5)

    plot_all(sim, axes=list(axes.keys()), only_active=True, v_eps=1e-5)

    print_boundaries(bounds, digits=3)
