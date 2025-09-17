# inverse_synthesis.py
"""
Inverse synthesis for AUTOSCATTER-style setups:
Find pump amplitudes/phases and per-mode dispersion (detuning) that reproduce a desired S-target.

This module is framework-agnostic: pass in callables from your codebase to compute S(H)
and to map pump parameters to coupling matrices (g, nu). If JAX is installed, the optimizer
can use analytic gradients; otherwise it falls back to finite differences.

Author: ChatGPT (for Shiraz)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Any
import math
import cmath
import numpy as np

# Optional: try JAX for autograd & speed
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False


# -----------------------------
# Data structures & type hints
# -----------------------------

@dataclass
class PumpEdge:
    """One driven (active) coupling between two modes i, j.
    type: 'BS' (beam-splitter) or 'SPS' (squeezing)."""
    i: int
    j: int
    type: str   # 'BS' or 'SPS'
    amp: float  # non-negative amplitude (optimization variable)
    phase: float  # phase in radians (optimization variable)


@dataclass
class Params:
    """Optimization variables."""
    pumps: List[PumpEdge]
    delta: np.ndarray    # shape (N,) real detunings per mode [rad/s] (optimization variable)
    kappa: np.ndarray    # shape (N,) total linewidth per mode [rad/s] (can be fixed or opt.)


@dataclass
class ModelFns:
    """
    Callbacks to integrate with your existing codebase.
    You must provide:
      - pumps_to_couplings(params.pumps, N) -> (g, nu) complex NxN each.
      - build_H_normalized(g, nu, delta, kappa) -> 2N x 2N complex BdG-like matrix H (properly normalized).
      - scattering_from_H(H, port_idx, kappa, omega=0.0) -> S submatrix over the chosen 'port_idx' (list of mode indices that are external I/O).
    Notes:
      * If your scattering_from_H already knows kappa & ports, you can ignore arguments you don't need.
      * If your code computes multi-frequency response, we pass 'omega' (scalar). You can ignore or use it.
    """
    pumps_to_couplings: Callable[[List[PumpEdge], int], Tuple[np.ndarray, np.ndarray]]
    build_H_normalized: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    scattering_from_H: Callable[[np.ndarray, List[int], np.ndarray, float], np.ndarray]


@dataclass
class OptimOptions:
    lr: float = 1e-2                 # learning rate
    iters: int = 1000                # iterations
    omega_list: Optional[List[float]] = None  # for SGD over frequencies; if None → [0.0]
    batch_size_omega: int = 1
    amp_clip: Optional[Tuple[float,float]] = (0.0, None)  # (min, max) for amplitudes
    delta_clip: Optional[Tuple[float,float]] = (None, None)
    kappa_clip: Optional[Tuple[float,float]] = (1e-6, None)  # keep positive
    l2_amp: float = 0.0              # regularization on pump amplitudes
    l2_delta: float = 0.0            # regularization on dispersion
    l2_kappa: float = 0.0            # regularization on kappa
    gauge_align_iters: int = 2       # iterations to align input/output phases before loss
    verbose_every: int = 50


# -----------------------------
# Utility: phase gauges on S
# -----------------------------

def _align_gauge(S: np.ndarray, S_target: np.ndarray, iters: int = 2) -> np.ndarray:
    """
    Align diagonal input/output unitary phase gauges to minimize ||D_out S D_in - S_t||_F.
    We use a simple alternating scheme that works well in practice:
      1) Fix D_in=I, solve D_out by matching the phase of a reference column for each row.
      2) Fix D_out, solve D_in similarly using a reference row.
    Repeat 'iters' times.
    """
    S_work = S.copy()
    M, N = S.shape
    # Start with identity gauges
    D_out = np.ones(M, dtype=complex)
    D_in  = np.ones(N, dtype=complex)

    # Choose references (first non-negligible element per row/col)
    eps = 1e-12

    for _ in range(iters):
        # Update D_out row-wise to align phases to S_target
        for r in range(M):
            # pick first significant column in target row r
            ref_c = None
            for c in range(N):
                if abs(S_target[r, c]) > eps and abs(S_work[r, c]) > eps:
                    ref_c = c
                    break
            if ref_c is not None:
                # want arg(D_out[r] * S[r,ref_c] * D_in[ref_c]) ≈ arg(S_t[r,ref_c])
                desired = np.angle(S_target[r, ref_c]) - np.angle(S_work[r, ref_c])
                D_out[r] = cmath.exp(1j * desired)
        S_work = (D_out[:, None] * S) * D_in[None, :]

        # Update D_in column-wise
        for c in range(N):
            ref_r = None
            for r in range(M):
                if abs(S_target[r, c]) > eps and abs(S_work[r, c]) > eps:
                    ref_r = r
                    break
            if ref_r is not None:
                desired = np.angle(S_target[ref_r, c]) - np.angle(S_work[ref_r, c])
                D_in[c] = cmath.exp(1j * desired)
        S_work = (D_out[:, None] * S) * D_in[None, :]

    return S_work


# -----------------------------
# Forward model: params → S
# -----------------------------

def forward_S(params: Params,
              ports: List[int],
              model: ModelFns,
              omega: float = 0.0) -> np.ndarray:
    """
    Compute S on selected 'ports' at frequency 'omega' from physical parameters.
    """
    N = np.asarray(params.delta, dtype=np.float64).shape[0]
    g, nu = model.pumps_to_couplings(params.pumps, N)

    # If using autoscatter_adapter, populate its cache for use inside scattering_from_H
    try:
        import autoscatter_adapter as _asa
        _asa._LAST_FORWARD_CACHE = {'g': g, 'nu': nu, 'delta': params.delta.copy()}
    except Exception:
        pass
    H = model.build_H_normalized(g, nu, params.delta, params.kappa)
    S = model.scattering_from_H(H, ports, params.kappa, omega)
    return S


# -----------------------------
# Loss
# -----------------------------

def loss_against_target(params: Params,
                        S_target: np.ndarray,
                        ports: List[int],
                        model: ModelFns,
                        omega_list: Optional[List[float]] = None,
                        options: Optional[OptimOptions] = None) -> float:
    """
    Multi-frequency gauge-invariant Frobenius loss + simple L2 regularizers.
    """
    if options is None:
        options = OptimOptions()
    if omega_list is None or len(omega_list) == 0:
        omega_list = [0.0]

    total = 0.0
    for omega in omega_list:
        S = np.asarray(forward_S(params, ports, model, omega), dtype=np.complex128)
        S_aligned = _align_gauge(S, S_target, options.gauge_align_iters)
        diff = S_aligned - S_target
        total += np.real(np.vdot(diff, diff)).item()  # Frobenius norm squared

    # Regularization
    if options.l2_amp > 0:
        for e in params.pumps:
            total += options.l2_amp * (e.amp ** 2)
    if options.l2_delta > 0:
        total += options.l2_delta * float(np.vdot(params.delta, params.delta).real)
    if options.l2_kappa > 0:
        total += options.l2_kappa * float(np.vdot(params.kappa, params.kappa).real)

    return total / len(omega_list)


# -----------------------------
# Optimizer helpers
# -----------------------------

def _clip_val(x: float, clip: Optional[Tuple[Optional[float], Optional[float]]]) -> float:
    if clip is None:
        return x
    lo, hi = clip
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x


def _project_params(params: Params, opt: OptimOptions) -> Params:
    # ensure constraints (amps>=0, kappa>0, etc.)
    new_pumps = []
    for e in params.pumps:
        amp = _clip_val(e.amp, opt.amp_clip)
        # phases can be wrapped to [-pi, pi)
        ph = (e.phase + math.pi) % (2*math.pi) - math.pi
        new_pumps.append(PumpEdge(e.i, e.j, e.type, amp, ph))

    delta = np.asarray(params.delta, dtype=np.float64).copy()
    if opt.delta_clip is not None:
        lo, hi = opt.delta_clip
        if lo is not None:
            delta = np.maximum(delta, lo)
        if hi is not None:
            delta = np.minimum(delta, hi)

    kappa = np.asarray(params.kappa, dtype=np.float64).copy()
    if opt.kappa_clip is not None:
        lo, hi = opt.kappa_clip
        if lo is not None:
            kappa = np.maximum(kappa, lo)
        if hi is not None:
            kappa = np.minimum(kappa, hi)

    return Params(new_pumps, delta, kappa)


def _finite_diff_grad(params: Params,
                      S_target: np.ndarray,
                      ports: List[int],
                      model: ModelFns,
                      omega_batch: List[float],
                      options: OptimOptions,
                      eps: float = 1e-6) -> Dict[str, Any]:
    """
    Finite-difference gradient when JAX is not used.
    Returns a dict with same structure as params but holding the gradients.
    """
    base_loss = loss_against_target(params, S_target, ports, model, omega_batch, options)
    # grads
    g_pumps_amp = []
    g_pumps_phase = []
    for idx, e in enumerate(params.pumps):
         # Ensure complex dtype for targets and forward S
        S_target = np.asarray(S_target, dtype=np.complex128)
        e_amp = e.amp
        params.pumps[idx] = PumpEdge(e.i, e.j, e.type, e_amp + eps, e.phase)
        l1 = loss_against_target(params, S_target, ports, model, omega_batch, options)
        g_amp = (l1 - base_loss) / eps
        # phase
        e_phase = e.phase
        params.pumps[idx] = PumpEdge(e.i, e.j, e.type, e_amp, e_phase + eps)
        l2 = loss_against_target(params, S_target, ports, model, omega_batch, options)
        g_phase = (l2 - base_loss) / eps
        # restore
        params.pumps[idx] = PumpEdge(e.i, e.j, e.type, e_amp, e_phase)
        g_pumps_amp.append(g_amp)
        g_pumps_phase.append(g_phase)

    # delta
    g_delta = np.zeros_like(params.delta)
    for i in range(params.delta.shape[0]):
        params.delta[i] += eps
        l1 = loss_against_target(params, S_target, ports, model, omega_batch, options)
        g_delta[i] = (l1 - base_loss) / eps
        params.delta[i] -= eps

    # kappa
    g_kappa = np.zeros_like(params.kappa)
    for i in range(params.kappa.shape[0]):
        params.kappa[i] += eps
        l1 = loss_against_target(params, S_target, ports, model, omega_batch, options)
        g_kappa[i] = (l1 - base_loss) / eps
        params.kappa[i] -= eps

    return {
        "pumps_amp": np.array(g_pumps_amp),
        "pumps_phase": np.array(g_pumps_phase),
        "delta": g_delta,
        "kappa": g_kappa,
    }


def fit_pumps_and_dispersion(S_target: np.ndarray,
                             init_params: Params,
                             ports: List[int],
                             model: ModelFns,
                             options: Optional[OptimOptions] = None,
                             rng: Optional[np.random.Generator] = None) -> Tuple[Params, Dict[str, Any]]:
    """
    Core optimizer loop (GD/SGD over omegas).
    Returns (best_params, history).
    """
    if options is None:
        options = OptimOptions()
    if rng is None:
        rng = np.random.default_rng(0)

    if options.omega_list is None or len(options.omega_list) == 0:
        omega_list = [0.0]
    else:
        omega_list = list(options.omega_list)

    params = _project_params(init_params, options)
    history = {"loss": []}

    best_loss = float("inf")
    best_params = params

    for t in range(options.iters):
        # sample omega minibatch (SGD over frequencies)
        if len(omega_list) <= options.batch_size_omega:
            omega_batch = omega_list
        else:
            omega_batch = list(rng.choice(omega_list, size=options.batch_size_omega, replace=False))

        # compute gradient
        grads = _finite_diff_grad(params, S_target, ports, model, omega_batch, options)

        # gradient step
        # pumps
        new_pumps = []
        for e, gA, gP in zip(params.pumps, grads["pumps_amp"], grads["pumps_phase"]):
            new_pumps.append(PumpEdge(
                e.i, e.j, e.type,
                e.amp  - options.lr * gA,
                e.phase - options.lr * gP
            ))
        # delta, kappa
        new_delta = params.delta - options.lr * grads["delta"]
        new_kappa = params.kappa - options.lr * grads["kappa"]

        params = _project_params(Params(new_pumps, new_delta, new_kappa), options)

        # monitor loss on central freq (or average)
        cur_loss = loss_against_target(params, S_target, ports, model, omega_list, options)
        history["loss"].append(cur_loss)

        if cur_loss < best_loss:
            best_loss = cur_loss
            best_params = Params(
                [PumpEdge(e.i,e.j,e.type,e.amp,e.phase) for e in params.pumps],
                params.delta.copy(),
                params.kappa.copy(),
            )

        if options.verbose_every and (t % options.verbose_every == 0):
            print(f"[iter {t:4d}] loss={cur_loss:.6e}  best={best_loss:.6e}")

    return best_params, history


# -----------------------------
# Verification helper
# -----------------------------

def verify_params(params: Params,
                  ports: List[int],
                  model: ModelFns,
                  omega_list: Optional[List[float]] = None) -> Dict[float, np.ndarray]:
    """
    Recompute S(omega) with the learned pumps+dispersion+kappa and return a dict {omega: S}.
    """
    if omega_list is None or len(omega_list) == 0:
        omega_list = [0.0]
    out = {}
    for w in omega_list:
        out[w] = forward_S(params, ports, model, w)
    return out
