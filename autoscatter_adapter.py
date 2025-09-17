# autoscatter_adapter.py
"""
Adapter that wires inverse_synthesis.ModelFns to the project's scattering.py API.

We:
- Build (g, nu) from PumpEdge list according to the graph.
- Insert dispersion Delta on diag(g) as -Delta (rotating-frame convention).
- Split kappa into external vs intrinsic using a provided 'port mask' and an external fraction.
- Call scattering.calc_scattering_matrix(omegas, g, nu, kappa_int, kappa_ext).
- Return the annihilation-to-annihilation (upper-left) submatrix S_aa on selected ports.
"""

from __future__ import annotations
from typing import List, Tuple
import os
import numpy as np
import jax.numpy as jnp
from jax import config as jax_config

# Try to enable x64 if requested by env (optional)
if os.environ.get("JAX_ENABLE_X64", "").lower() in ("1", "true", "yes"):
    try:
        jax_config.update("jax_enable_x64", True)
    except Exception:
        pass

# Decide dtypes according to JAX x64 setting
_X64 = bool(jax_config.read("jax_enable_x64"))
DT_R = jnp.float64 if _X64 else jnp.float32
DT_C = jnp.complex128 if _X64 else jnp.complex64
ASSUME_UNIT_EXT_ON_PORTS = False

from inverse_synthesis import PumpEdge, Params, ModelFns

# Import user's scattering module
import sys
if "/mnt/data" not in sys.path:
    sys.path.append("/mnt/data")
import autoscatter.scattering as sc

def pumps_to_couplings_autoscatter(pumps: List[PumpEdge], N: int) -> Tuple[np.ndarray, np.ndarray]:
    g = np.zeros((N, N), dtype=np.complex128)  # numpy can stay 64-bit; JAX cast happens at boundary
    nu = np.zeros((N, N), dtype=np.complex128)
    for e in pumps:
        z = e.amp * np.exp(1j * e.phase)
        if e.type.upper() == "BS":
            g[e.i, e.j] += z
            g[e.j, e.i] += np.conj(z)
        elif e.type.upper() == "SPS":
            nu[e.i, e.j] += z
            nu[e.j, e.i] += z  # symmetric
        else:
            raise ValueError(f"Unknown pump type: {e.type}")
    return g, nu

def build_H_normalized_autoscatter(g: np.ndarray, nu: np.ndarray, delta: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """
    For compatibility only; inverse_synthesis doesn't actually need H if we directly call scattering.calc_scattering_matrix.
    We still provide a BdG-like H for potential diagnostics.
    """
    g = np.asarray(g, dtype=np.complex128).copy()
    nu = np.asarray(nu, dtype=np.complex128).copy()
    delta = np.asarray(delta, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    N = len(delta)
    for j in range(N):
        g[j, j] = g[j, j] - delta[j]  # insert -Delta on diagonal
    top = np.concatenate([g, nu], axis=1)
    bot = np.concatenate([np.conjugate(nu), np.conjugate(g)], axis=1)
    H = np.concatenate([top, bot], axis=0)
    # Normalize by sqrt(kappa) to mirror the scaled BdG (optional)
    D = np.concatenate([kappa, kappa])
    Dinv_sqrt = np.diag(1.0/np.sqrt(D + 1e-18))
    return Dinv_sqrt @ H @ Dinv_sqrt

def _split_kappa(total_kappa: np.ndarray, ports: List[int], external_fraction: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    If ASSUME_UNIT_EXT_ON_PORTS=True:
      ext = min(1.0, total) on ports; int = total - ext  (so for total=1+gamma -> ext=1, int=gamma)
      ext = 0, int = total on non-ports (aux modes are internal)
    Else:
      legacy behavior: ext = external_fraction * total on ports, int = total - ext.
    """
    N = total_kappa.shape[0]
    total = np.asarray(total_kappa, dtype=float)
    ext = np.zeros(N, dtype=float)
    if ASSUME_UNIT_EXT_ON_PORTS:
        # external coupling only on ports
        for p in ports:
            ext[p] = min(1.0, total[p])
        intr = np.clip(total - ext, 0.0, None)
    else:
        ext[ports] = external_fraction * total[ports]
        intr = np.maximum(total - ext, 0.0)

    return jnp.asarray(intr, dtype=DT_R), jnp.asarray(ext, dtype=DT_R)


def scattering_from_H_autoscatter(H_unused, port_idx: List[int], kappa_total: np.ndarray, omega: float = 0.0) -> np.ndarray:
    """
    Compute S_aa (annihilation-to-annihilation block) restricted to selected ports using scattering.calc_scattering_matrix.
    """
    # We need g, nu, Delta again; instead of relying on H, we reconstruct from Params outside.
    # However, inverse_synthesis only passes H here. We therefore compute S elsewhere.
    # We'll implement a convenience to compute S from module-level 'last_params' set by forward_S.
    global _LAST_FORWARD_CACHE
    if _LAST_FORWARD_CACHE is None:
        raise RuntimeError("Adapter cache is empty. Use forward_S via inverse_synthesis which populates the cache.")
    g, nu, delta = _LAST_FORWARD_CACHE['g'], _LAST_FORWARD_CACHE['nu'], _LAST_FORWARD_CACHE['delta']

    # insert dispersion on diag(g)
    g = g.copy()
    N = len(delta)
    for j in range(N):
        g[j, j] = g[j, j] - delta[j]

    kappa_int, kappa_ext = _split_kappa(kappa_total, port_idx, external_fraction=1.0)

    omegas = jnp.asarray([omega], dtype=DT_R)
    S_full, info = sc.calc_scattering_matrix(
        omegas,
        jnp.asarray(g, dtype=DT_C),
        jnp.asarray(nu, dtype=DT_C),
        jnp.asarray(kappa_int, dtype=DT_R),
        jnp.asarray(kappa_ext, dtype=DT_R),
    )
    # S_full shape: (len(omegas), 2N, 2N). We take the first slice and its upper-left N×N block (aa).
    S0 = S_full[0]
    # Upper-left block:
    S_aa = np.array(S0[:N, :N])

    # Select only the requested ports for in/out (rows/cols)
    S_ports = S_aa[np.ix_(port_idx, port_idx)]
    return S_ports

# A tiny cache so inverse_synthesis.forward can pass (g,nu,delta) into scattering_from_H
_LAST_FORWARD_CACHE = None

def make_modelfns_adapter(assume_unit_ext_on_ports: bool = False) -> ModelFns:
    global ASSUME_UNIT_EXT_ON_PORTS
    ASSUME_UNIT_EXT_ON_PORTS = bool(assume_unit_ext_on_ports)
    return ModelFns(
        pumps_to_couplings=pumps_to_couplings_autoscatter,
        build_H_normalized=build_H_normalized_autoscatter,
        scattering_from_H=scattering_from_H_autoscatter,
    )