import numpy as np
import pandas as pd
from cv_autoscatter import build_g_nu, S_bdg_from_G_NU, scattering_matrix_from_H

import sympy as sp
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import autoscatter.architecture_optimizer as arch_opt
import autoscatter.constraints as msc
import autoscatter.symbolic as sym

# -----------------------------
#   MicroComb helpers
# -----------------------------
def _expand_signed(int_list, interleave=True, include_zero=False):
    """Return signed list: [+n,-n,...]; keeps a single 0 if requested."""
    uniq = sorted(set(abs(int(n)) for n in int_list if n != 0))
    signed = [x for n in uniq for x in ((+n, -n) if interleave else (-n, +n))]
    if include_zero or 0 in int_list:
        signed = [0] + signed
    return signed

def _mode_labels(modes):
    return [("0" if m == 0 else f"{m:+d}") for m in modes]

# -----------------------------
#   Main builder
# -----------------------------
def setup_microcomb(
    f0_THz: float,
    delta_cavity_Hz: float,
    mode_nums,                 # e.g. [1,2,4]  -> [+1,-1,+2,-2,+4,-4]
    pump_nums,                 # e.g. [0,3]   -> [0,+3,-3]
    pump_amps=None,            # defaults to 1+0j for each pump
    interleave=True,
    build_scattering=True,
    kappa_N: np.ndarray | None = None,   # N×N (will be expanded to 2N×2N)
    Gamma_N: np.ndarray | None = None,   # N×N
):
    """
    Builds mode/pump freqs; calls build_g_nu; pretty-prints G, NU.
    If build_scattering=True, constructs S_bdg and S using provided κ_N, Γ_N
    (both N×N). If not provided, defaults to I_N.
    Returns a dict with all artifacts.
    """
    # --- expand signed indices ---
    modes = _expand_signed(mode_nums, interleave=interleave, include_zero=False)
    pumps_signed = _expand_signed(pump_nums, interleave=interleave, include_zero=True)

    # --- Hz -> THz ---
    delta_cavity_THz = float(delta_cavity_Hz) / 1e12

    # --- arrays & freqs ---
    modes_arr = np.array(modes, dtype=int)
    pumps_arr = np.array(sorted(set(pumps_signed)), dtype=int)

    mode_freqs_THz = f0_THz + modes_arr * delta_cavity_THz
    pump_freqs_THz = f0_THz + pumps_arr * delta_cavity_THz

    # --- pump amplitudes ---
    if pump_amps is None:
        pump_amps = np.ones(len(pumps_arr), dtype=complex)
    else:
        pump_amps = np.asarray(pump_amps, dtype=complex)
        assert pump_amps.shape[0] == pumps_arr.shape[0], "pump_amps length must match pumps"

    # --- build G, NU ---
    G, NU = build_g_nu(mode_freqs_THz, pump_freqs_THz, pump_amps)

    # --- pretty print ---
    labels = _mode_labels(modes_arr)
    print("FSR (THz):", delta_cavity_THz)
    print("Pumps (indices):", pumps_arr)
    print("Pump freqs (THz):", pump_freqs_THz)
    print("Pump amps:", pump_amps)
    print("Modes (indices):", modes_arr)
    print("Mode freqs (THz):", mode_freqs_THz)

    print("\nBeam-splitter matrix G:")
    print(pd.DataFrame(G, index=labels, columns=labels))
    print("\nSqueezing matrix NU:")
    print(pd.DataFrame(NU, index=labels, columns=labels))

    out = {
        "delta_cavity_THz": delta_cavity_THz,
        "modes": modes_arr,
        "mode_labels": labels,
        "mode_freqs_THz": mode_freqs_THz,
        "pumps": pumps_arr,
        "pump_freqs_THz": pump_freqs_THz,
        "pump_amps": pump_amps,
        "G": G,
        "NU": NU,
    }

    # --- scattering stage ---
    if build_scattering:
        N = G.shape[0]

        # defaults: I_N
        if kappa_N is None:
            kappa_N = np.eye(N, dtype=float)
        if Gamma_N is None:
            Gamma_N = np.eye(N, dtype=complex)

        # sanity
        kappa_N = np.asarray(kappa_N)
        Gamma_N = np.asarray(Gamma_N)
        assert kappa_N.shape == (N, N), "kappa_N must be N×N"
        assert Gamma_N.shape == (N, N), "Gamma_N must be N×N"

        S_bdg = S_bdg_from_G_NU(G, NU, kappa_N)
        print("\nScaled BdG matrix (S_bdg):")
        print(pd.DataFrame(S_bdg))

        S = scattering_matrix_from_H(S_bdg, Gamma_N, kappa_N)
        print("\nScattering matrix S:")
        print(pd.DataFrame(S))

        out.update({
            "kappa_N": kappa_N,
            "Gamma_N": Gamma_N,
            "S_bdg": S_bdg,
            "S": S
        })

    return out

# -----------------------------
# Examples
# -----------------------------

# One pump PoC
f0 = 193.5             # THz
delta_cavity = 199.9e9 # Hz (≈0.1999 THz)

res1 = setup_microcomb(
    f0_THz=f0,
    delta_cavity_Hz=delta_cavity,
    mode_nums=[1, 2],      # -> [+1,-1,+2,-2]
    pump_nums=[0],         # -> [0]

)

# Three pumps PoC: P0, P±3 ; 8 modes e.g. [±1, ±2, ±4, ±5]
N_demo = 8 # matches number of modes produced by [1,2,4,5]
kappa_diag = 0.9 * np.ones(N_demo)      # out-coupling fraction
Gamma_diag = 0.473 * np.ones(N_demo)    # system loss after microcomb

res2 = setup_microcomb(
    f0_THz=f0,
    delta_cavity_Hz=delta_cavity,
    mode_nums=[1, 2, 4, 5],
    pump_nums=[0, 3],
    kappa_N=np.diag(kappa_diag),        # N×N
    Gamma_N=np.diag(Gamma_diag),        # N×N
)

# -----------------------------
#   CV_S
# -----------------------------

def min_aux_modes(S, tol=1e-9):
    S = np.asarray(S, dtype=np.complex128)
    I = np.eye(S.shape[0], dtype=np.complex128)
    A = I - S.conj().T @ S
    B = I - S @ S.conj().T
    # Hermitize before rank to fight tiny numerical skew
    A = 0.5*(A + A.conj().T)
    B = 0.5*(B + B.conj().T)
    rA = np.linalg.matrix_rank(A, tol)
    rB = np.linalg.matrix_rank(B, tol)
    return max(rA, rB), rA, rB

my_S_target_np = res2["S"]                 # this is a numpy.ndarray

n_aux_min, rA, rB = min_aux_modes(res2["S"])
print("min auxiliaries =", n_aux_min, "(rA, rB) =", (rA, rB))

# robust conversion to SymPy (handles complex entries too)
my_S_target_sp = sp.Matrix(my_S_target_np.tolist())
""" 
optimizer = arch_opt.Architecture_Optimizer(
    S_target=my_S_target_sp,
    num_auxiliary_modes=16,
)
"""


# -----------------------------
#   isolator
# -----------------------------
S_target = sp.Matrix([[0,0],[1,0]])
optimizer = arch_opt.Architecture_Optimizer(
    S_target=sp.Matrix([[0,0],[1,0]]),
    num_auxiliary_modes=1,
)
"""
irreducible_graphs = optimizer.perform_breadth_first_search()

#node_colors = ['orange', 'orange', 'gray'] # the port modes are orange, the auxiliary mode gray
#msc.plot_list_of_graphs(irreducible_graphs, node_colors=node_colors)
"""



