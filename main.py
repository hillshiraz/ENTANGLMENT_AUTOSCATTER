# main.py
from __future__ import annotations
import numpy as np
import sympy as sp
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
from typing import List, Tuple, Dict, Any, Optional

#the article python codes
import autoscatter.architecture_optimizer as arch_opt
from autoscatter.scattering import Multimode_system
from autoscatter.architecture import translate_upper_triangle_coupling_matrix_to_conditions

#my own scripts
import microcomb as mc
import viz_couplings as viz

import os
os.environ["JAX_ENABLE_X64"] = "True"

import jax.numpy as jnp
import autoscatter.constraints as msc

def kappa_from_Q(f0_Hz: float, Q: float) -> float:
    """return kappa (rad/s) given optical frequency in Hz and Q."""
    return 2*np.pi*f0_Hz / Q

def split_g_and_nu(info, mode_types):
    c_matrix = np.asarray(info['coupling_matrix'])  #dimensionless
    N = c_matrix.shape[0]
    g = np.zeros((N,N), dtype=complex)
    nu = np.zeros((N,N), dtype=complex)

    #operators implied by mode_types: if True -> a, if False -> a_dagger
    def is_a(i): return bool(mode_types[i])

    for j in range(N):
        for i in range(j):  #upper triangle indices (i<j)
            if mode_types[i] == mode_types[j]:
                #BS
                val = c_matrix[i,j] if is_a(i) else -np.conj(c_matrix[i,j])
                g[i,j] = val; g[j,i] = np.conj(val)  #BS block
            else:
                #SPS
                val = c_matrix[i,j] if is_a(i) else -np.conj(c_matrix[i,j])
                nu[i,j] = val; nu[j,i] = val  #symmetric for nu
    return g, nu

if __name__ == "__main__":
    """ Simple Isolator """
    s_target= sp.Matrix([[0,0],[1,0]])

    opt = arch_opt.Architecture_Optimizer(
        S_target=s_target,
        num_auxiliary_modes=1
    )

    irreducible_graphs = opt.perform_breadth_first_search()

    optimal_sol = None
    best_info = None
    for idx, triu in enumerate(irreducible_graphs):
        conds = translate_upper_triangle_coupling_matrix_to_conditions(triu)
        success, info = opt.optimize_given_conditions(
            conditions=conds, method='L-BFGS-B', max_violation_success=1e-6
        )
        if success and (optimal_sol is None or info['final_cost'] < best_info['final_cost']):
            optimal_sol, best_info = idx, info

    print("Best graph index:", optimal_sol)
    print("Final cost:", best_info['final_cost'])
    print("G_eff:\n", np.asarray(best_info['effective_coupling_matrix']))
    print("S:\n", np.asarray(best_info['scattering_matrix']))

    mode_types = opt.mode_types
    g_target, nu_target = split_g_and_nu(best_info, mode_types)
    print("\n[g, ν] from split_g_and_nu():")
    print("g:\n", g_target)
    print("ν:\n", nu_target)

    kappa = 1
    out = mc.auto_microcomb(best_info, g_target, nu_target,  kappa, omega0=2*np.pi*193.5e12, FSR=2*np.pi*100e9,
                                 w_g=1, w_nu=1)

    print("pump teeth:", out["pump_indices"])
    print("pump |A|:", out["pump_amplitudes"])
    print("pump phases [rad]:", out["pump_phases"])
    print("FSR used:", out["FSR_out"])
    print("||g_err||^2:", out["g_error_fro2"], "||nu_err||^2:", out["nu_error_fro2"])

    viz.pump_report(out, freq_unit="THz")  #prints table

    N = len(out["mode_indices"])
    deltas = viz.extract_deltas_from_info(best_info, N)
    omega_modes = viz.mode_frequencies(omega0=2*np.pi*193.5e12, FSR=2*np.pi*100e9,
                                   mode_indices=out["mode_indices"], deltas=deltas)

    viz.mode_report(omega_modes, out["mode_indices"], freq_unit="THz")
    viz.plot_comb_with_modes(out, omega_modes, freq_unit="THz")

    np.set_printoptions(precision=6, suppress=True)
    g_opt = out["g_synth"]
    nu_opt = out["nu_synth"]

    print("\nOptimal synthesized g:")
    print(g_opt)
    print("\nOptimal synthesized ν:")
    print(nu_opt)

    # compare to targets (optional but handy)
    Gdiff = g_opt - g_target
    Nudiff = nu_opt - nu_target
    print("\n||g_err||^2 =", out["g_error_fro2"], "  ||ν_err||^2 =", out["nu_error_fro2"])
    print("max |g - g_target| =", np.max(np.abs(Gdiff)))
    print("max |ν - ν_target| =", np.max(np.abs(Nudiff)))

    total_err = out["g_error_fro2"] + out["nu_error_fro2"]
    print("Total error (g+nu):", total_err)