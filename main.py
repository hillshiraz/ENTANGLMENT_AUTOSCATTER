# main.py
from __future__ import annotations
import os
import numpy as np
import sympy as sp


import jax
import jax.numpy as jnp
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
os.environ["JAX_ENABLE_X64"] = "True"

from typing import List, Tuple, Dict, Any, Optional

# the article python codes
import autoscatter.architecture_optimizer as arch_opt
from autoscatter.scattering import Multimode_system
from autoscatter.architecture import translate_upper_triangle_coupling_matrix_to_conditions
import autoscatter.constraints as msc

np.set_printoptions(linewidth=200)

# my own scripts
import microcomb as mc
import viz_couplings as viz

def kappa_from_Q(f0_Hz: float, Q: float) -> float:
    """return kappa (rad/s) given optical frequency in Hz and Q."""
    return 2 * np.pi * f0_Hz / Q


def split_g_and_nu(info, mode_types):
    c_matrix = np.asarray(info['coupling_matrix'])  # dimensionless
    N = c_matrix.shape[0]
    g = np.zeros((N, N), dtype=complex)
    nu = np.zeros((N, N), dtype=complex)

    # operators implied by mode_types: if True -> a, if False -> a_dagger
    def is_a(i): return bool(mode_types[i])

    for j in range(N):
        for i in range(j):  # upper triangle indices (i<j)
            if mode_types[i] == mode_types[j]:
                # BS
                val = c_matrix[i, j] if is_a(i) else -np.conj(c_matrix[i, j])
                g[i, j] = val
                g[j, i] = np.conj(val)  # BS block
            else:
                # SPS
                val = c_matrix[i, j] if is_a(i) else -np.conj(c_matrix[i, j])
                nu[i, j] = val
                nu[j, i] = val  # symmetric for nu
    return g, nu


# === move these OUT of the function (top-level) ===
class MinimalAddedInputNoise(msc.Base_Constraint):
    def __call__(self, scattering_matrix, coupling_matrix, kappa_int_matrix, mode_types):
        """
        Difference between added input photons and the quantum limit.
        """
        noise_matrix = (scattering_matrix - jnp.eye(scattering_matrix.shape[0])) @ jnp.complex_(
            jnp.sqrt(kappa_int_matrix)
        )
        total_noise = 0.5 * (
            jnp.sum(jnp.abs(scattering_matrix[0, 2:]) ** 2) +
            jnp.sum(jnp.abs(noise_matrix[0, :]) ** 2)
        )
        quantum_limit = 0.5
        return total_noise - quantum_limit


class MinimalAddedOutputNoise(msc.Base_Constraint):
    def __init__(self, Gval):
        self.Gval = Gval

    def __call__(self, scattering_matrix, coupling_matrix, kappa_int_matrix, mode_types):
        """
        Difference between added output photons and the quantum limit.
        """
        noise_matrix = (scattering_matrix - jnp.eye(scattering_matrix.shape[0])) @ jnp.complex_(
            jnp.sqrt(kappa_int_matrix)
        )
        total_noise = 0.5 * (
            jnp.sum(jnp.abs(scattering_matrix[1, 2:]) ** 2) +
            jnp.sum(jnp.abs(noise_matrix[1, :]) ** 2)
        )
        same_set = (mode_types[0] == mode_types[1])
        quantum_limit = (self.Gval - 1) / 2 if same_set else (self.Gval + 1) / 2
        return total_noise - quantum_limit
# ==================================================

#this is for the size of coupling matrix for some examples I want to run
def make_dummy_info_from_mode_count(N: int):
    # auto_microcomb only needs this to infer N
    return {"coupling_matrix": np.zeros((N, N), dtype=complex)}

if __name__ == "__main__":
    # 1 = isolator, 2 = amplifier (Gval=3), 3= simple SPS example of mine, 4 = both (run one after the other)
    RUN_CASE = 2

    # ---------- CASE 1: Simple Isolator ----------
    if RUN_CASE in (1, 4):
        print("\n=== CASE 1: Simple Isolator ===")
        S_target = sp.Matrix([[0, 0],
                              [1, 0]])

        opt = arch_opt.Architecture_Optimizer(
            S_target=S_target,
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

        if best_info is None:
            raise RuntimeError("No valid architecture found for CASE 1 (isolator).")

        print("Best graph index:", optimal_sol)
        print("Final cost:", best_info['final_cost'])
        print("G_eff:\n", np.asarray(best_info['effective_coupling_matrix']))
        print("S:\n", np.asarray(best_info['scattering_matrix']))

        mode_types = opt.mode_types
        g_target, nu_target = split_g_and_nu(best_info, mode_types)
        print("\n[g, ν] from split_g_and_nu():")
        print("g:\n", g_target)
        print("ν:\n", nu_target)

        kappa = 1.0
        out = mc.auto_microcomb(
            best_info, g_target, nu_target, kappa,
            omega0=2*np.pi*193.5e12, FSR=2*np.pi*100e9,
            w_g=1, w_nu=1, num_gens=5, separation=True,
    separation_mode="enforce",       # hard rule
    enforce_even_balance=True,       # when even #pumps -> equal on each side
    penalty_per_pump=0,           # lighter complexity penalty
    bias_initializer=True
        )

        print("pump teeth:", out["pump_indices"])
        print("pump |A|:", out["pump_amplitudes"])
        print("pump phases [rad]:", out["pump_phases"])
        print("FSR used:", out["FSR_out"])
        print("||g_err||^2:", out["g_error_fro2"], "||nu_err||^2:", out["nu_error_fro2"])

        viz.pump_report(out, freq_unit="THz")

        N = len(out["mode_indices"])
        deltas = viz.extract_deltas_from_info(best_info, N)
        omega_modes = viz.mode_frequencies(
            omega0=2*np.pi*193.5e12, FSR=2*np.pi*100e9,
            mode_indices=out["mode_indices"], deltas=deltas
        )

        viz.mode_report(omega_modes, out["mode_indices"], freq_unit="THz")
        viz.plot_comb_with_modes(out, omega_modes, freq_unit="THz")

        np.set_printoptions(precision=6, suppress=True)
        g_opt = out["g_synth"]
        nu_opt = out["nu_synth"]

        viz.plot_col_matrix(g_opt)
        viz.plot_col_matrix(nu_opt)


    # ---------- CASE 2: Amplifier with constraints ----------
    if RUN_CASE in (2, 4):
        print("\n=== CASE 2: Amplifier (with noise constraints) ===")
        Gval = 3.0
        S_target_2 = sp.Matrix([[0, 0],
                                [np.sqrt(Gval), 0]])

        enforced_constraints = [
            MinimalAddedInputNoise(),
            MinimalAddedOutputNoise(Gval=Gval),
        ]

        kwargs_optimization = {'num_tests': 10}
        solver_options = {'maxiter': 1000}

        optimizers = arch_opt.find_minimum_number_auxiliary_modes(
            S_target_2,
            enforced_constraints=enforced_constraints,
            allow_squeezing=True,
            port_intrinsic_losses=False,
            kwargs_optimization=kwargs_optimization,
            solver_options=solver_options,
        )

        opt_2 = optimizers[0]

        list_of_irreducible_graphs_2 = opt_2.perform_breadth_first_search()

        optimal_sol_2 = None
        best_info_2 = None

        for idx, triu in enumerate(list_of_irreducible_graphs_2):
            conds = translate_upper_triangle_coupling_matrix_to_conditions(triu)
            success, info = opt_2.optimize_given_conditions(
                conditions=conds,
                method='L-BFGS-B',
                max_violation_success=1e-6
            )
            if success and (optimal_sol_2 is None or info['final_cost'] < best_info_2['final_cost']):
                optimal_sol_2, best_info_2 = idx, info

        if best_info_2 is None:
            raise RuntimeError("No valid architecture found for CASE 2 (amplifier).")

        print("[Second example] Best graph index:", optimal_sol_2)
        print("[Second example] Final cost:", best_info_2['final_cost'])
        print("[Second example] G_eff:\n", np.asarray(best_info_2['effective_coupling_matrix']))
        print("[Second example] S:\n", np.asarray(best_info_2['scattering_matrix']))

        mode_types_2 = opt_2.mode_types
        g_target_2, nu_target_2 = split_g_and_nu(best_info_2, mode_types_2)
        print("\n[Second example] [g, ν] from split_g_and_nu():")
        print("g:\n", g_target_2)
        print("ν:\n", nu_target_2)

        kappa_2 = 1.0
        out_2 = mc.auto_microcomb(
            best_info_2, g_target_2, nu_target_2, kappa_2,
            omega0=2 * np.pi * 193.5e12, FSR=2 * np.pi * 100e9,
            w_g=1, w_nu=1, num_gens=5, separation=True,
            separation_mode="prefer",  # hard rule
            enforce_even_balance=True,  # when even #pumps -> equal on each side
            penalty_per_pump=0.25,  # lighter complexity penalty
            bias_initializer=True
        )


        print("\n[Second example] pump teeth:", out_2["pump_indices"])
        print("[Second example] pump |A|:", out_2["pump_amplitudes"])
        print("[Second example] pump phases [rad]:", out_2["pump_phases"])
        print("[Second example] FSR used:", out_2["FSR_out"])
        print("[Second example] ||g_err||^2:", out_2["g_error_fro2"], "||nu_err||^2:", out_2["nu_error_fro2"])

        viz.pump_report(out_2, freq_unit="THz")  # prints table

        N_2 = len(out_2["mode_indices"])
        deltas_2 = viz.extract_deltas_from_info(best_info_2, N_2)
        omega_modes_2 = viz.mode_frequencies(
            omega0=2*np.pi*193.5e12, FSR=2*np.pi*100e9,
            mode_indices=out_2["mode_indices"], deltas=deltas_2
        )

        viz.mode_report(omega_modes_2, out_2["mode_indices"], freq_unit="THz")
        viz.plot_comb_with_modes(out_2, omega_modes_2, freq_unit="THz")

        np.set_printoptions(precision=6, suppress=True)
        g_opt_2 = out_2["g_synth"]
        nu_opt_2 = out_2["nu_synth"]

        print("\n[Second example] Optimal synthesized g:")
        viz.plot_col_matrix(g_opt_2)

        print("\n[Second example] Optimal synthesized ν:")
        viz.plot_col_matrix(nu_opt_2)


    if RUN_CASE in (3, 4):
        print("\n=== CASE 3: Simple Squeezing ===")

        mode_idx_3 = np.array([-1, +1], dtype=int)

        N3 = len(mode_idx_3)

        # Targets: pure nu between the two outer modes; no g
        g_target_3 = np.zeros((N3, N3), dtype=complex)
        nu_target_3 = np.zeros((N3, N3), dtype=complex)

        # For 2-mode variant, indices are (0,1). For 3-mode, use (-1,+1) pair.
        if N3 == 2:
            k, l = 0, 1
        else:
            k = np.where(mode_idx_3 == -1)[0][0]
            l = np.where(mode_idx_3 == +1)[0][0]
        nu_target_3[k, l] = 1.0
        nu_target_3[l, k] = 1.0  # ν is symmetric in this model

        # Minimal stub "info" so GA knows N (= shape[0])
        info_3 = make_dummy_info_from_mode_count(N3)

        kappa_3 = 1.0

        out_3 = mc.auto_microcomb(
            info_3, g_target_3, nu_target_3, kappa_3,
            omega0=2 * np.pi * 193.5e12, FSR=2 * np.pi * 100e9,
            w_g=1.0, w_nu=1.0,
            num_gens=5,
            # === separation controls ===
            separation=False,  #turn on the constraint
            penalty_per_pump=1000
        )

        print("\n[Case 3] pump teeth:", out_3["pump_indices"])
        print("[Case 3] pump |A|:", out_3["pump_amplitudes"])
        print("[Case 3] pump phases [rad]:", out_3["pump_phases"])
        print("[Case 3] FSR used:", out_3["FSR_out"])
        print("[Case 3] ||g_err||^2:", out_3["g_error_fro2"], "||nu_err||^2:", out_3["nu_error_fro2"])

        viz.pump_report(out_3, freq_unit="THz")  # prints table

        # For synthetic runs we don't have real detunings; use zeros
        deltas_3 = np.zeros(len(out_3["mode_indices"]))
        omega_modes_3 = viz.mode_frequencies(
            omega0=2 * np.pi * 193.5e12, FSR=2 * np.pi * 100e9,
            mode_indices=out_3["mode_indices"], deltas=deltas_3
        )

        viz.mode_report(omega_modes_3, out_3["mode_indices"], freq_unit="THz")
        viz.plot_comb_with_modes(out_3, omega_modes_3, freq_unit="THz")

        np.set_printoptions(precision=6, suppress=True)
        g_opt_3 = out_3["g_synth"]
        nu_opt_3 = out_3["nu_synth"]

        print("\n[Case 3] Optimal synthesized g:\n", g_opt_3)
        viz.plot_col_matrix(g_opt_3)

        print("\n[Case 3] Optimal synthesized ν:\n", nu_opt_3)
        viz.plot_col_matrix(nu_opt_3)