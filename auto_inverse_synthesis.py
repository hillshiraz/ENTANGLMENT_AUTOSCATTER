# auto_inverse_synthesis.py
from __future__ import annotations
import numpy as np
import sympy as sp
from typing import List, Tuple, Dict, Any, Optional

from microcomb import CombSpec, propose_plan, plan_to_physical, compute_required_global_scale_and_powers
from viz_couplings import plot_couplings_freqline, plot_coupling_circle_from_edges, plot_pumps_vs_frequency_from_phys

import os
os.environ["JAX_ENABLE_X64"] = "True"

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
import jax.numpy as jnp
import autoscatter.constraints as msc

# Project modules
import sys
if "/mnt/data" not in sys.path:
    sys.path.append("/mnt/data")
import autoscatter.architecture_optimizer as arch_opt

from inverse_synthesis import PumpEdge, Params, OptimOptions, fit_pumps_and_dispersion, verify_params
from autoscatter_adapter import make_modelfns_adapter

def _sympy_to_numpy_S(S):
    if isinstance(S, sp.MatrixBase):
        return np.array(S.evalf(), dtype=complex)
    return np.array(S, dtype=complex)

def _extract_params_from_solution_dict(
    sol_full: Dict[str, Any],
    N: int,
    ports: list[int],
) -> Tuple[List[PumpEdge], np.ndarray, np.ndarray]:
    """Build PumpEdge list, delta vector, and port-gamma vector from solution_dict_complete."""
    pumps: List[PumpEdge] = []
    delta = np.zeros(N, dtype=float)
    gamma = np.zeros(N, dtype=float)  # port intrinsic losses (only meaningful on ports)

    # detunings Delta0, Delta1, ...
    for i in range(N):
        k = f"Delta{i}"
        if k in sol_full:
            delta[i] = float(sol_full[k])

    # collect BS/SPS magnitudes & phases and gammas
    g_amp, g_phase, nu_amp, nu_phase = {}, {}, {}, {}
    for k, v in sol_full.items():
        if k.startswith("|g_{"):
            idx = k[len("|g_{"):-2]     # 'i,j'
            g_amp[idx] = float(v)
        elif k.startswith("\\mathrm{arg}(g_{"):
            idx = k[len("\\mathrm{arg}(g_{"):-3]
            g_phase[idx] = float(v)
        elif k.startswith("|\\nu_{"):
            idx = k[len("|\\nu_{"):-2]
            nu_amp[idx] = float(v)
        elif k.startswith("\\mathrm{arg}(\\nu_{"):
            idx = k[len("\\mathrm{arg}(\\nu_{"):-3]
            nu_phase[idx] = float(v)
        # port intrinsic losses \gamma_i (only defined on port modes when port_intrinsic_losses=True)
        elif k.startswith("\\gamma_"):
            try:
                i = int(k.split("_")[-1])
                if i in ports:
                    gamma[i] = float(v)
            except Exception:
                pass

    def _parse_idx(s: str) -> Tuple[int, int]:
        a, b = s.split(",")
        return int(a), int(b)

    # beamsplitter edges
    for idx, amp in g_amp.items():
        if abs(amp) > 1e-12:
            ph = g_phase.get(idx, 0.0)
            i, j = _parse_idx(idx)
            pumps.append(PumpEdge(i=i, j=j, type="BS", amp=abs(amp), phase=ph))

    # squeezing edges
    for idx, amp in nu_amp.items():
        if abs(amp) > 1e-12:
            ph = nu_phase.get(idx, 0.0)
            i, j = _parse_idx(idx)
            pumps.append(PumpEdge(i=i, j=j, type="SPS", amp=abs(amp), phase=ph))

    return pumps, delta, gamma


def auto_inverse_synthesis(
    S_target: Any,
    num_auxiliary_modes: int,
    kappa_total: Optional[np.ndarray] = None,
    opt_options: Optional[OptimOptions] = None,
    optimizer_kwargs: Optional[dict] = None,
    allow_squeezing: bool = False,
    enforced_constraints: Optional[list] = None,
    port_intrinsic_losses: bool = False,
    # fast paths:
    use_min_aux_search: bool = True,
    mode_types_list: Optional[list] = None,
    # BFS throttles:
    max_graphs_per_level: Optional[int] = None,
    stop_after_irreducible: Optional[int] = None,
    # numeric fallback for FDA:
    seed_sps_on_bs: bool = True,
    sps_seed_amp: float = 1e-4,
) -> Dict[str, Any]:
    S_t_sym = S_target if isinstance(S_target, sp.MatrixBase) else sp.Matrix(S_target)
    num_ports = S_t_sym.shape[0]
    N = num_ports + num_auxiliary_modes
    ports = list(range(num_ports))

    # default kappa_total
    if kappa_total is None:
        kappa_total = np.ones(N, dtype=np.float64) * 0.1
    kappa_total = np.maximum(np.asarray(kappa_total, dtype=np.float64), 1e-12)

    # <<< build model with the right ext/int split policy
    model = make_modelfns_adapter(assume_unit_ext_on_ports=port_intrinsic_losses)

    if opt_options is None:
        opt_options = OptimOptions(lr=1e-2, iters=150, verbose_every=25)

    # choose assignments
    optimizers = []
    base_kwargs = dict(
        S_target=S_t_sym,
        num_auxiliary_modes=num_auxiliary_modes,
        port_intrinsic_losses=port_intrinsic_losses,
        enforced_constraints=(enforced_constraints or []),
        make_initial_test=False,
    )
    if optimizer_kwargs:
        base_kwargs.update(optimizer_kwargs)

    if mode_types_list is not None:
        for mt in mode_types_list:
            try:
                optimizers.append(arch_opt.Architecture_Optimizer(mode_types=list(mt), **base_kwargs))
            except Exception:
                continue
    elif use_min_aux_search and allow_squeezing:
        try:
            fm = arch_opt.find_minimum_number_auxiliary_modes(
                S_target=S_t_sym,
                start_value=num_auxiliary_modes,
                max_value=num_auxiliary_modes,
                allow_squeezing=True,
                port_intrinsic_losses=port_intrinsic_losses,
                kwargs_optimization=base_kwargs.get("kwargs_optimization", {}),
                solver_options=base_kwargs.get("solver_options", {}),
                enforced_constraints=base_kwargs.get("enforced_constraints", []),
            )
            if isinstance(fm, list):
                optimizers.extend(fm)
            elif fm is not None:
                optimizers.append(fm)
        except Exception:
            pass

    if not optimizers:
        if allow_squeezing:
            assignments = arch_opt.generate_all_possible_mode_assignments(
                num_port_modes=num_ports, num_auxiliary_modes=num_auxiliary_modes
            )
            for mt in assignments:
                try:
                    optimizers.append(arch_opt.Architecture_Optimizer(mode_types=list(mt), **base_kwargs))
                except Exception:
                    continue
        else:
            optimizers.append(arch_opt.Architecture_Optimizer(mode_types="no_squeezing", **base_kwargs))

    results = []
    global_best = {"loss": float("inf")}

    # BFS + numeric refine
    for opt_idx, optimizer in enumerate(optimizers):
        if max_graphs_per_level is not None or stop_after_irreducible is not None:
            optimizer.prepare_all_possible_combinations()
            irreducible = []
            for c in optimizer.unique_complexity_levels:
                potentials = optimizer.identify_potential_combinations(c)
                if max_graphs_per_level is not None:
                    potentials = potentials[:max_graphs_per_level]
                optimizer.find_valid_combinations(c, combinations_to_test=potentials)
                optimizer.cleanup_valid_combinations()
                irreducible = optimizer.valid_combinations
                if stop_after_irreducible is not None and len(irreducible) >= stop_after_irreducible:
                    irreducible = irreducible[:stop_after_irreducible]
                    break
            irreducible_graphs = np.array(irreducible, dtype="int8")
        else:
            irreducible_graphs = optimizer.perform_breadth_first_search()

        if irreducible_graphs is None or len(irreducible_graphs) == 0:
            results.append({"optimizer_index": opt_idx, "error": "no irreducible graphs found"})
            continue

        for gidx, triu_arch in enumerate(irreducible_graphs):
            try:
                success, info = optimizer.optimize_given_conditions(triu_matrix=triu_arch, verbosity=False)
                if not success:
                    results.append({"optimizer_index": opt_idx, "graph_index": gidx, "error": "symbolic stage not successful"})
                    continue

                sol_full = info["solution_dict_complete"]

                # get gamma (port intrinsic losses)
                pumps, delta, gamma = _extract_params_from_solution_dict(sol_full, N, ports)

                # allow SPS growth numerically if symbolic gave only BS
                if seed_sps_on_bs and not any(e.type.upper() == "SPS" for e in pumps):
                    bs_pairs = {(e.i, e.j) for e in pumps if e.type.upper() == "BS"}
                    for (i, j) in bs_pairs:
                        pumps.append(PumpEdge(i=i, j=j, type="SPS", amp=sps_seed_amp, phase=0.0))

                # <<< seed kappa: on ports set total = 1 + gamma_i when port_intrinsic_losses=True
                kappa_seed = kappa_total.copy()
                if port_intrinsic_losses:
                    for p in ports:
                        kappa_seed[p] = max(1.0 + float(gamma[p]), 1e-9)

                init_params = Params(pumps=pumps, delta=delta, kappa=kappa_seed)

                best_params, hist = fit_pumps_and_dispersion(
                    S_target=_sympy_to_numpy_S(S_t_sym).astype(np.complex128),
                    init_params=init_params,
                    ports=ports,
                    model=model,  # <<< uses the adapter with unit ext on ports if requested
                    options=opt_options,
                )
                final_loss = hist["loss"][-1] if len(hist.get("loss", ())) else np.inf
                if not np.isfinite(final_loss):
                    results.append({"optimizer_index": opt_idx, "graph_index": gidx, "error": f"non-finite loss: {final_loss}"})
                    continue

                item = {
                    "optimizer_index": opt_idx,
                    "mode_types": getattr(optimizer, "mode_types", None),
                    "graph_index": gidx,
                    "triu": triu_arch,
                    "best_params": best_params,
                    "history": hist,
                    "loss": final_loss,
                }
                results.append(item)

                if final_loss < global_best["loss"]:
                    global_best = {
                        "optimizer_index": opt_idx,
                        "mode_types": getattr(optimizer, "mode_types", None),
                        "graph_index": gidx,
                        "best_params": best_params,
                        "history": hist,
                        "loss": final_loss,
                        "triu": triu_arch,
                    }

            except Exception as e:
                results.append({"optimizer_index": opt_idx, "graph_index": gidx, "error": str(e)})
                continue

    candidates = [r for r in results if "best_params" in r and np.isfinite(r.get("loss", np.inf))]
    if not candidates:
        errs = [r.get("error") for r in results if "error" in r]
        raise RuntimeError(f"No successful graph/fit. allow_squeezing={allow_squeezing}. "
                           f"Tried {len(results)} attempts. Errors: {errs}")

    best_item = min(candidates, key=lambda r: r["loss"])
    best = {
        "optimizer_index": best_item["optimizer_index"],
        "mode_types": best_item["mode_types"],
        "graph_index": best_item["graph_index"],
        "best_params": best_item["best_params"],
        "history": best_item["history"],
        "loss": best_item["loss"],
        "triu": best_item["triu"],
    }
    return {"best": best, "all_results": results, "ports": ports, "N": N}



if __name__ == "__main__":
    # isolator 2x2 as S_target, 1 aux mode
    S_target = sp.Matrix([[0,0],[1,0]])
    out = auto_inverse_synthesis(S_target, num_auxiliary_modes=1)
    print("Best loss:", out["best"]["loss"])
    bp = out["best"]["best_params"]
    for e in bp.pumps:
        print(e)
    print("delta:", bp.delta, "kappa:", bp.kappa)

    #the opposite direction
    S_map = verify_params(bp, ports=out["ports"], model=make_modelfns_adapter(), omega_list=[0.0])
    print("S[NXN] for isolator:\n", S_map[0.0])

    #this part use a microcomb based on the optimization
    edges_from_fit = [e for e in bp.pumps if e.type.upper() == "BS" and e.amp >= 1e-4]

    comb = CombSpec(f0_THz=193.518, FSR_GHz=199.9, pump_lines=None, mode_lines=None)
    plan = propose_plan(
        edges_from_fit,
        comb,
        allowed_edge_types=("BS",),  # BS only
        min_edge_amp=1e-4,  # ignore tiny junk
        forbid_pump_on_mode_lines=True,  # keep pumps off your signal-mode lines
        auto_max_pumps=3,
        auto_max_span=6,
    )
    phys = plan_to_physical(plan)
    print("Pump freqs (THz):", phys["pump_freqs_THz"])
    print("Pump phases (rad):", phys["pump_phase_rad"])
    print("Relative powers:", phys["pump_rel_power"])
    print("Mode freqs (THz):", phys["mode_freqs_THz"])
    print("Edge realization:", phys["edge_realization"])

    # --- Build edge lists for plotting ---
    edges_bs = [e for e in bp.pumps if e.type.upper() == "BS" and e.amp >= 1e-4]
    edges_sps = [e for e in bp.pumps if e.type.upper() == "SPS" and e.amp >= 1e-4]

    # Mode frequencies (THz) for plotting — use the comb plan output
    mode_freqs = phys["mode_freqs_THz"]  # dict {mode_index: f_THz}

    # === BS only ===
    # (A) Frequency-axis view (arches)
    plot_couplings_freqline(
        edges_bs,
        mode_freqs_THz=mode_freqs,
        show_types=("BS",),  # BS only
        min_amp=1e-4,
        title="BS couplings (frequency axis)"
    )

    # (B) Circular view (matrix-style chords)
    plot_coupling_circle_from_edges(
        edges_bs,
        mode_freqs_THz=mode_freqs,
        which="BS",
        min_abs=1e-4,
        title="g (BS)"
    )

    # === SPS (only if present and non-negligible) ===
    if len(edges_sps) > 0:
        # (A) Frequency-axis view (arches)
        plot_couplings_freqline(
            edges_sps,
            mode_freqs_THz=mode_freqs,
            show_types=("SPS",),  # SPS only
            min_amp=1e-4,
            title="SPS couplings (frequency axis)"
        )

        # (B) Circular view
        plot_coupling_circle_from_edges(
            edges_sps,
            mode_freqs_THz=mode_freqs,
            which="SPS",
            min_abs=1e-4,
            title="nu (SPS)"
        )

        plot_pumps_vs_frequency_from_phys(
            phys,
            annotate=True,
            y_unit="rel. power",
            title="Pumps vs Frequency (relative power)"
        )

    """
    #Fully directional amplifier, 1 aux mode
    # define constraints (as in the notebook)
    # constraints (same as before)
    class MinimalAddedInputNoise(msc.Base_Constraint):
        def __call__(self, S, Hsig, Kint, mode_types):
            N = S.shape[0]
            noise = (S - jnp.eye(N)) @ jnp.complex_(jnp.sqrt(Kint))
            return 0.5 * (jnp.sum(jnp.abs(S[0, 2:]) ** 2) + jnp.sum(jnp.abs(noise[0, :]) ** 2)) - 0.5


    class MinimalAddedOutputNoise(msc.Base_Constraint):
        def __init__(self, Gval): self.Gval = Gval

        def __call__(self, S, Hsig, Kint, mode_types):
            N = S.shape[0]
            noise = (S - jnp.eye(N)) @ jnp.complex_(jnp.sqrt(Kint))
            total = 0.5 * (jnp.sum(jnp.abs(S[1, 2:]) ** 2) + jnp.sum(jnp.abs(noise[1, :]) ** 2))
            same_set = (mode_types[0] == mode_types[1])
            qlim = (self.Gval - 1) / 2 if same_set else (self.Gval + 1) / 2
            return total - qlim


    Gval = 5
    S_target_fda = sp.Matrix([[0, 0], [sp.sqrt(Gval), 0]])
    enforced = [MinimalAddedInputNoise(), MinimalAddedOutputNoise(Gval)]

    out_2 = auto_inverse_synthesis(
        S_target_fda,
        num_auxiliary_modes=1,
        allow_squeezing=True,
        enforced_constraints=enforced,
        port_intrinsic_losses=False,
        mode_types_list=[
            [True, True, True, False],  # particles/holes pattern from the notebook
            [True, False, True, False],
        ],
        max_graphs_per_level=20,
        stop_after_irreducible=7,
        seed_sps_on_bs=True,
        sps_seed_amp=1e-3,
        optimizer_kwargs={
            "kwargs_optimization": {"num_tests": 6, "interrupt_if_successful": True},
            "solver_options": {"maxiter": 1200},
        },
    )
    bp_2 = out_2["best"]["best_params"]
    S_map_2 = verify_params(bp_2, ports=out_2["ports"],
                            model=make_modelfns_adapter(assume_unit_ext_on_ports=False),
                            omega_list=[0.0])
    print("loss:", out_2["best"]["loss"])
    print("|S21|:", np.abs(S_map_2[0.0][1, 0]), " |S12|:", np.abs(S_map_2[0.0][0, 1]))
"""


    # Note: once you have a calibration A_p = α * sqrt(P_p[W]),
    # you can turn 'relative powers' into Watts: P_p = (s_amp/α)^2 * pump_rel_power[p].