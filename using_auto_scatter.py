import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib.lines import Line2D
from cv_autoscatter import build_g_nu, S_bdg_from_G_NU, scattering_matrix_from_H

import sympy as sp
import jax
jax.config.update("jax_enable_x64", True)
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

#make a colorful, easier way to observe the NU- matrix
def _parse_mode_int(label: str) -> int:
    return int(label)

def _default_value_str(z: complex, mode: str = "complex", precision: int = 3) -> str:
    if mode == "mag":
        return f"{abs(z):.{precision}g}"
    if mode == "real":
        return f"{np.real(z):.{precision}g}"
    if mode == "imag":
        return f"{np.imag(z):.{precision}g}i"
    if mode == "polar":
        r = abs(z); ang = np.degrees(np.angle(z))
        return f"{r:.{precision}g}∠{ang:.0f}°"
    a, b = np.real(z), np.imag(z)
    sgn = "+" if b >= 0 else "-"
    return f"{a:.{precision}g} {sgn} {abs(b):.{precision}g}i"

# --- main ---
def plot_coupling_circle(
    M: np.ndarray,
    mode_labels,
    mode_freqs_THz,
    *,
    min_abs: float = 0.0,
    rel_abs: float | None = None,
    max_edges: int | None = None,
    matrix_name: str = "NU",
    value_mode: str = "complex",
    value_precision: int = 3,
    title: str | None = None,
):
    M = np.asarray(M)
    assert M.ndim == 2 and M.shape[0] == M.shape[1], "M must be square"
    N = M.shape[0]
    assert len(mode_labels) == N, "mode_labels length must match M"
    mode_freqs_THz = np.asarray(mode_freqs_THz, float)
    assert mode_freqs_THz.shape[0] == N, "mode_freqs_THz length must match M"

    # --- node placement: +n on upper semicircle; −n antipodal; 0 at angle 0 ---
    vals = [_parse_mode_int(lbl) for lbl in mode_labels]
    idx_by_val = {v: i for i, v in enumerate(vals)}
    pos_vals = sorted([v for v in vals if v > 0])
    zero_present = (0 in vals)

    angles = np.zeros(N, float); used = set()
    if pos_vals:
        m = len(pos_vals)
        for k, v in enumerate(pos_vals, start=1):
            theta = np.pi * (k / (m + 1))  # (0, pi)
            ip = idx_by_val.get(+v); im = idx_by_val.get(-v)
            if ip is not None: angles[ip] = theta; used.add(ip)
            if im is not None: angles[im] = (theta + np.pi) % (2*np.pi); used.add(im)
    if zero_present:
        angles[idx_by_val[0]] = 0.0; used.add(idx_by_val[0])
    rem = [i for i in range(N) if i not in used]
    if rem:
        K = len(rem)
        for r, i in enumerate(rem):
            angles[i] = 2*np.pi*(r / K)

    R = 1.0
    xs, ys = R*np.cos(angles), R*np.sin(angles)

    # --- gather edges (upper triangle + diagonals), apply thresholds ---
    abs_all = np.abs(M[np.triu_indices(N)])
    overall_max = float(abs_all.max()) if abs_all.size else 0.0
    rel_thr = (rel_abs or 0.0) * overall_max
    eff_thr = max(min_abs, rel_thr)

    edges = []
    for i in range(N):
        for j in range(i, N):
            val = complex(M[i, j])
            mag = float(abs(val))
            if mag > eff_thr:
                edges.append((i, j, mag, val))

    if not edges:
        print("No connections above threshold.")
        return

    edges.sort(key=lambda e: e[2], reverse=True)
    if max_edges is not None:
        edges = edges[:max_edges]
    abs_max = max(e[2] for e in edges) if edges else 1.0

    def edge_lw(m):  # width scaling
        return 0.9 + 3.1 * (m / abs_max if abs_max > 0 else 0.0)

    # --- figure ---
    fig = plt.figure(figsize=(9.5, 8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.2])
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1]); ax_leg.axis("off")

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(title if title is not None else f"{matrix_name} — circular layout")

    #circle + nodes + labels
    ax.add_patch(plt.Circle((0, 0), R, fill=False, ls="--", alpha=0.3))
    ax.scatter(xs, ys, zorder=3)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x*1.08, y*1.08, f"{mode_labels[i]}\n{mode_freqs_THz[i]:.3f} THz",
                ha="center", va="center", fontsize=9)

    #colors
    cmap = plt.get_cmap("tab20")
    colors = [cmap(k % cmap.N) for k in range(len(edges))]

    #draw chords/loops
    legend_lines, legend_texts = [], []
    for k, (i, j, mag, val) in enumerate(edges):
        color = colors[k]; lw = edge_lw(mag)
        if i == j:
            loop_r = 0.08
            ax.add_patch(Arc((xs[i], ys[i]), width=2*loop_r, height=2*loop_r,
                             angle=np.degrees(angles[i]), theta1=30, theta2=330,
                             lw=lw, color=color, zorder=2))
            pair = f"{mode_labels[i]} (self)"
        else:
            line = Line2D([xs[i], xs[j]], [ys[i], ys[j]], lw=lw, color=color, alpha=0.95, zorder=1)
            ax.add_line(line)
            pair = f"{mode_labels[i]} – {mode_labels[j]}"

        # value shown in the bar (right panel)
        val_str = _default_value_str(val, mode=value_mode, precision=value_precision)
        legend_lines.append(Line2D([0], [0], color=color, lw=lw))
        legend_texts.append(f"{pair}  :  {val_str}")

    #right bar: pairs of modes with values of the elements on the matrix
    ax_leg.set_title("Connections (pair : value)", fontsize=11, pad=6)
    y_text = 0.95
    dy = 1.0 / max(10, len(legend_texts) + 2)
    for h, lab in zip(legend_lines, legend_texts):
        ax_leg.add_line(Line2D([0.02, 0.12], [y_text, y_text], transform=ax_leg.transAxes,
                               color=h.get_color(), lw=h.get_linewidth()))
        ax_leg.text(0.15, y_text, lab, transform=ax_leg.transAxes,
                    va="center", ha="left", fontsize=9)
        y_text -= dy

    # optional small footer showing thresholds used
    footer = f"threshold: |M| > {eff_thr:.2g}"
    if rel_abs: footer += f"  (min_abs={min_abs:g}, rel_abs={rel_abs:g} of max={overall_max:.2g})"
    ax_leg.text(0.02, 0.02, footer, transform=ax_leg.transAxes, fontsize=8, color="gray")

    plt.show()

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

# NU
plot_coupling_circle(res1["NU"], res1["mode_labels"], res1["mode_freqs_THz"],
                     min_abs=0, matrix_name="NU", value_mode="complex")

# G (with diagonal allowed)
plot_coupling_circle(res1["G"], res1["mode_labels"], res1["mode_freqs_THz"],
                     min_abs=0, matrix_name="G", value_mode="real")


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


# NU
plot_coupling_circle(res2["NU"], res2["mode_labels"], res2["mode_freqs_THz"],
                     min_abs=1e-6, matrix_name="NU", value_mode="complex")

# G
plot_coupling_circle(res2["G"], res2["mode_labels"], res2["mode_freqs_THz"],
                     min_abs=1e-6, matrix_name="G", value_mode="complex")

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



