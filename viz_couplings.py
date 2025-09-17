# viz_couplings.py
from __future__ import annotations
import numpy as np
from typing import Iterable, Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Arc
from dataclasses import dataclass

# ---- Data model for edges (matches your optimizer output) ----
# If you already import PumpEdge from your codebase, remove this dataclass and import instead.
@dataclass
class PumpEdge:
    i: int
    j: int
    type: str   # 'BS' or 'SPS'
    amp: float
    phase: float  # radians

# ---- Build BS (g) and SPS (nu) matrices from a list of edges ----
def edges_to_matrices(edges: Iterable[PumpEdge], N: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of PumpEdge into coupling matrices:
      - g  (Hermitian) for beamsplitter couplings
      - nu (symmetric) for two-mode squeezing couplings
    """
    if N is None:
        N = 1 + max(max(e.i, e.j) for e in edges)
    g  = np.zeros((N, N), dtype=complex)  # BS
    nu = np.zeros((N, N), dtype=complex)  # SPS
    for e in edges:
        z = e.amp * np.exp(1j * e.phase)
        if e.type.upper() == "BS":
            # Hermitian placement for BS
            g[e.i, e.j]  += z
            g[e.j, e.i]  += np.conj(z)
        elif e.type.upper() == "SPS":
            # Symmetric placement for SPS (Bogoliubov block)
            nu[e.i, e.j] += z
            nu[e.j, e.i] += z
    return g, nu

# ---- Utility: normalize/parse mode frequencies (THz) ----
def _coerce_mode_freqs(mode_freqs_THz: Dict[int, float] | np.ndarray, N: int) -> np.ndarray:
    """
    Accepts either:
      - dict {mode_index: f_THz}
      - numpy array of length N, where entry k is mode k's frequency
    Returns a numpy array aligned by mode index [0..N-1].
    """
    if isinstance(mode_freqs_THz, dict):
        arr = np.zeros(N, float)
        for k in range(N):
            arr[k] = float(mode_freqs_THz[k])
        return arr
    arr = np.asarray(mode_freqs_THz, float)
    if arr.shape[0] != N:
        raise ValueError("mode_freqs_THz length must equal number of modes")
    return arr

# ---- Helper: quadratic Bezier arc between two x-positions on a frequency line ----
def _quad_arc(x0, x1, h=0.25):
    """
    Build a quadratic Bezier path from (x0,0) to (x1,0)
    with a control point at height h*|x1-x0| to create an arch.
    """
    xm = 0.5*(x0 + x1)
    height = h * max(1e-12, abs(x1-x0))
    verts = [(x0, 0.0), (xm, height), (x1, 0.0)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    return Path(verts, codes)

# ---- Main frequency-axis plot (THz on x, couplings as arches) ----
def plot_couplings_freqline(
    edges: Iterable[PumpEdge],
    mode_freqs_THz: Dict[int, float] | np.ndarray,
    *,
    min_amp: float = 0.0,                    # drop edges with |amp| below this
    show_types: Tuple[str, ...] = ("BS", "SPS"),   # choose ("BS",) to see BS only
    scale_per_type: bool = True,             # separate line-width scaling per type
    title: Optional[str] = None,
    legend_value_mode: str = "polar",        # 'mag'/'polar'/'complex' (legend shows only types here)
):
    """
    Draw BS and/or SPS couplings between modes positioned at their actual frequencies (THz).
    - BS edges in color 'C0'
    - SPS edges in color 'C3'
    - Edge width encodes amplitude (optionally scaled independently per type).
    """
    # Filter edges by type and amplitude
    allowed_types = {t.upper() for t in show_types}
    E = [e for e in edges if e.type.upper() in allowed_types and abs(e.amp) >= float(min_amp)]
    if not E:
        print("No edges to plot (after filtering).")
        return

    N = 1 + max(max(e.i, e.j) for e in E)
    freqs = _coerce_mode_freqs(mode_freqs_THz, N)

    # Place modes along the frequency axis (sorted by frequency)
    order = np.argsort(freqs)
    x = freqs[order]
    idx_pos = {int(order[k]): k for k in range(N)}  # map mode index -> x-position index

    # Figure scaffolding
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhline(0, color="0.85", lw=1)
    ax.set_yticks([])
    ax.set_xlabel("Frequency [THz]")
    ax.set_xlim(x.min() - 0.1*(x.ptp()+1e-12), x.max() + 0.1*(x.ptp()+1e-12))
    ax.set_title(title or "Couplings on frequency axis")

    # Draw the modes (points + labels)
    ax.scatter(x, np.zeros_like(x), zorder=3)
    for k, m in enumerate(order):
        ax.text(x[k], 0.06, f"mode {m}\n{freqs[m]:.3f} THz", ha="center", va="bottom", fontsize=8)

    # Colors by type
    color_map = {"BS": "C0", "SPS": "C3"}

    # Line-width scaling per type or global
    mags_by_type = {}
    for t in ("BS","SPS"):
        mags_by_type[t] = [abs(e.amp) for e in E if e.type.upper()==t]
    max_global = max([max(v) for v in mags_by_type.values() if v] or [1.0])

    def edge_lw(e: PumpEdge):
        t = e.type.upper()
        m = abs(e.amp)
        denom = (max(mags_by_type[t]) if (scale_per_type and mags_by_type[t]) else max_global)
        r = (m / denom) if denom > 0 else 0.0
        return 0.8 + 3.5 * r

    # Draw arches
    legend_items = []
    seen_type = set()
    for e in sorted(E, key=lambda q: abs(q.amp), reverse=True):
        i, j = e.i, e.j
        xi, xj = x[idx_pos[i]], x[idx_pos[j]]
        path = _quad_arc(xi, xj, h=0.20)
        patch = PathPatch(path, facecolor="none", edgecolor=color_map[e.type.upper()], lw=edge_lw(e), alpha=0.95)
        ax.add_patch(patch)

        # One legend handle per type
        if e.type.upper() not in seen_type:
            legend_items.append(Line2D([0], [0], color=color_map[e.type.upper()], lw=2.5, label=e.type.upper()))
            seen_type.add(e.type.upper())

    ax.legend(handles=legend_items, loc="upper left", frameon=False)
    plt.tight_layout()
    plt.show()

# ---- Circle plot wrapper: build g/nu from edges, then draw a circular connection map ----
def _format_complex(z: complex, mode="polar", prec=3):
    if mode == "mag":   return f"{abs(z):.{prec}g}"
    if mode == "real":  return f"{np.real(z):.{prec}g}"
    if mode == "imag":  return f"{np.imag(z):.{prec}g}i"
    if mode == "polar":
        r, ang = abs(z), np.degrees(np.angle(z))
        return f"{r:.{prec}g}∠{ang:.0f}°"
    a, b = np.real(z), np.imag(z)
    sgn = "+" if b>=0 else "-"
    return f"{a:.{prec}g} {sgn} {abs(b):.{prec}g}i"

def plot_coupling_circle_from_edges(
    edges: Iterable[PumpEdge],
    mode_freqs_THz: Dict[int,float] | np.ndarray,
    *,
    which: str = "BS",              # "BS" or "SPS"
    min_abs: float = 0.0,           # drop entries with |value| < min_abs
    value_mode: str = "polar",      # how to annotate edge values
    title: Optional[str] = None,
):
    """
    Convenience: pick either BS or SPS, convert edges to the respective matrix (g or nu),
    and draw a circular layout (nodes on a circle, chords for |value|>0).
    """
    E = [e for e in edges if e.type.upper()==which.upper() and abs(e.amp) >= min_abs]
    if not E:
        print(f"No {which} edges to plot (after filtering).")
        return
    N = 1 + max(max(e.i, e.j) for e in E)
    M_bs, M_sps = edges_to_matrices(E, N)
    M = M_bs if which.upper()=="BS" else M_sps

    freqs = _coerce_mode_freqs(mode_freqs_THz, N)
    labels = [str(i) for i in range(N)]
    _plot_coupling_circle_core(M, labels, freqs, matrix_name=which.upper(), value_mode=value_mode, title=title)

def _plot_coupling_circle_core(M, mode_labels, mode_freqs_THz, *, matrix_name="M", value_mode="polar", title=None):
    """
    Core circular plot: draws nodes on a circle and connects i<=j entries with chords/loops.
    """
    M = np.asarray(M); N = M.shape[0]
    # Uniform node angles
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    R = 1.0
    xs, ys = R*np.cos(angles), R*np.sin(angles)

    # Collect edges (upper triangle incl. diagonal)
    entries = []
    for i in range(N):
        for j in range(i, N):
            z = complex(M[i,j]); mag = abs(z)
            if mag > 0:
                entries.append((i,j,mag,z))
    if not entries:
        print("Empty matrix.")
        return
    abs_max = max(e[2] for e in entries)
    def lw(m): return 0.9 + 3.1*(m/abs_max if abs_max>0 else 0)

    fig = plt.figure(figsize=(9.5, 8), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(plt.Circle((0,0), R, fill=False, ls="--", alpha=0.3))
    ax.set_title(title or f"{matrix_name} — circular layout")

    # Nodes + labels (include frequency for context)
    ax.scatter(xs, ys, zorder=3)
    for i,(x,y) in enumerate(zip(xs,ys)):
        ax.text(x*1.08, y*1.08, f"{mode_labels[i]}\n{mode_freqs_THz[i]:.3f} THz",
                ha="center", va="center", fontsize=9)

    colors = plt.get_cmap("tab20")
    for k,(i,j,mag,val) in enumerate(sorted(entries, key=lambda e:e[2], reverse=True)):
        c = colors(k % 20)
        if i==j:
            # self-loop for diagonal entries
            ax.add_patch(Arc((xs[i], ys[i]), width=0.16, height=0.16,
                             angle=np.degrees(angles[i]), theta1=30, theta2=330,
                             lw=lw(mag), color=c))
        else:
            ax.add_line(Line2D([xs[i], xs[j]], [ys[i], ys[j]], lw=lw(mag), color=c, alpha=0.95))
        # annotate with value
        xm, ym = (xs[i]+xs[j])/2, (ys[i]+ys[j])/2
        ax.text(xm, ym, _format_complex(val, mode=value_mode), fontsize=8, ha="center", va="center")
    plt.show()


def plot_pumps_vs_frequency(
    pump_freqs_THz: Dict[int, float],
    pump_rel_power: Dict[int, float],
    mode_freqs_THz: Optional[Dict[int, float]] = None,
    *,
    global_scale: float = 1.0,
    annotate: bool = True,
    y_unit: str = "rel. power",
    title: Optional[str] = None,
    yscale: str = "linear",
) -> None:
    """
    Plot pump amplitudes vs frequency, with optional mode frequencies.
    - x-axis: frequency in THz
    - y-axis: amplitude = relative power (optionally scaled by global_scale^2)
    - Pumps are drawn as stems; modes appear at y=0 (no power).

    Parameters
    ----------
    pump_freqs_THz : dict {pump_label -> frequency_THz}
    pump_rel_power : dict {pump_label -> relative_power}  (dimensionless)
    mode_freqs_THz : dict {mode_index -> frequency_THz} or None
    global_scale   : float  (if you used a global amplitude scale 's', pass it here;
                             plotted power becomes s^2 * relative_power)
    annotate       : bool   (write pump labels and power values near stems)
    y_unit         : str    (axis label, default "rel. power")
    title          : str or None
    yscale         : str    ("linear" or "log")
    """
    if not pump_freqs_THz or not pump_rel_power:
        print("Nothing to plot: empty pump set.")
        return

    # Build sorted arrays for pumps
    pump_items = sorted(pump_freqs_THz.items(), key=lambda kv: kv[1])  # sort by frequency
    p_labels = [k for (k, _) in pump_items]
    p_freqs  = np.array([v for (_, v) in pump_items], dtype=float)
    # Power after optional global scaling
    p_pows   = np.array([pump_rel_power.get(k, 0.0) for k in p_labels], dtype=float) * (global_scale ** 2)

    # Mode frequencies (optional)
    if mode_freqs_THz:
        m_items = sorted(mode_freqs_THz.items(), key=lambda kv: kv[1])
        m_labels = [k for (k, _) in m_items]
        m_freqs  = np.array([v for (_, v) in m_items], dtype=float)
    else:
        m_labels, m_freqs = [], np.array([], dtype=float)

    # Figure
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.set_title(title or "Pump spectrum (frequency vs amplitude)")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel(f"Amplitude ({y_unit})")
    if yscale in ("linear", "log"):
        ax.set_yscale(yscale)

    # Plot mode frequencies at y=0 (no amplitude)
    if m_freqs.size > 0:
        ax.scatter(m_freqs, np.zeros_like(m_freqs), marker="x", s=36, label="Modes (0 power)")
        # Light vertical guides for modes
        for f in m_freqs:
            ax.axvline(f, ymin=0.0, ymax=0.08, ls="--", alpha=0.25, lw=1)

    # Plot pump stems
    # vertical lines from 0 to power, and points at the top
    ax.vlines(p_freqs, 0.0, p_pows, lw=2, label="Pumps")
    ax.scatter(p_freqs, p_pows, s=36)

    # Optional annotations: pump label and numeric power
    if annotate:
        for x, y, lbl in zip(p_freqs, p_pows, p_labels):
            ax.annotate(f"p{lbl}\n{y:.3g}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 6),
                        ha="center", va="bottom", fontsize=8)

    # Cosmetics
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=False)
    # A bit of headroom on y for labels
    ymax = p_pows.max() if p_pows.size else 1.0
    if yscale == "linear":
        ax.set_ylim(bottom=0.0, top=max(1e-3, ymax * 1.15))
    plt.tight_layout()
    plt.show()


def plot_pumps_vs_frequency_from_phys(
    phys: Dict[str, Dict],
    *,
    global_scale: Optional[float] = None,
    annotate: bool = True,
    y_unit: str = "rel. power",
    title: Optional[str] = None,
    yscale: str = "linear",
) -> None:
    """
    Convenience wrapper for the dict returned by plan_to_physical().
    Expects keys:
      - phys["pump_freqs_THz"] : {pump_label -> f_THz}
      - phys["pump_rel_power"] : {pump_label -> relative_power}
      - phys["mode_freqs_THz"] : {mode_index -> f_THz}
      - (optional) phys["global_amp_scale"] : s

    Parameters mirror plot_pumps_vs_frequency() with global_scale defaulted to
    phys.get("global_amp_scale", 1.0) if not passed explicitly.
    """
    pump_f = phys.get("pump_freqs_THz", {})
    pump_P = phys.get("pump_rel_power", {})
    mode_f = phys.get("mode_freqs_THz", {})
    if global_scale is None:
        global_scale = float(phys.get("global_amp_scale", 1.0))

    return plot_pumps_vs_frequency(
        pump_f, pump_P, mode_f,
        global_scale=global_scale,
        annotate=annotate,
        y_unit=y_unit,
        title=title,
        yscale=yscale,
    )