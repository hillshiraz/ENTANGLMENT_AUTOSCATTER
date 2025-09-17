# microcomb_planner
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import itertools
import numpy as np

try:
    from inverse_synthesis import PumpEdge
except Exception:
    from dataclasses import dataclass as _dc
    @_dc
    class PumpEdge:
        i: int; j: int; type: str; amp: float; phase: float

@dataclass
class CombSpec:
    f0_THz: float                 # e.g. 193.518
    FSR_GHz: float                # e.g. 199.9
    pump_lines: Optional[List[int]] = None  # None => auto-select
    mode_lines: Optional[List[int]] = None  # map your N modes to comb indices (if None: auto map near 0)

@dataclass
class Plan:
    comb: CombSpec
    mode_to_line: Dict[int, int]                  # local mode index -> comb line m
    edge_realization: Dict[Tuple[int,int,str], Tuple[int,...]]  # (i,j,type)->pump idx tuple
    pump_A_rel: Dict[int, complex]                # complex pump amplitudes up to a global scale (max|A|=1)

# ---------- helpers ----------

def _edges_from_solution(pumps: Iterable[PumpEdge]) -> List[Tuple[int,int,str,float,float]]:
    out = []
    for e in pumps:
        t = e.type.upper()
        if t not in ("BS","SPS"):
            continue
        i, j = (e.i, e.j) if e.i <= e.j else (e.j, e.i)
        out.append((i, j, t, float(e.amp), float(e.phase)))
    # unique by (i,j,type) keep largest amplitude if duplicates
    uniq: Dict[Tuple[int,int,str], Tuple[float,float]] = {}
    for i,j,t,a,ph in out:
        k = (i,j,t)
        if k not in uniq or abs(a) > abs(uniq[k][0]):
            uniq[k] = (a, ph)
    return [(i,j,t, uniq[(i,j,t)][0], uniq[(i,j,t)][1]) for (i,j,t) in uniq]

def _default_mode_map(num_modes: int, span: int = 2) -> List[int]:
    """Pack N modes near comb center: [0,-1,+1,-2,+2,...]."""
    if num_modes == 1: return [0]
    if num_modes == 2: return [-1,+1]
    m = []
    k = 0
    while len(m) < num_modes and k <= span:
        if k == 0:
            m.append(0)
        else:
            if len(m) < num_modes: m.append(-k)
            if len(m) < num_modes: m.append(+k)
        k += 1
    return m[:num_modes]

def _cover_edge_with_pumps(i_line: int, j_line: int, edge_type: str, pump_lines: List[int]) -> Optional[Tuple[int,...]]:
    """Return which pump(s) implement a given BS/SPS edge, or None if impossible."""
    di = j_line - i_line
    if edge_type.upper() == "BS":
        # need p,q with q - p = di
        for p in pump_lines:
            q = p + di
            if q in pump_lines:
                return (p, q)
        return None
    # SPS
    s = i_line + j_line
    # try degenerate first
    if s % 2 == 0:
        p = s // 2
        if p in pump_lines:
            return (p,)
    # try non-degenerate: p+q = s
    for p in pump_lines:
        q = s - p
        if q in pump_lines:
            return (p, q)
    return None

def _auto_select_pumps_for_edges(
    mode_lines: List[int],
    edges: List[Tuple[int,int,str,float,float]],
    max_pumps: int = 3,
    max_span: int = 6,
    must_include_zero: bool = True,
    forbidden_lines: Optional[Iterable[int]] = None,   # NEW
) -> List[int]:

    # candidate universe of lines
    universe = list(range(-max_span, max_span+1))
    if must_include_zero and 0 not in universe:
        universe.append(0)
    universe = sorted(universe)

    #remove forbidden
    forb = set(forbidden_lines or [])
    universe = [m for m in universe if m not in forb]

    # progressively grow cardinality
    for k in range(1, max_pumps+1):
        if must_include_zero and 0 not in forb and 0 in universe:
            pool = [m for m in universe if m != 0]
            for subset in itertools.combinations(pool, k-1):
                cand = tuple(sorted((0,)+subset))
                if _covers_all_edges(cand, mode_lines, edges):
                    return list(cand)
        else:
            for cand in itertools.combinations(universe, k):
                if _covers_all_edges(cand, mode_lines, edges):
                    return list(cand)
    return []

def _covers_all_edges(pump_lines: Iterable[int], mode_lines: List[int], edges) -> bool:
    pump_lines = list(pump_lines)
    for (i,j,t,_,_) in edges:
        mi, mj = mode_lines[i], mode_lines[j]
        if _cover_edge_with_pumps(mi, mj, t, pump_lines) is None:
            return False
    return True

# ---------- main planning ----------
def propose_plan(
    fitted_edges: Iterable[PumpEdge],
    comb: CombSpec,
    normalize_max_A: bool = True,
    auto_max_pumps: int = 3,
    auto_max_span: int = 6,
    allowed_edge_types: Iterable[str] = ("BS","SPS"),   # NEW
    min_edge_amp: float = 0.0,                          # NEW
    forbid_pump_on_mode_lines: bool = False,            # NEW
) -> Plan:
    edges_all = _edges_from_solution(fitted_edges)

    # NEW: filter by type & amplitude threshold
    allowed = {t.upper() for t in allowed_edge_types}
    edges = [(i,j,t,a,ph) for (i,j,t,a,ph) in edges_all
             if t in allowed and abs(a) >= float(min_edge_amp)]
    if not edges:
        raise ValueError("No edges left after filtering (check allowed_edge_types / min_edge_amp).")

    num_modes = max(max(i,j) for (i,j,_,_,_) in edges) + 1
    mode_lines = comb.mode_lines if comb.mode_lines is not None else _default_mode_map(num_modes)

    # choose pumps
    if comb.pump_lines is None:
        forbidden = set(mode_lines) if forbid_pump_on_mode_lines else set()
        pump_lines = _auto_select_pumps_for_edges(
            mode_lines, edges,
            max_pumps=auto_max_pumps,
            max_span=auto_max_span,
            must_include_zero=True,
            forbidden_lines=forbidden,          # NEW
        )
        if not pump_lines:
            raise RuntimeError(f"Could not realize all edges with ≤{auto_max_pumps} pumps and span ≤{auto_max_span}. "
                               f"Try increasing auto_max_pumps or auto_max_span, or adjust mode_lines.")
    else:
        pump_lines = sorted([m for m in comb.pump_lines
                             if (not forbid_pump_on_mode_lines) or (m not in set(mode_lines))])

    # assign pumps to each edge
    mode_to_line = {i: mode_lines[i] for i in range(num_modes)}
    edge_realization: Dict[Tuple[int,int,str], Tuple[int,...]] = {}
    for (i,j,t,a,ph) in edges:
        mi, mj = mode_to_line[i], mode_to_line[j]
        tup = _cover_edge_with_pumps(mi, mj, t, pump_lines)
        if tup is None:
            raise RuntimeError(f"Edge {(i,j,t)} cannot be realized by pump set {pump_lines} and mode_lines {mode_lines}")
        edge_realization[(i,j,t)] = tup

    # solve for pump phases/mags up to global scale using least-squares
    P = len(pump_lines)
    idx = {p:k for k,p in enumerate(pump_lines)}
    # Phase equations: Mφ = y; Magnitude (log): Nλ = z  with λ_k = log|A_k|
    Mφ, y = [], []
    Nλ, z = [], []

    for (i,j,t,amp,ph) in edges:
        ps = edge_realization[(i,j,t)]
        if t == "BS":
            p,q = ps
            row = np.zeros(P); row[idx[p]] = 1.0; row[idx[q]] = -1.0
            Mφ.append(row); y.append(ph)
            rowm = np.zeros(P); rowm[idx[p]] = 1.0; rowm[idx[q]] = 1.0
            Nλ.append(rowm); z.append(np.log(max(amp,1e-16)))
        else:
            if len(ps) == 1:
                p = ps[0]
                row = np.zeros(P); row[idx[p]] = 2.0
                Mφ.append(row); y.append(ph)
                rowm = np.zeros(P); rowm[idx[p]] = 2.0
                Nλ.append(rowm); z.append(np.log(max(amp,1e-16)))
            else:
                p,q = ps
                row = np.zeros(P); row[idx[p]] = 1.0; row[idx[q]] = 1.0
                Mφ.append(row); y.append(ph)
                rowm = np.zeros(P); rowm[idx[p]] = 1.0; rowm[idx[q]] = 1.0
                Nλ.append(rowm); z.append(np.log(max(amp,1e-16)))

    Mφ = np.array(Mφ) if Mφ else np.zeros((0,P))
    y  = np.array(y)  if y  else np.zeros((0,))
    Nλ = np.array(Nλ) if Nλ else np.zeros((0,P))
    z  = np.array(z)  if z  else np.zeros((0,))

    # fix phase gauge: φ_0pump = 0
    if P > 0:
        prior = np.zeros(P); prior[0] = 1.0
        Mφ = np.vstack([Mφ, prior]); y = np.hstack([y, 0.0])

    φ_sol, *_  = np.linalg.lstsq(Mφ, y, rcond=None)
    λ_sol, *_  = np.linalg.lstsq(Nλ, z, rcond=None)

    A = np.exp(λ_sol) * np.exp(1j*φ_sol)
    if normalize_max_A and np.max(np.abs(A)) > 0:
        A = A / np.max(np.abs(A))

    pump_A_rel = {p: A[idx[p]] for p in pump_lines}
    return Plan(CombSpec(comb.f0_THz, comb.FSR_GHz, pump_lines, mode_lines),
                mode_to_line, edge_realization, pump_A_rel)

def plan_to_physical(plan: Plan) -> Dict[str, object]:
    """Turn plan into THz frequencies and relative powers (max |A| = 1 ⇒ powers are relative)."""
    f0 = plan.comb.f0_THz
    FSR_THz = plan.comb.FSR_GHz / 1000.0
    mode_freqs_THz = {i: f0 + plan.mode_to_line[i]*FSR_THz for i in plan.mode_to_line}
    pump_freqs_THz = {p: f0 + p*FSR_THz for p in plan.pump_A_rel}
    pump_phase_rad = {p: float(np.angle(A)) for p, A in plan.pump_A_rel.items()}
    pump_rel_power = {p: float(np.abs(A)**2) for p, A in plan.pump_A_rel.items()}
    return dict(
        mode_freqs_THz=mode_freqs_THz,
        pump_freqs_THz=pump_freqs_THz,
        pump_phase_rad=pump_phase_rad,
        pump_rel_power=pump_rel_power,
        edge_realization=plan.edge_realization,
        note="Relative powers assume max|A|=1. Scale all together later after calibration."
    )

def compute_required_global_scale_and_powers(plan: Plan, fitted_edges: Iterable[PumpEdge]) -> Tuple[float, Dict[int, float]]:
    """
    Given the fitted couplings, compute the *minimum* global amplitude scale s
    such that s^2 * (pump products from plan) match all |g|,|nu|.
    Returns:
      s (amplitude scale), scaled relative powers per pump = (s^2)*|A_p|^2.
    Use this to check feasibility vs power caps once you know the calibration.
    """
    edges = _edges_from_solution(fitted_edges)
    A = plan.pump_A_rel
    # map pump index to |A|
    absA = {p: np.abs(A[p]) for p in A}
    # find s needed per edge and take the max (most demanding)
    s_needed = 0.0
    for (i,j,t,amp,_) in edges:
        ps = plan.edge_realization[(i,j,t)]
        if t == "BS":
            p,q = ps
            base = absA[p]*absA[q]
        else:
            if len(ps) == 1:
                p = ps[0]
                base = absA[p]**2
            else:
                p,q = ps
                base = absA[p]*absA[q]
        base = max(base, 1e-16)
        s_needed = max(s_needed, np.sqrt(amp / base))
    scaled_powers = {p: (s_needed**2) * (absA[p]**2) for p in absA}
    return s_needed, scaled_powers
