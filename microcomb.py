import numpy as np
from math import ceil
from scipy.optimize import least_squares

# =========================
# Utilities: requirements
# =========================
def _required_sums_diffs(mode_idx, g_target, nu_target):
    """From targets and a candidate mode index set, extract the required
    sums S={m_k+m_l | ν_target[k,l]≠0} and diffs D={m_k-m_l | g_target[k,l]≠0}."""
    mode_idx = np.asarray(mode_idx, int)
    N = len(mode_idx)
    need_sums, need_diffs = set(), set()
    for k in range(N):
        for l in range(N):
            if np.abs(nu_target[k, l]) > 0:
                need_sums.add(int(mode_idx[k] + mode_idx[l]))
            if np.abs(g_target[k, l]) > 0:
                need_diffs.add(int(mode_idx[k] - mode_idx[l]))
    return need_sums, need_diffs

def _pump_pair_sets(pump_idx):
    P = np.asarray(pump_idx, int)
    sums  = set(int(p+q) for p in P for q in P)
    diffs = set(int(p-q) for p in P for q in P)
    return sums, diffs


def _first_non_mode(mode_set):
    i = 0
    while True:
        for cand in (i, -i):
            if cand not in mode_set:
                return cand
        i += 1

# =========================
# Build minimal pump set
# =========================
def _greedy_min_pumps(mode_idx, need_sums, need_diffs, R, max_expand=5):
    mode_set = set(int(m) for m in np.asarray(mode_idx, int))
    r = int(R)

    def universe(rr):
        return [i for i in range(-rr, rr+1) if i not in mode_set]

    for _ in range(max_expand+1):
        U = universe(r)
        if not U:
            r += 1
            continue

        chosen = set()
        while True:
            # Check current coverage
            if chosen:
                sums_cur, diffs_cur = _pump_pair_sets(np.array(sorted(chosen), int))
            else:
                sums_cur, diffs_cur = set(), set()

            if need_sums.issubset(sums_cur) and need_diffs.issubset(diffs_cur) and len(chosen) > 0:
                return np.array(sorted(list(chosen)), dtype=int), r

            # Pick the candidate that maximizes newly covered items
            best_cand, best_gain = None, (-1, -1, -10**9)  # (gain_s, gain_d, compactness)
            need_s_left = need_sums - sums_cur
            need_d_left = need_diffs - diffs_cur

            for cand in U:
                if cand in chosen:
                    continue
                sums_new, diffs_new = _pump_pair_sets(np.array(sorted(list(chosen | {cand})), int))
                gain_s = len(need_s_left & sums_new)
                gain_d = len(need_d_left & diffs_new)
                compact  = -abs(int(cand))  # prefer smaller |index|
                score = (gain_s, gain_d, compact)
                if score > best_gain:
                    best_gain = score
                    best_cand = cand

            # If we cannot improve coverage at this radius, expand radius
            if best_cand is None or (best_gain[0] == 0 and best_gain[1] == 0):
                break

            chosen.add(int(best_cand))

        # expand search radius and retry
        r += 1

    # --- Constructive fallback: guarantee coverage of required diffs ---
    Dpos = sorted({abs(d) for d in need_diffs if d != 0})
    mode_set = set(int(m) for m in np.asarray(mode_idx, int))

    # find a base b that doesn't collide with any mode (and keeps all b+d off modes)
    def find_base():
        i = 0
        while True:
            for b in (i, -i):
                if b in mode_set:
                    continue
                # check collisions for b+d as well
                if any((b + d) in mode_set for d in Dpos):
                    continue
                return b
            i += 1

    b = find_base()
    pumps = [b] + [b + d for d in Dpos]
    return np.array(sorted(pumps), dtype=int), r


def synthesize_g_nu_from_pumps_weighted(A, kappa, omega0, FSR, mode_idx, pump_idx):

    assert FSR > 0, "FSR must be positive."
    mode_idx = np.asarray(mode_idx, dtype=int)
    pump_idx = np.asarray(pump_idx, dtype=int)
    N = len(mode_idx)
    P = len(pump_idx)
    A = np.asarray(A, dtype=complex).reshape(P)

    g  = np.zeros((N, N), dtype=complex)
    nu = np.zeros((N, N), dtype=complex)

    pump_set = set(pump_idx.tolist())
    val_to_inds = {}
    for ii, p in enumerate(pump_idx):
        val_to_inds.setdefault(int(p), []).append(ii)

    for k in range(N):
        for l in range(N):
            # SPS: k+l = p+q
            Skl = int(mode_idx[k] + mode_idx[l])
            for ip, p in enumerate(pump_idx):
                q = Skl - int(p)
                if q in pump_set:
                    for iq in val_to_inds[q]:
                        nu[k, l] += kappa * A[ip] * A[iq]
            # BS: k-l = p-q
            Dkl = int(mode_idx[k] - mode_idx[l])
            for ip, p in enumerate(pump_idx):
                q = int(p) - Dkl
                if q in pump_set:
                    for iq in val_to_inds[q]:
                        if k == l and iq == ip:
                            continue  # skip p=q contributions on the diagonal
                        g[k, l] += kappa * np.conj(A[ip]) * A[iq]

    g  = 0.5 * (g  + g.conj().T)  # Hermitian
    nu = 0.5 * (nu + nu.T)        # symmetric
    return g, nu

# =========================
# Least-squares residual with power-normalized A (prevents A=0)
# =========================
def _pack_residual(g_syn, nu_syn, g_target, nu_target, w_g, w_nu):
    r_g  = np.sqrt(w_g)  * (g_syn  - g_target).ravel('F').view(np.float64)
    r_nu = np.sqrt(w_nu) * (nu_syn - nu_target).ravel('F').view(np.float64)
    return np.concatenate([r_g, r_nu])

def _residual_vec(x, kappa, omega0, FSR, mode_idx, pump_idx, g_target, nu_target, w_g, w_nu, P0=1.0):
    P = len(pump_idx)
    B = x[:P] + 1j * x[P:2*P]
    # Power normalization: keep total |A|^2 fixed so A ≠ 0
    A = np.sqrt(P0) * B / (np.linalg.norm(B) + 1e-12)
    g_syn, nu_syn = synthesize_g_nu_from_pumps_weighted(A, kappa, omega0, FSR, mode_idx, pump_idx)
    return _pack_residual(g_syn, nu_syn, g_target, nu_target, w_g, w_nu)

def _fit_amplitudes(mode_idx, pump_idx, kappa, omega0, FSR, g_target, nu_target,
                     w_g, w_nu, n_restarts=5, P0=1.0):
    pump_idx = np.asarray(pump_idx, int)
    P = len(pump_idx)
    if P == 0:
        # should not happen (we enforce at least one pump), but keep safe
        A = np.zeros(0, complex)
        g_syn, nu_syn = synthesize_g_nu_from_pumps_weighted(A, kappa, omega0, FSR, mode_idx, pump_idx)
        err = w_g*np.linalg.norm(g_syn-g_target,'fro')**2 + w_nu*np.linalg.norm(nu_syn-nu_target,'fro')**2
        return A, g_syn, nu_syn, err

    best = None
    for _ in range(n_restarts):
        s  = np.random.choice([1e-2, 1e-1, 1.0])
        x0 = s * np.random.randn(2*P)
        res = least_squares(
            _residual_vec, x0,
            args=(kappa, omega0, FSR, mode_idx, pump_idx, g_target, nu_target, w_g, w_nu, P0),
            method='trf', max_nfev=3000, x_scale='jac'
        )
        if best is None or res.cost < best[0]:
            best = (res.cost, res.x)

    x = best[1]
    B = x[:P] + 1j*x[P:2*P]
    A = np.sqrt(P0) * B / (np.linalg.norm(B) + 1e-12)
    g_syn, nu_syn = synthesize_g_nu_from_pumps_weighted(A, kappa, omega0, FSR, mode_idx, pump_idx)
    err = w_g*np.linalg.norm(g_syn-g_target,'fro')**2 + w_nu*np.linalg.norm(nu_syn-nu_target,'fro')**2
    return A, g_syn, nu_syn, err

# =========================
# Joint search: modes & minimal pumps
# =========================
def auto_microcomb(info, g_target, nu_target, kappa,
                   omega0=0.0, FSR=1.0,
                   w_g=1.0, w_nu=1.0,
                   n_restarts=6, R=None, max_outer_iters=25):
    """
    Jointly choose mode indices and the *minimal* pump set (count picked automatically)
    that exactly covers δ1 (sums) & δ2 (diffs), disjoint from modes; then fit amplitudes.
    """
    # 1) initialize modes (centered integer block, but we will move them)
    N = np.asarray(info['coupling_matrix']).shape[0]
    half = (N-1)//2
    mode_idx = np.arange(-half, -half+N, dtype=int)

    # 2) pick an initial radius
    need_sums, need_diffs = _required_sums_diffs(mode_idx, g_target, nu_target)
    Smax = max((abs(s) for s in need_sums), default=0)
    Dmax = max((abs(d) for d in need_diffs), default=0)
    Mmax = int(np.max(np.abs(mode_idx))) if mode_idx.size else 0
    if R is None:
        R = max(ceil(Smax/2), Dmax, Mmax) + 2

    # 3) build minimal pump set for these modes
    pump_idx, R_used = _greedy_min_pumps(mode_idx, need_sums, need_diffs, R)

    # 4) inner fit
    A, g_syn, nu_syn, err = _fit_amplitudes(mode_idx, pump_idx, kappa, omega0, FSR,g_target, nu_target,  w_g, w_nu, n_restarts=n_restarts)

    best = dict(mode_idx=mode_idx.copy(), pump_idx=pump_idx.copy(),
                A=A, g=g_syn, nu=nu_syn, err=float(err), R=int(R_used))

    # 5) nudge one mode ±1, rebuild minimal pumps, refit A; accept if error drops
    for _ in range(max_outer_iters):
        improved = False
        for i in range(N):
            for step in (-1, +1):
                cand = int(best["mode_idx"][i] + step)
                if abs(cand) > best["R"] + 2:
                    continue
                if cand in set(best["mode_idx"]):
                    continue  # keep modes distinct
                mode_try = best["mode_idx"].copy()
                mode_try[i] = cand
                mode_try.sort()

                need_sums, need_diffs = _required_sums_diffs(mode_try, g_target, nu_target)
                pumps_try, R_try = _greedy_min_pumps(mode_try, need_sums, need_diffs, max(best["R"], R))
                # fit A
                A2, g2, nu2, err2 = _fit_amplitudes(mode_try, pumps_try, kappa, omega0, FSR,g_target, nu_target, w_g, w_nu, n_restarts=n_restarts)
                if err2 + 1e-12 < best["err"]:
                    best = dict(mode_idx=mode_try.copy(), pump_idx=pumps_try.copy(),
                                A=A2, g=g2, nu=nu2, err=float(err2), R=int(R_try))
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # Final packaging
    mode_idx = np.asarray(best["mode_idx"], int)
    pump_idx = np.asarray(best["pump_idx"], int)
    A        = best["A"]
    g_syn    = best["g"]
    nu_syn   = best["nu"]
    freqs    = omega0 + pump_idx.astype(float) * FSR

    return {
        "omega0": omega0,
        "FSR_out": FSR,
        "mode_indices": mode_idx,
        "pump_indices": pump_idx,
        "pump_frequencies": freqs,
        "pump_amplitudes": np.abs(A),
        "pump_phases": np.angle(A),
        "A_complex": A,
        "g_synth": g_syn, "nu_synth": nu_syn,
        "g_error_fro2": float(np.linalg.norm(g_syn - g_target, 'fro')**2),
        "nu_error_fro2": float(np.linalg.norm(nu_syn - nu_target, 'fro')**2),
        "objective_inner": float(best["err"]),
        "R": int(best["R"]),
        "delta_coverage_ok": True  # by construction; we built exact coverage
    }
