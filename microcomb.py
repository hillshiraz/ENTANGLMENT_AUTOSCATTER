import numpy as np
from math import ceil
from scipy.optimize import least_squares
import random
from deap import base, creator, tools, algorithms


# =========================
#functiouns I need for the optimization
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
    sums = set(int(p + q) for p in P for q in P)
    diffs = set(int(p - q) for p in P for q in P)
    return sums, diffs


def synthesize_g_nu_from_pumps_weighted(A, kappa, omega0, FSR, mode_idx, pump_idx):
    """Synthesize g and nu matrices from given pumps and amplitudes."""
    assert FSR > 0, "FSR must be positive."
    mode_idx = np.asarray(mode_idx, dtype=int)
    pump_idx = np.asarray(pump_idx, dtype=int)
    N = len(mode_idx)
    P = len(pump_idx)
    A = np.asarray(A, dtype=complex).reshape(P)

    g = np.zeros((N, N), dtype=complex)
    nu = np.zeros((N, N), dtype=complex)

    pump_set = set(pump_idx.tolist())
    val_to_inds = {}
    for ii, p in enumerate(pump_idx):
        val_to_inds.setdefault(int(p), []).append(ii)

    for k in range(N):
        for l in range(N):
            Skl = int(mode_idx[k] + mode_idx[l])
            for ip, p in enumerate(pump_idx):
                q = Skl - int(p)
                if q in pump_set:
                    for iq in val_to_inds[q]:
                        nu[k, l] += kappa * A[ip] * A[iq]

            Dkl = int(mode_idx[k] - mode_idx[l])
            for ip, p in enumerate(pump_idx):
                q = int(p) - Dkl
                if q in pump_set:
                    for iq in val_to_inds[q]:
                        if k == l and iq == ip:
                            continue
                        g[k, l] += kappa * np.conj(A[ip]) * A[iq]

    g = 0.5 * (g + g.conj().T)
    nu = 0.5 * (nu + nu.T)
    return g, nu


def _pack_residual(g_syn, nu_syn, g_target, nu_target, w_g, w_nu):
    """Pack the residual vector for least_squares."""
    r_g = np.sqrt(w_g) * (g_syn - g_target).ravel('F').view(np.float64)
    r_nu = np.sqrt(w_nu) * (nu_syn - nu_target).ravel('F').view(np.float64)
    return np.concatenate([r_g, r_nu])


def _residual_vec(x, kappa, omega0, FSR, mode_idx, pump_idx, g_target, nu_target, w_g, w_nu, P0=1.0):
    """Residual function for least_squares."""
    P = len(pump_idx)
    A = x[:P] + 1j * x[P:2 * P]
    g_syn, nu_syn = synthesize_g_nu_from_pumps_weighted(A, kappa, omega0, FSR, mode_idx, pump_idx)
    return _pack_residual(g_syn, nu_syn, g_target, nu_target, w_g, w_nu)


def _fit_amplitudes(mode_idx, pump_idx, kappa, omega0, FSR, g_target, nu_target, w_g, w_nu, n_restarts=5, P0=1.0):
    pump_idx = np.asarray(pump_idx, int)
    P = len(pump_idx)
    if P == 0:
        A = np.zeros(0, complex)
        g_syn, nu_syn = synthesize_g_nu_from_pumps_weighted(A, kappa, omega0, FSR, mode_idx, pump_idx)
        err = w_g * np.linalg.norm(g_syn - g_target, 'fro') ** 2 + w_nu * np.linalg.norm(nu_syn - nu_target, 'fro') ** 2
        return A, g_syn, nu_syn, err

    best = None
    for _ in range(n_restarts):
        s = np.random.choice([1e-2, 1e-1, 1.0])
        x0 = s * np.random.randn(2 * P)
        res = least_squares(
            _residual_vec, x0,
            args=(kappa, omega0, FSR, mode_idx, pump_idx, g_target, nu_target, w_g, w_nu, P0),
            method='trf', max_nfev=3000, x_scale='jac'
        )
        if best is None or res.cost < best[0]:
            best = (res.cost, res.x)

    x = best[1]
    A = x[:P] + 1j * x[P:2 * P]
    g_syn, nu_syn = synthesize_g_nu_from_pumps_weighted(A, kappa, omega0, FSR, mode_idx, pump_idx)
    err = w_g * np.linalg.norm(g_syn - g_target, 'fro') ** 2 + w_nu * np.linalg.norm(nu_syn - nu_target, 'fro') ** 2
    return A, g_syn, nu_syn, err


def _fit_amplitudes_and_get_cost(mode_idx, pump_idx, kappa, omega0, FSR, g_target, nu_target, w_g, w_nu, n_restarts=3,
                                 P0=1.0):
    pump_idx = np.asarray(pump_idx, int)
    P = len(pump_idx)
    if P == 0:
        return float('inf')

    A, g_syn, nu_syn, best_cost = _fit_amplitudes(
        mode_idx, pump_idx, kappa, omega0, FSR, g_target, nu_target, w_g, w_nu, n_restarts, P0
    )
    return best_cost

def _split_side_counts(mode_idx, pump_idx):
    """Count pumps left/right/inside the mode span."""
    if len(mode_idx) == 0 or len(pump_idx) == 0:
        return 0, 0, 0
    mn, mx = int(np.min(mode_idx)), int(np.max(mode_idx))
    left  = int(np.sum(pump_idx < mn))
    right = int(np.sum(pump_idx > mx))
    inside = int(np.sum((pump_idx >= mn) & (pump_idx <= mx)))
    return left, right, inside


def fitness_function(individual, info, g_target, nu_target, kappa, omega0, FSR, w_g, w_nu,
                     penalty_per_pump=0.5,
                     separation=False,
                     separation_mode="enforce",   # "enforce" or "prefer"
                     balance_penalty=0.0,         # only for prefer; penalize |left-right|
                     inside_penalty=1e3,          # only for prefer; per pump inside span
                     enforce_even_balance=False):  # when even #pumps, force left==right
    """
    Genetic Algorithm Fitness Function.
    'individual' is a list of integers representing pump and mode indices.
    """
    N = np.asarray(info['coupling_matrix']).shape[0]
    num_pumps = int(individual[0])

    if num_pumps <= 0 or num_pumps > 15:
        return float('inf'),

    pump_idx_list = individual[1:1 + num_pumps]
    mode_idx_list = individual[1 + num_pumps:]

    mode_idx = np.array(sorted(list(set(mode_idx_list))), dtype=int)
    if len(mode_idx) != N:
        return float('inf'),

    pump_idx = np.array(sorted(list(set(pump_idx_list))), dtype=int)

    if set(mode_idx).intersection(set(pump_idx)):
        return float('inf'),

    # --- separation constraint/penalty ---
    sep_cost = 0.0  # <—— initialize!
    if separation:
        left, right, inside = _split_side_counts(mode_idx, pump_idx)
        if separation_mode == "enforce":
            # hard rule: all pumps outside and at least one on each side
            if inside > 0 or left == 0 or right == 0:
                return float('inf'),
            # optional even balance rule (only when number of pumps is even)
            if enforce_even_balance and (num_pumps % 2 == 0) and (left != right):
                return float('inf'),
        else:
            # prefer: add penalties instead of discarding
            sep_cost += inside_penalty * inside
            sep_cost += balance_penalty * abs(left - right)

    # sums/diffs coverage
    need_sums, need_diffs = _required_sums_diffs(mode_idx, g_target, nu_target)
    pump_sums, pump_diffs = _pump_pair_sets(pump_idx)

    if (len(need_sums) > 0 and not need_sums.issubset(pump_sums)) or \
       (len(need_diffs) > 0 and not need_diffs.issubset(pump_diffs)):
        return float('inf'),

    # inner fit
    cost = _fit_amplitudes_and_get_cost(mode_idx, pump_idx, kappa, omega0, FSR, g_target, nu_target, w_g, w_nu)
    if np.isinf(cost):
        return (1000000.0,)

    # complexity + separation penalties
    cost += penalty_per_pump * num_pumps
    cost += sep_cost
    return cost,

def auto_microcomb(info, g_target, nu_target, kappa, omega0=0.0, FSR=1.0, w_g=1.0, w_nu=1.0,
            pop_size=100, num_gens=30, cx_prob=0.8, mut_prob=0.2,
            separation=False,  # enforce/prefer pumps outside the mode span
            separation_mode="enforce",  # "enforce" or "prefer"
            penalty_per_pump=0.5,  # complexity penalty
            inside_penalty=1e3,  # used when separation_mode="prefer"
            balance_penalty=0.0,  # used when separation_mode="prefer"
            enforce_even_balance=False,  # if True and num_pumps even -> left==right
            bias_initializer=True  # bias initial individuals to satisfy separation
    ):

    try:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
    except:
        pass

    N = np.asarray(info['coupling_matrix']).shape[0]

    toolbox = base.Toolbox()

    MAX_PUMPS = 6
    MODE_RANGE = 20
    PUMP_RANGE = 30

    def get_individual():
        num_pumps = random.randint(1, MAX_PUMPS)
        pump_indices = [random.randint(-PUMP_RANGE, PUMP_RANGE) for _ in range(num_pumps)]
        mode_indices = [random.randint(-MODE_RANGE, MODE_RANGE) for _ in range(N)]
        return creator.Individual([num_pumps] + pump_indices + mode_indices)

    toolbox.register("individual", get_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        fitness_function,
        info=info, g_target=g_target, nu_target=nu_target, kappa=kappa,
        omega0=omega0, FSR=FSR, w_g=w_g, w_nu=w_nu,
        penalty_per_pump=penalty_per_pump,
        separation=separation,
        separation_mode=separation_mode,
        balance_penalty=balance_penalty,
        inside_penalty=inside_penalty,
        enforce_even_balance=enforce_even_balance
    )

    toolbox.register("mate", tools.cxTwoPoint)

    def mutate_individual(individual, indpb):
        # The number of modes (N) is a constant. We can access it from the outer scope.
        # This will fix the NameError.
        N_modes = len(individual) - (individual[0] + 1)

        # Mutate the number of pumps
        if random.random() < indpb:
            individual[0] = random.randint(1, MAX_PUMPS)
            new_len_pumps = individual[0]
            current_len_pumps = len(individual) - 1 - N_modes
            if new_len_pumps > current_len_pumps:
                individual.extend([random.randint(-PUMP_RANGE, PUMP_RANGE)] * (new_len_pumps - current_len_pumps))
            else:
                del individual[1 + new_len_pumps:1 + current_len_pumps]

        # Mutate the pump and mode indices
        num_pumps_final = individual[0]
        tools.mutUniformInt(individual[1:1 + num_pumps_final], low=-PUMP_RANGE, up=PUMP_RANGE, indpb=indpb)
        tools.mutUniformInt(individual[1 + num_pumps_final:], low=-MODE_RANGE, up=MODE_RANGE, indpb=indpb)

        return individual,

    toolbox.register("mutate", mutate_individual, indpb=mut_prob)

    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cx_prob, mut_prob, num_gens, stats=stats, halloffame=hof,
                                       verbose=True)

    if not hof:
        return "No valid solution found."

    best_individual = hof[0]
    num_pumps = int(best_individual[0])
    pump_idx = np.array(list(set(best_individual[1:1 + num_pumps])), dtype=int)
    mode_idx = np.array(sorted(list(set(best_individual[1 + num_pumps:]))), dtype=int)

    A, g_syn, nu_syn, err = _fit_amplitudes(mode_idx, pump_idx, kappa, omega0, FSR, g_target, nu_target, w_g, w_nu)

    freqs = omega0 + pump_idx.astype(float) * FSR
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
        "g_error_fro2": float(np.linalg.norm(g_syn - g_target, 'fro') ** 2),
        "nu_error_fro2": float(np.linalg.norm(nu_syn - nu_target, 'fro') ** 2),
        "objective_inner": float(err),
        "R": max(abs(max(mode_idx, default=0)), abs(min(mode_idx, default=0)), abs(max(pump_idx, default=0)),
                 abs(min(pump_idx, default=0))) + 1,
        "delta_coverage_ok": True
    }