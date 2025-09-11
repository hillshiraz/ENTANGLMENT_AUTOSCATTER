import numpy as np
from typing import Sequence
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def build_g_nu(mode_freqs: Sequence[float], pump_freqs: Sequence[float], pump_amps: Sequence[complex], kappa: float = 1,h_bar: float= 1) :
    w = np.asarray(mode_freqs, dtype=float)
    Wp = np.asarray(pump_freqs, dtype=float)
    A = np.asarray(pump_amps, dtype=complex)
    N = w.size
    P = Wp.size
    if A.size != P:
        raise ValueError("Error: pump_amps must have the same length as pump_freqs")

    G = np.zeros((N, N), dtype=complex) #linear coupling
    NU = np.zeros((N, N), dtype=complex) #squeezing coupling

    # --- SPS contribution → NU_{k,l} += -h_bar*kappa * A_i * A_j
    for i in range(P):
        for j in range(P):
            sum_pumps = Wp[i] + Wp[j]
            coef_sps = -h_bar* kappa * A[i] * A[j]
            for k in range(N):
                for l in range(N):
                    if (w[k] + w[l]) == sum_pumps:
                        NU[k, l] += coef_sps

    # BS contribution → G[k,l]+= -h_bar*kappa * (A_i^*) * A_j
    for i in range(P):
        for j in range(P):
            coef_bs = -h_bar* kappa * np.conj(A[i]) * A[j]
            for k in range(N):
                for l in range(N):
                    if (Wp[i] + w[k])== (Wp[j] + w[l]): #when a pump+mode equals another pump+mode (delta 2)
                        G[k, l] += coef_bs
    return G, NU

def S_bdg_from_G_NU(G, NU, kappa):
    """
    the rescaled BdG Hamiltonian:
        H = kappa^{-1/2} [[G, NU],
                          [NU*, G*]] kappa^{-1/2}
    """
    G  = np.asarray(G, dtype=complex)
    NU = np.asarray(NU, dtype=complex)
    n = G.shape[0]
    if G.shape != (n, n) or NU.shape != (n, n):
        raise ValueError("G and NU must be N×N")

    #create the BdG matrix
    H_bdg = np.block([[G,           NU        ],
                      [NU.conj(),   G.conj()  ]])

    #kappa must be 2N×2N diagonal
    kappa = np.asarray(kappa, dtype=float)
    if kappa.shape != (n, n):
        raise ValueError("kappa must be a 2N×2N matrix")

    Kappa_2N = np.block([[kappa, np.zeros((n, n), complex)],
                         [np.zeros((n, n), complex), kappa]])

    kdiag = np.diag(Kappa_2N) #so I will check only the diagonal elements not all of them
    if np.any(kdiag <= 0):
        raise ValueError("all κ_i must be positive to form κ^{-1/2}")

    # κ^{-1/2}
    k_inv_sqrt = np.diag(1.0 / np.sqrt(kdiag))
    H = k_inv_sqrt @ H_bdg @ k_inv_sqrt
    return H

def scattering_matrix_from_H(H,Gamma,Kappa):
    """
    Build the 2N×2N scattering matrix S based on: S = I_{2N} + ( -i σ_z H - Γ_2N/2 - I_{2N}/2 )^(-1)
    where Γ_2N = diag(gamma, gamma). (from γ = Γ*κ^(-1))
    """
    H = np.asarray(H, dtype=complex)
    two_N = H.shape[0]
    size_n = two_N // 2

    #σ_z = diag(I_N, -I_N)
    I_N = np.eye(size_n, dtype=complex)
    sigma_z = np.block([[ I_N, np.zeros((size_n, size_n), complex)],
                        [ np.zeros((size_n, size_n), complex), -I_N ]])

    #Γ_2N = diag(Gamma, Gamma)
    Gamma_2N = np.block([[Gamma, np.zeros((size_n, size_n), complex)],
                         [np.zeros((size_n, size_n), complex), Gamma]])
    # Gamma_2N = diag(Gamma, Gamma)
    Kappa_2N = np.block([[Kappa, np.zeros((size_n, size_n), complex)],
                         [np.zeros((size_n, size_n), complex), Kappa]])
    kappa_inv = np.diag(1.0 / np.diag(Kappa_2N))
    gamma= Gamma_2N @ kappa_inv
    I_2N = np.eye(two_N, dtype=complex)

    A = -1j * (sigma_z @ H) - 0.5 * gamma - 0.5 * I_2N
    #S=I+A^(-1)-> it is like solving AX=I
    X = np.linalg.solve(A, I_2N)
    S = I_2N + X
    return S

#ver2- start (now try to create the isolator)

def candidate_G_from_params(g12, phi12, g13, g23, Delta2):
    """
    H:
        [ 0               g12 e^{+i phi12}    g13 ]
        [ g12 e^{-i phi12}    Delta2          g23 ]
        [ g13                 g23              0  ]

    NU = 0 (isolator uses beam-splitter couplings only)
    """
    G = np.array([[0,                        g12*np.exp(1j*phi12), g13],
                  [g12*np.exp(-1j*phi12),   Delta2,                g23],
                  [g13,                     g23,                   0  ]], dtype=complex)
    NU = np.zeros_like(G)
    return G, NU


# ---------- Build S and pick the ports block ----------
def S_ports_from_params(params):
    """
    params = [g12, phi12, g13, g23, Delta2, gamma3, gout1, gout2, gin1, gin2]
    returns: S_ports (2x2), S_aa (3x3), gauge-out(2,), gauge-in(2,)
    """
    g12, phi12, g13, g23, Delta2, gamma3, gout1, gout2, gin1, gin2 = params
    G, NU = candidate_G_from_params(g12, phi12, g13, g23, Delta2)

    N = 3
    Kappa = np.eye(2*N)               # 2N×2N
    GammaN = np.diag([0.0, 0.0, gamma3])     # only the auxiliary (mode 3) is intrinsically lossy

    H_bdg = S_bdg_from_G_NU(G, NU, Kappa)    # your function
    S_big = scattering_matrix_from_H(H_bdg, GammaN, Kappa)

    S_aa = S_big[:N, :N]
    ports = [0, 1]                           # modes 1 & 2 are the I/O ports
    S_ports = S_aa[np.ix_(ports, ports)]  # 2x2
    gamma_out = np.array([gout1, gout2])     # gauge phases (out)
    gamma_in  = np.array([gin1,  gin2 ])     # gauge phases (in)
    return S_ports, S_aa, gamma_out, gamma_in

# ---------- Loss function L exactly like in equation 3 ----------
def loss_function(S_target, params, return_terms=False):
    S_ports, _, g_out, g_in = S_ports_from_params(params) #i dont use the S_aa
    phase = np.exp(1j*(g_out[:, None] - g_in[None, :]))  #e^{i(gamma_out - gamma_in)}
    res = S_ports - S_target * phase
    L = float(np.sum(np.abs(res)**2))

    if return_terms: #in case of an error so I can see those parms and detect the problem
        return L, res, np.abs(res)**2
    return L

# ---------- print what we are minimizing (matrix view) ----------
def print_loss_breakdown(S_target, params):
    L, R, R2 = loss_function(S_target, params, return_terms=True)
    np.set_printoptions(precision=3, suppress=True)
    S_ports, *_ = S_ports_from_params(params)
    g12, phi12, g13, g23, Delta2, gamma3, gout1, gout2, gin1, gin2 = params

    print("\n--- Loss Eq.(3) breakdown ---")
    print("Current S_ports:\n", np.round(S_ports, 3))
    print("S_target:\n", np.round(S_target, 3))
    print("Residual R = S - S_target * e^{i(γ_out-γ_in)}:\n", np.round(R, 3))
    print("|R|^2 elementwise:\n", np.round(R2, 6))
    print("Loss =", L)
    print("Params: g12=%.3f, phi12=%.3f, g13=%.3f, g23=%.3f, Δ2=%.3f, γ3=%.3f,"
          " gout=(%.3f,%.3f), gin=(%.3f,%.3f)" %
          (g12, phi12, g13, g23, Delta2, gamma3, gout1, gout2, gin1, gin2))


def optimize_isolator(show_every=1):
    # target 2×2 isolator: perfect forward 1→2, zero reverse 2→1
    S_target = np.array([[0, 0],
                         [1, 0]], dtype=complex)

    p0 = np.array([1.0,  np.pi/2,  1.0,  1.0,  0.0,   5.0,   0.0, 0.0, 0.0, 0.0])
    #               g12  phi12     g13   g23    Δ2    γ3   gout1 gout2 g_in1 g_in2

    bounds = [(0,None), (-np.pi, np.pi), (0,None), (0,None), (None,None),
              (0,None), (None,None), (None,None), (None,None), (None,None)]

    history = []
    def cb(p):
        history.append(loss_function(S_target, p))
        if show_every and len(history) % show_every == 0:
            print(f"iter {len(history):3d}: loss= {history[-1]:.6e}")

    res = minimize(lambda p: loss_function(S_target, p),
                   p0, method='Powell', bounds=bounds, callback=cb,
                   options=dict(maxiter=1500, xtol=1e-10, ftol=1e-12))

    # final report
    print_loss_breakdown(S_target, res.x)

    # isolation quality on ports (|S21|^2 forward vs |S12|^2 reverse)
    S_ports, *_ = S_ports_from_params(res.x)
    T_fwd = abs(S_ports[1,0])**2
    T_rev = abs(S_ports[0,1])**2
    iso_dB = 10*np.log10((T_fwd + 1e-16)/(T_rev + 1e-16))
    print(f"Forward |S21|^2 = {T_fwd:.6f}, Reverse |S12|^2 = {T_rev:.6f}, Isolation ≈ {iso_dB:.2f} dB")

    return res, np.array(history)

# ----------plotting of loss history ----------
def plot_loss(history):
    y = history
    x = np.arange(1, len(y)+1)
    plt.figure()
    plt.semilogy(x, y)
    plt.xlabel("Iteration")
    plt.ylabel("Loss function")
    plt.title("Equation 3 loss minimization")
    plt.grid(True, which="both", ls=":")
    plt.show()

