import numpy as np
import math
import matplotlib.pyplot as plt

def extract_deltas_from_info(info, N):
    """
    Pulls Delta0..Delta{N-1} from info['solution_dict_complete'].
    Missing entries default to 0.
    Returns array of length N in rad/s (or your chosen units).
    """
    sol = info.get('solution_dict_complete', {})
    deltas = np.zeros(N, dtype=float)
    for i in range(N):
        key = f"Delta{i}"
        if key in sol:
            deltas[i] = float(sol[key])
    return deltas

def mode_frequencies(omega0, FSR, mode_indices, deltas=None):
    """
    Compute absolute mode frequencies ω_i = ω0 + m_i*FSR (+ Δ_i if provided).
    All inputs are in rad/s; returns rad/s.
    """
    omega = omega0 + np.asarray(mode_indices, dtype=float) * FSR
    if deltas is not None:
        omega = omega + np.asarray(deltas, dtype=float)
    return omega

def pump_report(out, freq_unit="THz"):
    """
    Pretty-print the selected pump tones from `out` returned by synthesize_pumps_auto(...).
    freq_unit: "THz" (default) or "GHz" or "Hz".
    """
    assert all(k in out for k in ["pump_indices","pump_frequencies","pump_amplitudes","pump_phases","FSR_out","mode_indices"]), \
        "out dict missing required keys."

    # unit conversion
    conv = {"Hz": 1/(2*np.pi),
            "GHz": 1e-9/(2*np.pi),
            "THz": 1e-12/(2*np.pi)}[freq_unit]

    idx   = np.array(out["pump_indices"])
    frad  = np.array(out["pump_frequencies"])          # rad/s
    f     = frad * conv                                 # to unit
    amps  = np.array(out["pump_amplitudes"])
    ph    = np.array(out["pump_phases"])
    phdeg = np.degrees(ph)

    # sort by frequency for a tidy table
    order = np.argsort(f)
    idx, f, amps, ph, phdeg = idx[order], f[order], amps[order], ph[order], phdeg[order]

    # header
    print("\nSelected pump tones")
    print(f"FSR_out = {out['FSR_out']*conv:.6f} {freq_unit}")
    if "g_error_fro2" in out and "nu_error_fro2" in out:
        print(f"Fit errors: ||g_err||^2={out['g_error_fro2']:.3e}, ||nu_err||^2={out['nu_error_fro2']:.3e}")
    # table
    print("\n idx    freq [{unit}]        |A|         phase [rad]    phase [deg]".format(unit=freq_unit))
    for i,fi,ai,phi,phd in zip(idx, f, amps, ph, phdeg):
        print(f"{i:>4d}  {fi:>12.6f}    {ai:>10.6f}    {phi:>12.6f}    {phd:>12.3f}")

def mode_report(omega_modes, mode_indices, freq_unit="THz"):
    conv = {"Hz": 1/(2*np.pi), "GHz": 1e-9/(2*np.pi), "THz": 1e-12/(2*np.pi)}[freq_unit]
    f = omega_modes * conv
    order = np.argsort(f)
    print("\nMode teeth (amplitudes ~ 0)")
    print("\n idx    freq [{unit}]".format(unit=freq_unit))
    for i in order:
        print(f"{int(mode_indices[i]):>4d}  {f[i]:>12.6f}")

def plot_comb_with_modes(out, omega_modes, freq_unit="THz", stem_scale=1.0):
    import matplotlib.pyplot as plt
    conv = {"Hz": 1/(2*np.pi), "GHz": 1e-9/(2*np.pi), "THz": 1e-12/(2*np.pi)}[freq_unit]

    # Frequencies
    f_pumps = np.array(out.get("pump_frequencies", []), float) * conv
    f_modes = np.array(omega_modes, float) * conv

    # Indices (relative to omega_0)
    pump_idx = np.array(out.get("pump_indices", []), int)
    mode_idx = np.array(out.get("mode_indices", []), int)

    # Amplitudes for stem heights
    A = np.array(out.get("pump_amplitudes", []), complex)
    if A.size == 0:
        heights = np.array([], float)
    else:
        maxA = np.max(np.abs(A))
        heights = stem_scale * (np.abs(A)/maxA) if maxA > 0 else np.zeros_like(A, float)

    plt.figure(figsize=(9, 3.6))

    # Plot pumps as stems
    for x, h, p in zip(f_pumps, heights, pump_idx):
        plt.vlines(x, 0.0, h, linewidth=2)
        plt.plot([x], [h], 'o', ms=5)
        # label above stem tip
        plt.text(x, h + 0.03*max(1.0, stem_scale), rf"$P_{{{int(p)}}}$",
                 ha='center', va='bottom', fontsize=10)

    # Plot modes as x markers on baseline
    plt.plot(f_modes, np.zeros_like(f_modes), 'x', markersize=6)
    # label each mode a bit above baseline (or slightly below if you prefer)
    baseline_offset = 0.04*max(1.0, stem_scale)
    for xm, m in zip(f_modes, mode_idx):
        plt.text(xm, baseline_offset, rf"$M_{{{int(m)}}}$",
                 ha='center', va='bottom', fontsize=10)

    # Axes cosmetics
    plt.xlabel(f"Frequency [{freq_unit}]")
    plt.ylabel("Normalized |A|")
    plt.title("Elctro-optic comb with Pumps (stems) and Modes (×)")

    # Nice x limits with margin
    xs = []
    if f_pumps.size: xs.append((f_pumps.min(), f_pumps.max()))
    if f_modes.size: xs.append((f_modes.min(), f_modes.max()))
    if xs:
        xmin = min(a for a, b in xs)
        xmax = max(b for a, b in xs)
        margin = 0.07 * max(1e-9, (xmax - xmin))
        plt.xlim(xmin - margin, xmax + margin)

    # y limits
    ymax = 1.1 * (heights.max() if heights.size else 1.0)
    plt.ylim(0, max(ymax, baseline_offset * 3))

    plt.grid(True, linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.show()
