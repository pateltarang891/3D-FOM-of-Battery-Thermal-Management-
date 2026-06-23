import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse


# -------------------------
# USER SETTINGS
# -------------------------
parser = argparse.ArgumentParser()

# Fixed paper-comparison case
parser.add_argument("--c_rate", type=float, default=3.0)

# Paper x-direction unit spacing:
# 0.1908 / 300 = 0.000636 m
parser.add_argument("--nx_core", type=int, default=41) #300

parser.add_argument("--ny_core", type=int, default=21) # 61
parser.add_argument("--nz_core", type=int, default=21) # 61


parser.add_argument("--n_case", type=int, default=5)
parser.add_argument("--n_contact", type=int, default=5)

parser.add_argument("--output_tag", type=str, default="C3_paper_dx")
parser.add_argument("--make_plots", action="store_true", default=True)
parser.add_argument("--no_plots", action="store_true")

args = parser.parse_args()

C_RATE = args.c_rate
NX_CORE = args.nx_core
NY_CORE = args.ny_core
NZ_CORE = args.nz_core
N_CASE = args.n_case
N_CONTACT = args.n_contact

CONVECTION_MODE = "natural"
H_FIXED = 100.0
MAKE_PLOTS = bool(args.make_plots and not args.no_plots)
SHOW_PLOTS = False

OUTPUT_DIR = Path(f"outputs_{args.output_tag}")

THETA = 0.5

# -------------------------
# GEOMETRY
# -------------------------
Lcore_x = 0.1908
Lcore_y = 0.1000
Lcore_z = 0.1000

La = 0.0007       # case thickness [m]
Lb = 0.0005       # contact-layer thickness [m]
Llayer = La + Lb

Lx = Lcore_x + 2.0 * Llayer
Ly = Lcore_y + 2.0 * Llayer
Lz = Lcore_z + 2.0 * Llayer
V_core = Lcore_x * Lcore_y * Lcore_z

# -------------------------
# MATERIAL PROPERTIES
# -------------------------
kx_core = 1.035
ky_core = 24.840
kz_core = 24.840
rhoCp_core = 2.4569e6

ka = 170.0
kb = 0.60
rhoCp_case = 2770.0 * 875.0
rhoCp_contact = 1129.95 * 2055.1

T_inf = 300.0
T0 = 300.0

eps_surface = 0.25
sigma_SB = 5.67051e-8

# -------------------------
# ELECTRICAL / HEAT GENERATION INPUTS
# -------------------------
capacity_Ah = 185.3
I_batt = C_RATE * capacity_Ah

dEoc_dT = 0.00022

DOD_TABLE = np.array([
    0, 0.025,0.05,0.075,0.1,0.125,
    0.15,0.175,0.2,0.225,0.25,0.275,
    0.3,0.325,0.35,0.375,0.4,0.425,
    0.45,0.475,0.5,0.525,0.55,0.575,
    0.6,0.625,0.65,0.675,0.7,0.725,
    0.75,0.775,0.8,0.825,0.85,0.875,
    0.9,0.925,0.95,0.975,1
], dtype=float)

EOC_TABLE = np.array([
    4.1421,4.1226,4.1011,4.0796,4.0576,4.0391,
    4.0214,4.0034,3.985,3.9694,3.9528,3.9379,
    3.9217,3.9047,3.8893,3.8731,3.8569,3.8463,
    3.8354,3.8252,3.8149,3.8077,3.8009,3.7949,
    3.7874,3.7805,3.7736,3.7668,3.76,3.7515,
    3.743,3.7345,3.726,3.714,3.7029,3.6919,
    3.68,3.6606,3.641,3.6213,3.6051,
], dtype=float)

E_1C_TABLE = np.array([
    3.9966,3.9009,3.8551,3.8298,3.8125,3.7974,
    3.7838,3.771,3.7595,3.7486,3.7379,3.7286,
    3.7175,3.7073,3.6979,3.6889,3.6795,3.6715,
    3.663,3.6558,3.6477,3.6409,3.634,3.6276,
    3.6215,3.6153,3.6085,3.6024,3.5955,3.5881,
    3.5813,3.5722,3.5626,3.5511,3.5366,3.5186,
    3.4936,3.455,3.392,3.2871,3.1089,
], dtype=float)

E_2C_TABLE = np.array([
    3.8911,3.7078,3.5689,3.543,3.5492,3.5606,
    3.5716,3.5813,3.5881,3.5926,3.5966,3.5976,
    3.5983,3.5966,3.5966,3.5936,3.5913,3.5882,
    3.5852,3.5818,3.5782,3.5745,3.5711,3.567,
    3.5626,3.5579,3.554,3.5479,3.5429,3.5369,
    3.5291,3.5211,3.5115,3.5009,3.4863,3.4698,
    3.4443,3.4086,3.3475,3.2488,3.0264,
], dtype=float)

E_3C_TABLE = np.array([
    3.7626,3.5125,3.2867,3.2374,3.2465,3.2721,
    3.3029,3.3358,3.3665,3.3952,3.4194,3.44,
    3.4569,3.4689,3.4792,3.4877,3.4934,3.4977,
    3.5013,3.5025,3.5047,3.504,3.5038,3.5028,
    3.4989,3.4963,3.4928,3.4885,3.4826,3.4773,
    3.4689,3.4604,3.4502,3.4379,3.4217,3.4003,
    3.3707,3.3263,3.2498,3,3
], dtype=float)


def get_working_voltage_table(C_rate):
    if np.isclose(C_rate, 1.0):
        return E_1C_TABLE
    if np.isclose(C_rate, 2.0):
        return E_2C_TABLE
    if np.isclose(C_rate, 3.0):
        return E_3C_TABLE
    raise ValueError("Use only 1C, 2C, or 3C digitized voltage curves.")


E_WORK_TABLE = get_working_voltage_table(C_RATE)


def E_oc_from_DoD(dod):
    return np.interp(np.clip(dod, 0.0, 1.0), DOD_TABLE, EOC_TABLE)


def E_work_from_DoD(dod):
    return np.interp(np.clip(dod, 0.0, 1.0), DOD_TABLE, E_WORK_TABLE)


def current_profile(tt):
    return I_batt


def heat_generation_core(tt, T_avg_core):
    """
    Bernardi heat source in the core:
        qgen = I/V_core * [(Eoc - Ework) + T*dEoc_dT]
    """
    dod = np.clip(C_RATE * tt / 3600.0, 0.0, 1.0)
    I = current_profile(tt)

    qgen = (I / V_core) * (
        E_oc_from_DoD(dod)
        - E_work_from_DoD(dod)
        + T_avg_core * dEoc_dT
    )

    return float(qgen)


# ============================================================
# NONUNIFORM GRID
# ============================================================

def region_nodes(a, b, n, include_end=False):
    if include_end:
        return np.linspace(a, b, n)
    return np.linspace(a, b, n, endpoint=False)


def make_nonuniform_axis(Lcore, n_core, n_case, n_contact):
    """
    Node-centered nonuniform axis with nodes at material interfaces.
    Regions: case | contact | core | contact | case
    """
    x0 = 0.0
    x1 = La
    x2 = La + Lb
    x3 = La + Lb + Lcore
    x4 = La + Lb + Lcore + Lb
    x5 = La + Lb + Lcore + Lb + La

    pts = np.concatenate([
        region_nodes(x0, x1, n_case, include_end=False),
        region_nodes(x1, x2, n_contact, include_end=False),
        region_nodes(x2, x3, n_core, include_end=False),
        region_nodes(x3, x4, n_contact, include_end=False),
        region_nodes(x4, x5, n_case, include_end=True),
    ])
    return np.unique(np.round(pts, 15))


def cell_widths(axis):
    n = len(axis)
    w = np.zeros(n)
    w[0] = 0.5 * (axis[1] - axis[0])
    w[-1] = 0.5 * (axis[-1] - axis[-2])
    for i in range(1, n - 1):
        w[i] = 0.5 * (axis[i + 1] - axis[i - 1])
    return w


x = make_nonuniform_axis(Lcore_x, NX_CORE, N_CASE, N_CONTACT)
y = make_nonuniform_axis(Lcore_y, NY_CORE, N_CASE, N_CONTACT)
z = make_nonuniform_axis(Lcore_z, NZ_CORE, N_CASE, N_CONTACT)

nx, ny, nz = len(x), len(y), len(z)
N = nx * ny * nz

dx_cv = cell_widths(x)
dy_cv = cell_widths(y)
dz_cv = cell_widths(z)

DX, DY, DZ = np.meshgrid(dx_cv, dy_cv, dz_cv, indexing="ij")
Vcell_field = DX * DY * DZ


def idx(i, j, k):
    return (i * ny + j) * nz + k


X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

core_x_min = Llayer
core_x_max = Llayer + Lcore_x
core_y_min = Llayer
core_y_max = Llayer + Lcore_y
core_z_min = Llayer
core_z_max = Llayer + Lcore_z

contact_x_min = La
contact_x_max = La + Lb + Lcore_x + Lb
contact_y_min = La
contact_y_max = La + Lb + Lcore_y + Lb
contact_z_min = La
contact_z_max = La + Lb + Lcore_z + Lb

tol = 1.0e-12
core_mask = (
    (X >= core_x_min - tol) & (X <= core_x_max + tol) &
    (Y >= core_y_min - tol) & (Y <= core_y_max + tol) &
    (Z >= core_z_min - tol) & (Z <= core_z_max + tol)
)

contact_plus_core_mask = (
    (X >= contact_x_min - tol) & (X <= contact_x_max + tol) &
    (Y >= contact_y_min - tol) & (Y <= contact_y_max + tol) &
    (Z >= contact_z_min - tol) & (Z <= contact_z_max + tol)
)
contact_mask = contact_plus_core_mask & (~core_mask)
case_mask = ~(core_mask | contact_mask)

kx_field = np.zeros((nx, ny, nz))
ky_field = np.zeros((nx, ny, nz))
kz_field = np.zeros((nx, ny, nz))
rhoCp_field = np.zeros((nx, ny, nz))

kx_field[core_mask] = kx_core
ky_field[core_mask] = ky_core
kz_field[core_mask] = kz_core
rhoCp_field[core_mask] = rhoCp_core

kx_field[contact_mask] = kb
ky_field[contact_mask] = kb
kz_field[contact_mask] = kb
rhoCp_field[contact_mask] = rhoCp_contact

kx_field[case_mask] = ka
ky_field[case_mask] = ka
kz_field[case_mask] = ka
rhoCp_field[case_mask] = rhoCp_case


def harmonic_mean(a, b):
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


def volume_average(T, mask):
    return float(np.sum(T[mask] * Vcell_field[mask]) / np.sum(Vcell_field[mask]))


# ============================================================
# CONVECTION + RADIATION
# ============================================================

def h_natural(Ts, orientation, P):
    Ts = np.asarray(Ts, dtype=float)
    deltaT = np.maximum(np.abs(Ts - T_inf), 1.0e-9)

    if orientation == "vertical":
        if P > 0.152:
            f1, n_exp = 1.485088, 0.25
        else:
            f1, n_exp = 0.941145, 0.35
    elif orientation == "top":
        if P > 0.152:
            f1, n_exp = 1.36133, 0.25
        else:
            f1, n_exp = 0.830233, 0.33
    elif orientation == "bottom":
        if P > 0.152:
            f1, n_exp = 0.680665, 0.25
        else:
            f1, n_exp = 0.415117, 0.33
    else:
        raise ValueError("orientation must be vertical, top, or bottom")

    return f1 * (deltaT / P) ** n_exp


def h_surface_combined(Ts, orientation, P):
    Ts = np.asarray(Ts, dtype=float)

    if CONVECTION_MODE == "natural":
        hc = h_natural(Ts, orientation, P)
    elif CONVECTION_MODE == "fixed":
        hc = H_FIXED * np.ones_like(Ts)
    else:
        raise ValueError("CONVECTION_MODE must be 'natural' or 'fixed'")

    hr = eps_surface * sigma_SB * (Ts**2 + T_inf**2) * (Ts + T_inf)
    return hc + hr


# ============================================================
# NONUNIFORM FINITE-VOLUME CONDUCTION OPERATOR
# ============================================================

def build_conduction_operator_nonuniform():
    rows, cols, data = [], [], []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                p = idx(i, j, k)
                rhoCp_p = rhoCp_field[i, j, k]
                Vp = Vcell_field[i, j, k]
                diag = 0.0

                if i > 0:
                    kface = harmonic_mean(kx_field[i, j, k], kx_field[i - 1, j, k])
                    delta = x[i] - x[i - 1]
                    Aface = dy_cv[j] * dz_cv[k]
                    coeff = (kface * Aface / delta) / (rhoCp_p * Vp)
                    rows.append(p); cols.append(idx(i - 1, j, k)); data.append(coeff)
                    diag -= coeff

                if i < nx - 1:
                    kface = harmonic_mean(kx_field[i, j, k], kx_field[i + 1, j, k])
                    delta = x[i + 1] - x[i]
                    Aface = dy_cv[j] * dz_cv[k]
                    coeff = (kface * Aface / delta) / (rhoCp_p * Vp)
                    rows.append(p); cols.append(idx(i + 1, j, k)); data.append(coeff)
                    diag -= coeff

                if j > 0:
                    kface = harmonic_mean(ky_field[i, j, k], ky_field[i, j - 1, k])
                    delta = y[j] - y[j - 1]
                    Aface = dx_cv[i] * dz_cv[k]
                    coeff = (kface * Aface / delta) / (rhoCp_p * Vp)
                    rows.append(p); cols.append(idx(i, j - 1, k)); data.append(coeff)
                    diag -= coeff

                if j < ny - 1:
                    kface = harmonic_mean(ky_field[i, j, k], ky_field[i, j + 1, k])
                    delta = y[j + 1] - y[j]
                    Aface = dx_cv[i] * dz_cv[k]
                    coeff = (kface * Aface / delta) / (rhoCp_p * Vp)
                    rows.append(p); cols.append(idx(i, j + 1, k)); data.append(coeff)
                    diag -= coeff

                if k > 0:
                    kface = harmonic_mean(kz_field[i, j, k], kz_field[i, j, k - 1])
                    delta = z[k] - z[k - 1]
                    Aface = dx_cv[i] * dy_cv[j]
                    coeff = (kface * Aface / delta) / (rhoCp_p * Vp)
                    rows.append(p); cols.append(idx(i, j, k - 1)); data.append(coeff)
                    diag -= coeff

                if k < nz - 1:
                    kface = harmonic_mean(kz_field[i, j, k], kz_field[i, j, k + 1])
                    delta = z[k + 1] - z[k]
                    Aface = dx_cv[i] * dy_cv[j]
                    coeff = (kface * Aface / delta) / (rhoCp_p * Vp)
                    rows.append(p); cols.append(idx(i, j, k + 1)); data.append(coeff)
                    diag -= coeff

                rows.append(p); cols.append(p); data.append(diag)

    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))


def add_outer_surface_cooling(S, T):
    # x = 0
    Ts = T[0, :, :]
    h = h_surface_combined(Ts, "vertical", Lz)
    Aface = dy_cv[:, None] * dz_cv[None, :]
    S[0, :, :] += -(h * Aface / (rhoCp_field[0, :, :] * Vcell_field[0, :, :])) * (Ts - T_inf)

    # x = Lx
    Ts = T[-1, :, :]
    h = h_surface_combined(Ts, "vertical", Lz)
    Aface = dy_cv[:, None] * dz_cv[None, :]
    S[-1, :, :] += -(h * Aface / (rhoCp_field[-1, :, :] * Vcell_field[-1, :, :])) * (Ts - T_inf)

    # y = 0
    Ts = T[:, 0, :]
    h = h_surface_combined(Ts, "vertical", Lz)
    Aface = dx_cv[:, None] * dz_cv[None, :]
    S[:, 0, :] += -(h * Aface / (rhoCp_field[:, 0, :] * Vcell_field[:, 0, :])) * (Ts - T_inf)

    # y = Ly
    Ts = T[:, -1, :]
    h = h_surface_combined(Ts, "vertical", Lz)
    Aface = dx_cv[:, None] * dz_cv[None, :]
    S[:, -1, :] += -(h * Aface / (rhoCp_field[:, -1, :] * Vcell_field[:, -1, :])) * (Ts - T_inf)

    # z = 0 bottom
    Ts = T[:, :, 0]
    h = h_surface_combined(Ts, "bottom", Lx)
    Aface = dx_cv[:, None] * dy_cv[None, :]
    S[:, :, 0] += -(h * Aface / (rhoCp_field[:, :, 0] * Vcell_field[:, :, 0])) * (Ts - T_inf)

    # z = Lz top
    Ts = T[:, :, -1]
    h = h_surface_combined(Ts, "top", Lx)
    Aface = dx_cv[:, None] * dy_cv[None, :]
    S[:, :, -1] += -(h * Aface / (rhoCp_field[:, :, -1] * Vcell_field[:, :, -1])) * (Ts - T_inf)

    return S


def source_vector(q, tt):
    T = q.reshape((nx, ny, nz))
    S = np.zeros((nx, ny, nz), dtype=float)

    T_avg_core = volume_average(T, core_mask)
    qgen = heat_generation_core(tt, T_avg_core)

    S[core_mask] += qgen / rhoCp_field[core_mask]
    S = add_outer_surface_cooling(S, T)

    return S.reshape(-1)


# ============================================================
# TIME INTEGRATION
# ============================================================

def solve_CN(A, q0, t):
    n_steps = len(t)
    dt = t[1] - t[0]

    I_mat = sp.eye(q0.size, format="csc")
    LHS = (I_mat - THETA * dt * A).tocsc()
    RHS_mat = (I_mat + (1.0 - THETA) * dt * A).tocsc()

    print("Factorizing CN matrix...")
    lu = spla.splu(LHS)

    q = q0.copy()

    T_min = np.zeros(n_steps)
    T_max = np.zeros(n_steps)
    T_avg_total = np.zeros(n_steps)
    T_avg_core = np.zeros(n_steps)
    T_avg_contact = np.zeros(n_steps)
    T_avg_case = np.zeros(n_steps)
    Qgen_profile = np.zeros(n_steps)

    def record(step, q_now):
        Tn = q_now.reshape((nx, ny, nz))
        T_min[step] = Tn.min()
        T_max[step] = Tn.max()
        T_avg_total[step] = np.sum(Tn * Vcell_field) / np.sum(Vcell_field)
        T_avg_core[step] = volume_average(Tn, core_mask)
        T_avg_contact[step] = volume_average(Tn, contact_mask)
        T_avg_case[step] = volume_average(Tn, case_mask)
        Qgen_profile[step] = heat_generation_core(t[step], T_avg_core[step])

    record(0, q)

    print("Running CN time integration...")
    for n in range(1, n_steps):
        t_mid = 0.5 * (t[n - 1] + t[n])
        rhs = RHS_mat @ q + dt * source_vector(q, t_mid)
        q = lu.solve(rhs)

        record(n, q)

        if n == 1 or n % max(1, ((n_steps - 1) // 10)) == 0 or n == n_steps - 1:
            print(
                f"step {n:4d}/{n_steps-1}, t={t[n]:7.2f} s, "
                f"DoD={C_RATE*t[n]/3600.0:.3f}, "
                f"Tmin={T_min[n]:.3f} K, "
                f"Tavg_core={T_avg_core[n]:.3f} K, "
                f"Tmax={T_max[n]:.3f} K"
            )

    histories = {
        "T_min": T_min,
        "T_max": T_max,
        "T_avg_total": T_avg_total,
        "T_avg_core": T_avg_core,
        "T_avg_contact": T_avg_contact,
        "T_avg_case": T_avg_case,
        "Qgen_profile": Qgen_profile,
    }

    return q, histories


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Case: 3C discharge")
    print("Time setup: Nt = 1200 steps, tf = 3600 / C_RATE")
    print(f"Total time tf = {3600.0 / C_RATE:.6f} s")
    print(f"Grid: nx={nx}, ny={ny}, nz={nz}, N={N}")
    print(f"x spacing min/max: {np.min(np.diff(x)):.6e}, {np.max(np.diff(x)):.6e} m")
    print(f"y spacing min/max: {np.min(np.diff(y)):.6e}, {np.max(np.diff(y)):.6e} m")
    print(f"z spacing min/max: {np.min(np.diff(z)):.6e}, {np.max(np.diff(z)):.6e} m")
    print(f"Core dx target/check: {Lcore_x / NX_CORE:.6e} m")
    print(f"Node counts: core={np.count_nonzero(core_mask)}, contact={np.count_nonzero(contact_mask)}, case={np.count_nonzero(case_mask)}")

    A = build_conduction_operator_nonuniform()
    print(f"A shape={A.shape}, nnz={A.nnz}")

    tf = 3600.0 / C_RATE
    Nt = 1200
    t = np.linspace(0.0, tf, Nt + 1)
    DoD = np.clip(C_RATE * t / 3600.0, 0.0, 1.0)

    print(f"Actual dt = {t[1] - t[0]:.6f} s")

    q0 = T0 * np.ones(N)
    q_final, histories = solve_CN(A, q0, t)

    T_min = histories["T_min"]
    T_max = histories["T_max"]
    T_avg_total = histories["T_avg_total"]
    T_avg_core = histories["T_avg_core"]
    T_avg_contact = histories["T_avg_contact"]
    T_avg_case = histories["T_avg_case"]
    Qgen_profile = histories["Qgen_profile"]

    summary = pd.DataFrame({
        "time_s": t,
        "DoD": DoD,
        "current_A": [current_profile(tt) for tt in t],
        "qgen_core_W_per_m3": Qgen_profile,
        "T_min_K": T_min,
        "T_avg_total_K": T_avg_total,
        "T_avg_core_K": T_avg_core,
        "T_avg_contact_K": T_avg_contact,
        "T_avg_case_K": T_avg_case,
        "T_max_K": T_max,
        "T_min_rise_K": T_min - T0,
        "T_avg_total_rise_K": T_avg_total - T0,
        "T_avg_core_rise_K": T_avg_core - T0,
        "T_max_rise_K": T_max - T0,
    })

    summary_csv = OUTPUT_DIR / "model_N10_CN_nonuniform_summary.csv"
    summary.to_csv(summary_csv, index=False)

    final = q_final.reshape((nx, ny, nz))

    x_profile = pd.DataFrame({
        "x_m": x,
        "T_centerline_K": final[:, ny // 2, nz // 2],
        "T_side_y_surface_K": final[:, 0, nz // 2],
        "T_bottom_z_surface_K": final[:, ny // 2, 0],
    })
    x_profile_csv = OUTPUT_DIR / "model_N10_CN_nonuniform_final_x_profile.csv"
    x_profile.to_csv(x_profile_csv, index=False)

    if MAKE_PLOTS:

        def save_or_show(filename):
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / filename, dpi=250, bbox_inches="tight")
            if SHOW_PLOTS:
                plt.show()
            plt.close()

        # ------------------------------------------------------------
        # Plot 1: Material map at middle y-plane
        # ------------------------------------------------------------
        midy = ny // 2
        mat = np.zeros((nx, nz), dtype=float)
        mat[case_mask[:, midy, :]] = 1.0
        mat[contact_mask[:, midy, :]] = 2.0
        mat[core_mask[:, midy, :]] = 3.0
        Xp_mat, Zp_mat = np.meshgrid(x, z, indexing="ij")

        plt.figure(figsize=(10, 4.8))
        cf = plt.contourf(Xp_mat, Zp_mat, mat, levels=[0.5, 1.5, 2.5, 3.5])
        plt.colorbar(cf, ticks=[1, 2, 3], label="1 = case, 2 = contact, 3 = core")
        plt.axvline(core_x_min, linewidth=1.0)
        plt.axvline(core_x_max, linewidth=1.0)
        plt.axhline(core_z_min, linewidth=1.0)
        plt.axhline(core_z_max, linewidth=1.0)
        plt.xlabel("X-coordinate [m]")
        plt.ylabel("Z-coordinate [m]")
        plt.title("Material map at y = mid-plane")
        save_or_show("material_map_xz_mid_y.png")

        # ------------------------------------------------------------
        # Plot 2: Temperature vs DoD
        # ------------------------------------------------------------
        plt.figure(figsize=(9, 5.5))
        plt.plot(DoD, T_max, linewidth=2.2, label="Maximum temperature")
        plt.plot(DoD, T_avg_core, linewidth=2.2, linestyle="--", label="Core-average temperature")
        plt.plot(DoD, T_avg_total, linewidth=2.0, linestyle="-.", label="Total-average temperature")
        plt.plot(DoD, T_min, linewidth=2.2, linestyle=":", label="Minimum temperature")
        plt.xlabel("Depth of discharge")
        plt.ylabel("Temperature [K]")
        plt.title(f"Model N10 CN nonuniform grid: Temperature vs DoD at {C_RATE:.1f}C")
        plt.xlim(0, 1)
        plt.grid(True)
        plt.legend()
        save_or_show("temperature_vs_DoD.png")

        # ------------------------------------------------------------
        # Plot 3: Temperature rise vs DoD
        # ------------------------------------------------------------
        plt.figure(figsize=(9, 5.5))
        plt.plot(DoD, T_max - T0, linewidth=2.2, label="Maximum temperature rise")
        plt.plot(DoD, T_avg_core - T0, linewidth=2.2, linestyle="--", label="Core-average temperature rise")
        plt.plot(DoD, T_avg_total - T0, linewidth=2.0, linestyle="-.", label="Total-average temperature rise")
        plt.plot(DoD, T_min - T0, linewidth=2.2, linestyle=":", label="Minimum temperature rise")
        plt.xlabel("Depth of discharge")
        plt.ylabel("Temperature rise [K]")
        plt.title(f"Model N10 CN nonuniform grid: Temperature rise at {C_RATE:.1f}C")
        plt.xlim(0, 1)
        plt.grid(True)
        plt.legend()
        save_or_show("temperature_rise_vs_DoD.png")

        # ------------------------------------------------------------
        # Plot 4: Final x-direction temperature profile
        # ------------------------------------------------------------
        j_mid = ny // 2
        k_mid = nz // 2
        j_side = 0
        k_bottom = 0

        plt.figure(figsize=(9, 5.5))
        plt.plot(x, final[:, j_mid, k_mid], linewidth=2.2, label="Centerline: y = mid, z = mid")
        plt.plot(x, final[:, j_side, k_mid], linewidth=2.2, linestyle="--", label="Outer side: y = surface, z = mid")
        plt.plot(x, final[:, j_mid, k_bottom], linewidth=2.2, linestyle=":", label="Outer bottom: y = mid, z = bottom")
        plt.axvline(core_x_min, linewidth=1.0)
        plt.axvline(core_x_max, linewidth=1.0)
        plt.xlabel("X-coordinate [m]")
        plt.ylabel("Temperature [K]")
        plt.title(f"Final X-direction temperature profile, DoD = 1, {C_RATE:.1f}C")
        plt.xlim(0, Lx)
        plt.grid(True)
        plt.legend()
        save_or_show("final_x_temperature_profile.png")

        # ------------------------------------------------------------
        # Plot 5: Final X-Z temperature contour at middle y-plane
        # ------------------------------------------------------------
        def safe_contour_levels(Tslice, nlevels=50, pad=1.0e-3):
            local_min = float(np.nanmin(Tslice))
            local_max = float(np.nanmax(Tslice))
            if np.isclose(local_min, local_max):
                local_min -= pad
                local_max += pad
            return local_min, local_max, np.linspace(local_min, local_max, nlevels)

        Tslice_xz = final[:, midy, :]
        Xp_xz, Zp_xz = np.meshgrid(x, z, indexing="ij")
        local_min, local_max, levels = safe_contour_levels(Tslice_xz, nlevels=50)

        plt.figure(figsize=(10, 5))
        cf = plt.contourf(Xp_xz, Zp_xz, Tslice_xz, levels=levels)
        cs = plt.contour(
            Xp_xz,
            Zp_xz,
            Tslice_xz,
            levels=np.linspace(local_min, local_max, 10),
            linewidths=0.6,
        )
        plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
        plt.axvline(core_x_min, linewidth=1.0)
        plt.axvline(core_x_max, linewidth=1.0)
        plt.axhline(core_z_min, linewidth=1.0)
        plt.axhline(core_z_max, linewidth=1.0)
        plt.colorbar(cf, label="Temperature [K]")
        plt.xlabel("X-coordinate [m]")
        plt.ylabel("Z-coordinate [m]")
        plt.title(
            f"Final X-Z temperature contour at y = mid-plane\n"
            f"Model N10 CN nonuniform grid, DoD = 1, {C_RATE:.1f}C"
        )
        plt.xlim(0, Lx)
        plt.ylim(0, Lz)
        save_or_show("final_xz_temperature_contour_mid_y.png")

        # ------------------------------------------------------------
        # Plot 6: Heat generation vs DoD
        # ------------------------------------------------------------
        plt.figure(figsize=(9, 5.5))
        plt.plot(DoD, Qgen_profile, linewidth=2.2)
        plt.xlabel("Depth of discharge")
        plt.ylabel("Core volumetric heat generation [W/m$^3$]")
        plt.title("Bernardi heat generation applied only inside the core")
        plt.xlim(0, 1)
        plt.grid(True)
        save_or_show("qgen_vs_DoD.png")

    print(f"Final Tmin        = {T_min[-1]:.6f} K")
    print(f"Final Tavg_total  = {T_avg_total[-1]:.6f} K")
    print(f"Final Tavg_core   = {T_avg_core[-1]:.6f} K")
    print(f"Final Tmax        = {T_max[-1]:.6f} K")
    print(f"Final Tmin rise   = {T_min[-1]-T0:.6f} K")
    print(f"Final Tavg rise   = {T_avg_total[-1]-T0:.6f} K")
    print(f"Final Tmax rise   = {T_max[-1]-T0:.6f} K")
    print(f"Final qgen        = {Qgen_profile[-1]:.6f} W/m^3")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {x_profile_csv}")

if __name__ == "__main__":
    main()
