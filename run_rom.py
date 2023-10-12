import sys
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from domain import domain
from fem import generate_snapshots_polar
from timetable import build_ray_sequence, get_light_cone_args
from load_examples import load_example_rom

#%% choose example
if len(sys.argv) > 1:
    example_tag = sys.argv[1]
else:
    example_tag = input("Input example_tag:\n")
example_tag = example_tag.lower().replace(" ","").strip()

allowed_tags = ["wedge_1", "wedge_2", "wedge_3", "wedge_4", "cavity",
                "room_tol2.5e-2", "room_tol1e-3",
                "room_harmonic_1", "room_harmonic_5"]
if example_tag not in allowed_tags:
    raise Exception(("Value of example_tag not recognized. Allowed values:\n"
                     "{}").format(allowed_tags))

#%% load example
(t_max, R, u0, u1, outer, inner, bcs, x, y, mesh,
 ts, cutoff, add_forcing, ks_diffraction) = load_example_rom(example_tag)

D = domain()
D.points = outer
D.edges = [[i, i + 1] for i in range(len(outer) - 1)] + [[len(outer) - 1, 0]]
for pt in inner:
    D.edges += [[len(D.points) + i, len(D.points) + i + 1]
                                                   for i in range(len(pt) - 1)]
    D.edges += [[len(D.points) + len(pt) - 1, len(D.points)]]
    D.points += pt
D.bcs = bcs

#%% perform 1D FEM simulation
V = generate_snapshots_polar(mesh, ts, u0(mesh), u1(mesh), add_forcing)
V_amplitude = np.maximum.accumulate(np.max(np.abs(V), axis = 0)[::-1])[::-1]

def weights_cutoff(t):
    i_t_supp = np.where(ts <= t)[0]
    if len(i_t_supp) == 0: i_t_supp = [0]
    return cutoff / V_amplitude[i_t_supp[-1]]

#%% build ray
rays, angle_weights = build_ray_sequence(D, (0.,) * 2, R, t_max,
                                         weights_cutoff = weights_cutoff,
                                         ks_diffraction = ks_diffraction)

#%% set up interpolation/extrapolation of 1D FEM simulation
def predict_2d(x_1, x_2, X_1, X_2):
    # spline over x_1, hat-function over x_2
    # X_1 is mesh for x_1, X_2 is mesh for x_2
    # x_2 must be scalar!
    if x_2 < 0: return 0. * x_1
    idx = np.where(X_2 <= x_2)[0][-1]
    if x_2 < X_2[0]:
        V_ = V[:, 0]
    elif idx == len(X_2) - 1:
        V_ = V[:, -1]
    else:
        w = (X_2[idx + 1] - x_2) / (X_2[idx + 1] - X_2[idx])
        V_ = w * V[:, idx] + (1 - w) * V[:, idx + 1]
    v = interp1d(X_1, V_, bounds_error = False, fill_value = 0.)(x_1)
    v[np.logical_or(np.isinf(v), np.isnan(v))] = 0.
    return v
# spline over r, hat-function over t
predict_V = lambda r_, t_: predict_2d(r_, t_, mesh, ts)

#%% predict at t_max
Y, X = np.meshgrid(y, x)
XY = np.c_[X.flatten(), Y.flatten()]
U = np.zeros(len(XY))
for ray, weight in zip(rays, angle_weights):
    r = ((XY[:, 0] - ray.c_x[0]) ** 2 + (XY[:, 1] - ray.c_x[1]) ** 2) ** .5
    theta = np.angle((XY[:, 0] - ray.c_x[0]) + 1j * (XY[:, 1] - ray.c_x[1]))
    lc = D.check_light_cone(XY, ray.c_x, **get_light_cone_args(ray))
    U += weight(theta) * lc * predict_V(r + ray.c_t, t_max)
Umax = max(- np.min(U), np.max(U)) + 1e-10
U_ = U.reshape(-1, len(y))

#%% plot
plt.figure(figsize=(15, 15))
p = plt.contourf(X, Y, U_, levels = np.linspace(-Umax, Umax, 100),
                 cmap = "seismic")
plt.colorbar(p)
plt.show()
