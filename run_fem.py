import sys
import numpy as np
from matplotlib import pyplot as plt
from fem import generate_snapshots_mesh
from load_examples import load_example_fem

try:
    import fenics as fen
except Exception as e:
    print("Could not find FEniCS!")
    raise e

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
(t_max, R, u0, u1, Vspace, where_dirichlet,
 x, y, ts, cutoff, add_forcing) = load_example_fem(example_tag)

#%% perform 2D FEM simulation
xy = Vspace.tabulate_dof_coordinates()
V = generate_snapshots_mesh(Vspace, ts, u0(xy), u1(xy), where_dirichlet,
                            add_forcing)

#%% predict at t_max
Y, X = np.meshgrid(y, x)
XY = np.c_[X.flatten(), Y.flatten()]
V_f = fen.Function(Vspace)
V_f.vector().set_local(V[:, -1])
U = np.zeros(len(XY))
for j, xy in enumerate(XY):
    try:
        U[j] = V_f(fen.Point(xy[0], xy[1]))
    except RuntimeError:
        pass
Umax = max(- np.min(U), np.max(U)) + 1e-10
U_ = U.reshape(-1, len(y))

#%% plot
plt.figure(figsize=(15, 15))
p = plt.contourf(X, Y, U_, levels = np.linspace(-Umax, Umax, 100),
                 cmap = "seismic")
plt.colorbar(p)
plt.show()
