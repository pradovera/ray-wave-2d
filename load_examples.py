import numpy as np

def load_example_rom(tag):
    # t_max - time horizon
    if tag[: 5] == "wedge":
        t_max = 5.
    elif tag == "cavity":
        t_max = 9.
    else:#if tag[: 4] == "room":
        t_max = 20.

    # R - radius of data support
    if tag[: 5] == "wedge" or tag == "cavity":
        width_front = .2
    elif tag[: 8] =="room_tol":
        width_front = .25
    else:#if tag[: 13] == "room_harmonic":
        width_front = 0.
    R = 5 * width_front

    # ts - t-grid for timestepping
    if tag[: 5] == "wedge" or tag == "cavity":
        n_mesh = 1000
    elif tag[: 8] =="room_tol":
        n_mesh = 840
    elif tag == "room_harmonic_1":
        n_mesh = 400
    else:#if tag == "room_harmonic_5":
        n_mesh = 2000
    ts = np.linspace(0., (1 + 1. / (2 * n_mesh)) * t_max, 2 * (n_mesh + 1))
    dt = ts[1]

    # mesh - r-grid for FEM
    mesh = np.linspace(0., t_max + R, n_mesh + 1)

    # u0 - initial condition
    if tag[: 5] == "wedge" or tag == "cavity":
        u0 = lambda x: np.exp(-.5 * (x / width_front) ** 2.)
    elif tag[: 8] =="room_tol":
        u0 = lambda x: (np.exp(-.5 * (x / width_front) ** 2.)
                      * (1 - (x / width_front) ** 2.))
    else:#if tag[: 13] == "room_harmonic":
        u0 = lambda x: np.zeros_like(x)

    # u1 - initial velocity
    u1 = lambda x: np.zeros_like(x)

    # outer - points on outer boundary
    if tag[: 5] == "wedge":
        if tag == "wedge_1":
            outer = [(-1.5, 1.), (4., 1.), (4., -4.), (-1.5, -4.)]
        if tag == "wedge_2":
            outer = [(-1.5, 1.), (4., 1.), (4., -4.), (.5, -4.)]
        if tag == "wedge_3":
            outer = [(-1.5, 1.), (-1.5, 4.), (4., 4.), (4., -4.), (-3.5, -4.)]
        if tag == "wedge_4":
            outer = [(-1., -4.), (-1., 6.), (6., 6.), (6., -9.), (-3., -9.)]
        mult = (t_max - 1) / np.linalg.norm(outer[0])
        outer = [(x[0] * mult, x[1] * mult) for x in outer]
    elif tag == "cavity":
        outer = [(5., 14.), (1., 4.), (1., -3.),
                 (-5., -3.), (-3., 2.), (-3., 14.)]
    else:#if tag[: 4] == "room":
        outer = [(-10., -1.5), (-10., 4.), (-7., 4.), (-6.95, 2.), (-6.9, 5.),
                 (-6., 5.05), (-18.2, 5.1), (-18.2, 19.9), (13.9, 19.9),
                 (13.9, 5.1), (-1., 5.05), (5., 5.), (5., -10.), (-6.9, -10.),
                 (-6.95, .5), (-7., -1.5)]

    # inner - points on inner boundary
    if tag[: 5] == "wedge" or tag == "cavity":
        inner = []
    elif tag[: 4] == "room":
        centers = [(2., -8.), (-1.5, -8.), (-3., -3.5)]
        scales = [.75, 1.25, 1.]
        angles = [np.pi / 7, 6 * np.pi / 13, 3 * np.pi / 5]
        inner = []
        for c, s, a in zip(centers, scales, angles):
            inner += [[(c[0] + s * np.cos(a + 2 * j * np.pi / 3),
                        c[1] + s * np.sin(a + 2 * j * np.pi / 3)
                        ) for j in range(3)]]

    # bcs - type of boundary
    if tag[: 5] == "wedge":
        bcs = [1.] * len(outer)
    elif tag == "cavity":
        bcs = [0.] * len(outer)
    else:#if tag[: 4] == "room":
        bcs = [1.] * len(outer) + [0.] * sum([len(i) for i in inner])

    # x, y - x and y grids of evaluation points
    if tag[: 5] == "wedge":
        if tag == "wedge_1" or tag == "wedge_2":
            x, y = np.linspace(-4., 4., 200), np.linspace(-5.5, 2.5, 200)
        if tag == "wedge_3":
            x, y = np.linspace(-6., 2., 200), np.linspace(-1.5, 6.5, 200)
        if tag == "wedge_4":
            x, y = np.linspace(-2., 6., 200), np.linspace(-5.5, 2.5, 200)
    elif tag == "cavity":
        x, y = np.linspace(-5., 3., 200), np.linspace(-3., 9., 300)
    else:#if tag[: 4] == "room":
        x, y = np.linspace(-10., 6., 250), np.linspace(-10., 6., 250)

    # cutoff - tolerance for cutoff
    if tag[: 5] == "wedge" or tag == "cavity":
        cutoff = -1
    elif tag == "room_tol1e-3":
        cutoff = 1e-3
    else:
        cutoff = 2.5e-2

    # add_forcing - function to enforce forcing term
    if tag[: 5] == "wedge" or tag == "cavity" or tag[: 8] =="room_tol":
        add_forcing = None
    else:# if tag[: 13] == "room_harmonic":
        from fem import generate_snapshots_polar
        mass = generate_snapshots_polar(mesh, ts[: 2], 0., 0.,
                                        return_matrices = 1)[2]
        if tag == "room_harmonic_1":
            omega_f = 2 * np.pi
        if tag == "room_harmonic_5":
            omega_f = 10 * np.pi
        def add_forcing(u_, t_):
            if np.abs(t_ - dt) < 1e-10:
                u_[0] = 2 * mass[0] * (np.sin(omega_f * (t_))
                                     - np.sin(omega_f * (t_ - dt))
                                       ) / dt ** 2
            else:
                u_[0] = mass[0] * (np.sin(omega_f * (t_ - 2 * dt))
                                 - 2 * np.sin(omega_f * (t_ - dt))
                                 + np.sin(omega_f * (t_))) / dt ** 2
            return u_
    ks_diffraction = 10
    return (t_max, R, u0, u1, outer, inner, bcs, x, y,
            mesh, ts, cutoff, add_forcing, ks_diffraction)

def load_example_fem(tag):
    (t_max, R, _, _, outer, inner,
     bcs, x, y, _, _, cutoff, _, _) = load_example_rom(tag)

    # Vspace - FEniCS FEM function space
    if tag[: 5] == "wedge" or tag == "cavity":
        n_mesh = 1000
    elif tag[: 8] =="room_tol":
        n_mesh = 960
    elif tag == "room_harmonic_1":
        n_mesh = 400
    else:#if tag == "room_harmonic_5":
        n_mesh = 2000
    import mshr, fenics as fen
    domain = mshr.Polygon([fen.Point(*x) for x in outer[::-1]])
    for inn in inner:
        domain = domain - mshr.Polygon([fen.Point(*x) for x in inn])
    mesh = mshr.generate_mesh(domain, n_mesh)
    Vspace = fen.FunctionSpace(mesh, "P", 1)

    # ts - t-grid for timestepping
    n_mesh_t = int(np.round(2 * t_max / mesh.hmin()))
    ts = np.linspace(0., t_max, n_mesh_t + 1)
    dt = ts[1]

    # u0 - initial condition
    if tag[: 5] == "wedge" or tag == "cavity":
        u0 = lambda x: np.exp(-.5 * ((x[:, 0] / .2) ** 2.
                                   + (x[:, 1] / .2) ** 2.))
    elif tag[: 8] =="room_tol":
        u0 = lambda x: np.exp(-.5 * ((x[:, 0] / .25) ** 2.
                                   + (x[:, 1] / .25) ** 2.)
                              ) * (1 - (x[:, 0] / .25) ** 2.
                                     - (x[:, 1] / .25) ** 2.)
    else:#if tag[: 13] == "room_harmonic":
        u0 = lambda x: np.zeros(x.shape[0])

    # u1 - initial velocity
    u1 = lambda x: np.zeros_like(x.shape[0])

    # where_dirichlet - identifier for dirichlet boundary
    if tag[: 5] == "wedge":
        where_dirichlet = lambda x, on_b: False
    elif tag == "cavity":
        where_dirichlet = lambda x, on_b: on_b
    else:#if tag[: 4] == "room":
        where_dirichlet = lambda x, on_b: (on_b and x[0] > -5. and x[0] < 4.
                                                and x[1] > -9. and x[1] < 0.)

    # add_forcing - function to enforce forcing term
    if tag[: 5] == "wedge" or tag == "cavity" or tag[: 8] =="room_tol":
        add_forcing = None
    else:# if tag[: 13] == "room_harmonic":
        from fem import generate_snapshots_mesh
        mass = generate_snapshots_mesh(Vspace, ts[: 2], 0., 0.,
                                       where_dirichlet, return_matrices = 1)[2]
        if tag == "room_harmonic_1":
            omega_f = 2 * np.pi
        if tag == "room_harmonic_5":
            omega_f = 10 * np.pi
        xy = Vspace.tabulate_dof_coordinates()
        idx_f = np.argmin(np.linalg.norm(xy, axis = 1))
        def add_forcing(u_, t_):
            if np.abs(t_ - dt) < 1e-10:
                u_[idx_f] = 2 * mass[idx_f] * (np.sin(omega_f * (t_))
                                             - np.sin(omega_f * (t_ - dt))
                                               ) / dt ** 2
            else:
                u_[idx_f] = mass[idx_f] * (np.sin(omega_f * (t_ - 2 * dt))
                                         - 2 * np.sin(omega_f * (t_ - dt))
                                         + np.sin(omega_f * (t_))) / dt ** 2
            return u_
    
    return (t_max, R, u0, u1, Vspace, where_dirichlet,
            x, y, ts, cutoff, add_forcing)

