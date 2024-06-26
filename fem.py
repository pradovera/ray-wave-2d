import numpy as np
import scipy.sparse as sp

def assemble_stiffmass_polar(n, dx):
    # Assemble stiffness and lumped matrices for order-2 FE
    diags = np.zeros([3, n])
    k_range = np.arange(.5, n - .5)
    diags[0, 1:] -= k_range
    diags[1, :-1] += k_range
    diags[0, :-1] -= k_range
    diags[2, 1:] += k_range
    stiff = sp.spdiags(diags.reshape(diags.shape[0], -1), [0, -1, 1],
                       n, n, format = "csr")
    mass = np.zeros(n)
    m_range_l = np.arange(1. / 3, n - 2. / 3) * dx
    m_range_r = np.arange(2. / 3, n - 1. / 3) * dx
    mass[:-1] += m_range_l
    mass[1:] += m_range_r
    mass *= dx / 2.
    # check for zeros on diagonal (can appear when holes are adjacent)
    mass[np.abs(mass) < 1e-14] = 1.
    massm1 = lambda x: x / mass
    return stiff, mass, massm1

def assemble_forcing_polar(n, dx, f):
    # Assemble RHS for order-2 FE
    b = np.zeros(n)
    n_quad = 50
    hdq = dx / n_quad

    grid_quad = hdq * (.5 + np.arange((n - 1) * n_quad)).reshape(n - 1, -1)
    try:
        rf_quad = grid_quad * f(grid_quad)
    except:
        rf_quad = np.array([[g * f(g) for g in gl] for gl in grid_quad])

    rising_quad = np.tile(np.linspace(.5 / n_quad, 1. - .5 / n_quad,
                                      n_quad).reshape(1, -1), (n - 1, 1))
    waning_quad = 1 - rising_quad
    b[:-1] += np.sum(rf_quad * waning_quad, axis = 1) * hdq
    b[1:] += np.sum(rf_quad * rising_quad, axis = 1) * hdq
    return b

def generate_snapshots_polar(x, t, U0, U1, add_forcing = None,
                             return_matrices = False, only_final = False):
    """
    Run full wave polar 1d HF simulation using an order-2 discretization scheme
    Input:
        x: 1d array of mesh nodes coordinates.
        t: 1d array of time instants.
        U0: vector of inital condition.
        U1: vector of inital velocity.
        add_forcing: lambda function that adds forcing term contribution within
            timestepping.
        return_matrices: whether to also return linear system matrices.
        only_final: whether to only output snapshot at final time.
    Output:
        U: output of timestepping_2.
        stiff: stiffness matrix (sparse).
        mass: mass matrix (diagonal).
    """
    stiff, mass, massm1 = assemble_stiffmass_polar(len(x), x[1] - x[0])
    U = timestepping_2(U0, U1, t[1] - t[0], len(t) - 1, stiff, massm1,
                       add_forcing, only_final)
    if return_matrices: return U, stiff, mass
    return U

def assemble_stiffmass_mesh(Vspace, where_dirichlet = None):
    import fenics as fen
    u = fen.TrialFunction(Vspace)
    v = fen.TestFunction(Vspace)
    k = fen.assemble(- fen.inner(fen.grad(u), fen.grad(v)) * fen.dx)
    m = fen.assemble(fen.inner(u, v) * fen.dx)
    if where_dirichlet is None: where_dirichlet = lambda x, on_b: False
    DBC = fen.DirichletBC(Vspace, 0., where_dirichlet)
    DBC.zero(k)
    DBC.apply(m)
    k_mat = fen.as_backend_type(k).mat()
    kr, kc, kv = k_mat.getValuesCSR()
    stiff = sp.csr_matrix((kv, kc, kr), shape = k_mat.size)
    m_mat = fen.as_backend_type(m).mat()
    mr, mc, mv = m_mat.getValuesCSR()
    mass = sp.csr_matrix((mv, mc, mr), shape = m_mat.size)
    mass = mass @ np.ones(mass.shape[1])
    massm1 = lambda x: x / mass
    return stiff, mass, massm1

def assemble_forcing_mesh(Vspace, f_string, where_dirichlet = None):
    import fenics as fen
    f = fen.Expression(f_string, degree = 10)
    v = fen.TestFunction(Vspace)
    b = fen.assemble(fen.inner(f, v) * fen.dx)
    if where_dirichlet is None: where_dirichlet = lambda x, on_b: False
    DBC = fen.DirichletBC(Vspace, 0., where_dirichlet)
    DBC.apply(b)
    return np.array(b)

def generate_snapshots_mesh(Vspace, t, U0, U1, where_dirichlet = None,
                            add_forcing = None, return_matrices = False,
                            only_final = False):
    """
    Run full wave 2d HF simulation using an order-2 discretization scheme
    Input:
        Vspace: FEniCS function space.
        t: 1d array of time instants.
        U0: vector of inital condition.
        U1: vector of inital velocity.
        where_dirichlet: lambda function to label Dirichlet boundaries.
        add_forcing: lambda function that adds forcing term contribution within
            timestepping.
        return_matrices: whether to also return linear system matrices.
        only_final: whether to only output snapshot at final time.
    Output:
        U: output of timestepping_2.
        stiff: stiffness matrix (sparse).
        mass: mass matrix (diagonal).
    """
    stiff, mass, massm1 = assemble_stiffmass_mesh(Vspace, where_dirichlet)
    U = timestepping_2(U0, U1, t[1] - t[0], len(t) - 1, stiff, massm1,
                       add_forcing, only_final)
    if return_matrices: return U, stiff, mass
    return U

def timestepping_2(U0, U1, dt, Nt, stiff, massm1, add_forcing = None,
                   only_final = False):
    """
    Perform order-2 timestepping for full wave HF simulation
    Input:
        U0: 1d array of initial condition on U.
        U1: 1d array of initial condition on time derivative of U.
        dt: time step.
        Nt: number of time steps.
        stiff: sparse stiffness matrix.
        massm1: lambda function encoding left multiplication by mass^-1.
        add_forcing: lambda function that adds forcing term contribution within
            timestepping.
        only_final: whether to only output snapshot at final time.
    Output:
        U: 2d snapshot matrix. If store and chunks are small enough, only last
            chunk is returned.
        store_list (optional): list of output filenames.
    """
    if add_forcing is None: add_forcing = lambda x, t: x

    if only_final:
        u_ = U0
    else:
        U = np.empty((stiff.shape[0], Nt + 1))
        U[:, 0] = U0
    
    if only_final:
        u = u_ + massm1(dt * U1 + .5 * dt ** 2. * add_forcing(stiff @ u_, dt))
        u__, u_ = u_, u
    else:
        U[:, 1] = U[:, 0] + massm1(dt * U1 + .5 * dt ** 2.
                                            * add_forcing(stiff @ U[:, 0], dt))
    
    for j in range(2, Nt + 1):
        if only_final:
            u =  2 * u_ - u__ + dt ** 2. * massm1(add_forcing(stiff @ u_, j * dt))
            u__, u_ = u_, u
        else:
            U[:, j] = (2 * U[:, j - 1] - U[:, j - 2]
                     + dt ** 2. * massm1(add_forcing(stiff @ U[:, j - 1], j * dt)))
    if only_final: return u_
    return U
