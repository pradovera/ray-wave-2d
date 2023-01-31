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

def generate_snapshots_polar(x, t, U0, U1, add_forcing = None,
                             return_matrices = False):
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
    Output:
        U: output of timestepping_2.
        stiff: stiffness matrix (sparse).
        mass: mass matrix (diagonal).
    """
    stiff, mass, massm1 = assemble_stiffmass_polar(len(x), x[1] - x[0])
    U = timestepping_2(U0, U1, t[1] - t[0], len(t) - 1, stiff, massm1,
                       add_forcing)
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

def generate_snapshots_mesh(Vspace, t, U0, U1, where_dirichlet = None,
                            add_forcing = None, return_matrices = False):
    stiff, mass, massm1 = assemble_stiffmass_mesh(Vspace, where_dirichlet)
    U = timestepping_2(U0, U1, t[1] - t[0], len(t) - 1, stiff, massm1,
                       add_forcing)
    if return_matrices: return U, stiff, mass
    return U

def timestepping_2(U0, U1, dt, Nt, stiff, massm1, add_forcing = None):
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
    Output:
        U: 2d snapshot matrix. If store and chunks are small enough, only last
            chunk is returned.
        store_list (optional): list of output filenames.
    """
    if add_forcing is None: add_forcing = lambda x, t: x

    U = np.empty((stiff.shape[0], Nt + 1))
    U[:, 0] = U0
    
    U[:, 1] = U[:, 0] + massm1(dt * U1 + .5 * dt ** 2.
                                            * add_forcing(stiff @ U[:, 0], dt))
    
    for j in range(2, Nt + 1):
        U[:, j] = (2 * U[:, j - 1] - U[:, j - 2]
                 + dt ** 2. * massm1(add_forcing(stiff @ U[:, j - 1], j * dt)))
    return U
