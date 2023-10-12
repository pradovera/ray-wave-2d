import numpy as np
from domain import EPS
from angular_weight import (ray, ray_r, ray_d, ray_list, constant_one,
                            build_angular_weight_reflect,
                            build_angular_weight_diffract)

def build_ray_sequence(D, u0_c, u0_supp, t_max, geometrical_optics_only = False,
                       weights_cutoff = None, verbose = 0, ks_diffraction = 5.):
    # D is domain (from polygon.domain class)
    # u0_c is tuple with location of wave origin
    # u0_supp is half-width of initial wave support
    # t_max is time horizon
    # if geometrical_optics_only == True, ignore diffraction effects
    timetable = np.empty((0, D.nedges + D.npoints * (not geometrical_optics_only)))
    angle_supp_r = [] # list of reflection support angles
    rays = [] # list of rays
    c_x = u0_c # x-ray center
    c_t = 0. # t-ray center
    idx_i = idx_j = None # row and column indices of next timetable entry
    weights = []
    if weights_cutoff is None: weights_cutoff = lambda t: -1
    while c_t - u0_supp <= t_max:
        range_edges_args = {}
        s_t = max(0., c_t - u0_supp)
        ignore_ray_loc = False
        if idx_i is None: # first ray
            wave_kind = "BASE"
            ignore_e, ignore_p = [], []
            ray_loc = ray(c_x, c_t, s_t)
        elif idx_j < D.nedges: # reflection
            wave_kind = "REFLECTION"
            c_x = D.get_symmetric_point_off_edge(rays[idx_i].c_x, idx_j)
            c_t = rays[idx_i].c_t
            ignore_e, ignore_p = [idx_j], []
            angle_supp = angle_supp_r[idx_i][idx_j]
            ignore_ray_loc = np.mean(angle_supp[1 :: 2]) < EPS
            ray_loc = ray_r(c_x, c_t, s_t, idx_i, idx_j, ignore_e, angle_supp)
            range_edges_args["force_through_edge"] = idx_j
            range_edges_args["force_through_angle"] = angle_supp
        else: # if idx_j >= D.nedges: # diffraction
            wave_kind = "DIFFRACTION"
            idx_j -= D.nedges
            c_x = D.points[idx_j]
            ignore_e = [j for j in range(D.nedges) if idx_j in D.edges[j]]
            ignore_p = [idx_j]
            int_angle = D.get_point_interior_angle(idx_j)
            if int_angle[1] <= np.pi: # convex
                angle_supp, angle_ext = int_angle, None
                ignore_ray_loc = np.mean(angle_supp[1 :: 2]) < EPS
            else:
                angle_supp, angle_ext = None, D.get_point_exterior_angle(idx_j)
            range_edges_args["force_through_angle"] = angle_supp
            range_edges_args["prevent_angle"] = angle_ext
            ray_loc = ray_d(c_x, c_t, s_t, idx_i, idx_j, ignore_e, angle_supp,
                            angle_ext)
        if not ignore_ray_loc:
            if wave_kind == "BASE":
                weights_loc = constant_one
            elif wave_kind == "REFLECTION":
                # -1 = Dirichlet, 1 = Neumann
                r_coeff = 2 * D.BCs(ray_loc.edge) - 1
                weights_loc = build_angular_weight_reflect(D, ray_loc,
                                                           weights[ray_loc.gen],
                                                           r_coeff)
            elif wave_kind == "DIFFRACTION":
                # -1 = Dirichlet, 1 = Neumann
                r_coeffs = [2 * D.BCs(ray_loc.ignore[0]) - 1,
                            2 * D.BCs(ray_loc.ignore[len(ray_loc.ignore) - 1]) - 1]
                weights_loc = build_angular_weight_diffract(D, ray_loc,
                                                            rays[ray_loc.gen],
                                                            weights[ray_loc.gen],
                                                            r_coeffs,
                                                            ks_diffraction)
            ignore_ray_loc = weights_loc.maxabs < weights_cutoff(ray_loc.s_t)
        if not ignore_ray_loc:
            range_edges_args["x"] = c_x
            range_edges_args["ignore"] = ignore_e
            # add new ray
            if verbose:
                print("Adding ray #{} at time {}".format(len(rays),
                                                         ray_loc.s_t))
            rays += [ray_loc]
            weights += [weights_loc]
            # cast new ray to update angular supports
            ts_edges = D.get_range_edges(** range_edges_args)[1]
            angle_supp_r_loc = []
            for j, a in enumerate(ts_edges):
                angle_supp_r_loc += [D.get_symmetric_angle_off_edge(c_x, a, j)]
            angle_supp_r += [angle_supp_r_loc]
            # cast new ray to update timetable
            timetable = np.pad(timetable, [(0, 1), (0, 0)], "constant",
                               constant_values = np.inf)
            t_arrival_e = D.get_effective_edges(** range_edges_args)[0]
            for j, t in enumerate(t_arrival_e):
                if j not in ignore_e and t is not None:
                    timetable[-1, j] = c_t + t
            if not geometrical_optics_only:
                t_arrival_p = D.get_effective_points(** range_edges_args)[0]
                angle_p = D.get_shortest_points(c_x)[1]
                # for diffraction t_arrival_p excludes the points linked to
                #   the current support point by edges, preventing adjacent
                #   diffractions of diffractions
                # for reflection t_arrival_p excludes the points on the current
                #   support edge, preventing internal diffractions of
                #   reflections
                for e in ignore_e:
                    for j in D.edges[e]:
                        if wave_kind == "REFLECTION":
                            # check if vertex j is actually in angle range
                            angle_is_far = 1
                            for k in range(0, len(angle_supp), 2):
                                for th in [angle_supp[k],
                                           angle_supp[k] + angle_supp[k + 1]]:
                                    if np.abs(angle_p[j] - th) < 1e-6:
                                        angle_is_far = 0
                                        break
                                if not angle_is_far: break
                            if angle_is_far: continue
                        elif j == idx_j:
                            continue
                        t_arrival_p[j] = np.abs((D.points[j][0] - c_x[0])
                                         + 1j * (D.points[j][1] - c_x[1]))
                for j, t in enumerate(t_arrival_p):
                    if j in ignore_p or t is None: continue
                    t_j, dt_int = D.get_point_interior_angle(j)
                    n_int = np.pi / dt_int
                    if np.abs(n_int - np.round(n_int)) < 1e-6: # no diff from integer angle
                        continue
                    if wave_kind == "REFLECTION" and j in D.edges[idx_j]:
                        # if j is part of the edge supporting reflection,
                        #   its diffraction is already accounted for
                        continue
                    timetable[-1, D.npoints + j] = c_t + t
        # index of next collision
        idx = np.argmin(timetable)
        idx_i, idx_j = idx // timetable.shape[1], idx % timetable.shape[1]
        c_t = timetable[idx_i, idx_j]
        timetable[idx_i, idx_j] = np.inf
    rays = ray_list(rays)
    return rays, weights

def get_light_cone_args(ray):
    if isinstance(ray, ray_r): # reflected wave
        return {"ignore": ray.ignore, "force_through_edge": ray.edge,
                "force_through_angle": ray.angle}
    if isinstance(ray, ray_d): # diffraction wave
        return {"ignore": ray.ignore, "force_through_angle": ray.angle,
                "prevent_angle": ray.ext_angle}
    return {}
