from warnings import warn
import numpy as np
from matplotlib import pyplot as plt

PI_EPS = 3.
eps = 1e-12
EPS = 1e-8
EPS_ = 1e-6
EPS__ = 1e-3

def versor(t, reim = False):
    # Compute versor with phase t
    z = np.exp(1.j * t)
    if reim: return np.array([np.real(z), np.imag(z)])
    return z

def maptoangle(t, start = -np.pi):
    # Map angle using periodicity
    return t - 2 * np.pi * np.floor(.5 * (t - start) / np.pi)

def angleabstorel(t1, t2, return_idx = False):
    # Map angle interval to [start, size]
    dts = maptoangle(t2 - t1)
    if dts >= 0.:
        out = [t1, dts]
        if return_idx: out += [0]
    else:
        out = [t2, - dts]
        if return_idx: out += [1]
    return out

def compute_length_on_interval(i_t, i_r, j_t):
    # Take triangle ABC. Angle at A is i_t. Lengths of AB and AC are i_r[0] and
    #   i_r[1]. Find point D on line BC such that the angle BAD is j_t.
    #   This function computes the length of segment BD.
    side_CB = i_r[0] - i_r[1] * versor(i_t)
    angle_B = - np.angle(side_CB)
    angle_ADB = np.pi - j_t - angle_B
    # sin theorem says that AB / AD = sin(ADB) / sin(B)
    return i_r[0] * np.sin(angle_B) / np.sin(angle_ADB)

def segment_intersect_line(p1, q1, p2, q2):
    # check if segments from p1[j] to q1 intersect line through p2 and q2
    return np.sign(- ((q2[0] - p2[0]) * (p1[:, 1] - p2[1])
                    - (q2[1] - p2[1]) * (p1[:, 0] - p2[0]))
                   * ((q2[0] - p2[0]) * (q1[1] - p2[1])
                    - (q2[1] - p2[1]) * (q1[0] - p2[0]))).astype('int')

def segments_intersect(p1, q1, p2, q2):
    # check if segment from p1 to q1 intersets segment from p2 to q2
    # code from:
    #  https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        return 1 * (val > 0) + 2 * (val < 0)
    def onSegment(p, q, r):
        return ((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0]))
            and (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1])))
    # Find the 4 orientations required for the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    return (((o1 != o2) and (o3 != o4))
         or ((o1 == 0) and onSegment(p1, p2, q1))
         or ((o2 == 0) and onSegment(p1, q2, q1))
         or ((o3 == 0) and onSegment(p2, p1, q2))
         or ((o4 == 0) and onSegment(p2, q1, q2)))

class SegmentIntersectionException(Exception):
    pass

def interval_difference_polar(i_, j_, i_r_, j_r_):
    # i_ and j_ are intervals in [start, size] form
    # i_r_ and j_r_ are pairs of lengths
    # we assume no intersection between the intervals, except at endpoint
    if not i_ or not j_: return i_, i_r_
    i, j, i_r, j_r = [[*x] for x in [i_, j_, i_r_, j_r_]]
    def interval_difference_simple(ix_1, ix_2, jx_1, jx_2,
                                   ir_1, ir_2, jr_1, jr_2):
        ix_2 += ix_1
        jx_2 += jx_1
        error_msg = ("intersection between intervals ({}, {}) and ({}, {})"
                     "").format([ix_1, ix_2], [ir_1, ir_2],
                                [jx_1, jx_2], [jr_1, jr_2])
        if np.abs(jx_1 - ix_1) < EPS_: # jx_1 and ix_1 coincide
            jx_1 = ix_1
        elif np.abs(jx_1 - ix_2) < EPS_: # jx_1 and ix_2 coincide
            jx_1 = ix_2
        if np.abs(jx_2 - ix_1) < EPS_: # jx_2 and ix_1 coincide
            jx_2 = ix_1
        elif np.abs(jx_2 - ix_2) < EPS_: # jx_2 and ix_2 coincide
            jx_2 = ix_2

        if jx_2 < ix_1 + EPS_ or jx_1 > ix_2 - EPS_:
            # jx and ix are disjoint
            return [ix_1, ix_2 - ix_1], [ir_1, ir_2]

        if np.abs(jx_1 - ix_1) < EPS_ and np.abs(jx_2 - ix_2) < EPS_:
            # ix_1 = jx_1 and ix_2 = jx_2
            if ir_1 < jr_1 + eps and ir_2 < jr_2 + eps:
                return [ix_1, ix_2 - ix_1], [ir_1, ir_2]
            if jr_1 < ir_1 + eps and jr_2 < ir_2 + eps:
                return [], []
            raise SegmentIntersectionException(error_msg)

        # let {ix_1, ix_2, jx_1, jx_2} be sorted as a, b, c, d
        # Note: to be here, {ix_1, jx_1} are {a, b} and {ix_2, jx_2} are {c, d}
        # we find the projected distance of b and c on the opposite intervals
        if jx_1 < ix_1 - EPS_: # b is ix_1
            delta_r_1 = compute_length_on_interval(jx_2 - jx_1, [jr_1, jr_2],
                                                   ix_1 - jx_1) - ir_1
        elif ix_1 < jx_1 - EPS_: # b is jx_1
            delta_r_1 = jr_1 - compute_length_on_interval(ix_2 - ix_1,
                                                          [ir_1, ir_2],
                                                          jx_1 - ix_1)
        else: # ix_1 = jx_1
            delta_r_1 = 0
        if jx_2 > ix_2 + EPS_: # c is ix_2
            delta_r_2 = compute_length_on_interval(jx_2 - jx_1, [jr_1, jr_2],
                                                   ix_2 - jx_1) - ir_2
        elif ix_2 > jx_2 + EPS_: # c is jx_2
            delta_r_2 = jr_2 - compute_length_on_interval(ix_2 - ix_1,
                                                          [ir_1, ir_2],
                                                          jx_2 - ix_1)
        else: # ix_2 = jx_2
            delta_r_2 = 0

        if delta_r_1 > - eps and delta_r_2 > - eps:
            # interval j is far
            return [ix_1, ix_2 - ix_1], [ir_1, ir_2]

        if not (delta_r_1 < eps and delta_r_2 < eps):
            raise SegmentIntersectionException(error_msg)

        # interval j is close
        # clip a window out of ix
        if ix_1 < jx_1 - EPS_ and ix_2 > jx_2 + EPS_:
            # b is jx_1 and c is ix_2
            return ([ix_1, jx_1 - ix_1, jx_2, ix_2 - jx_2],
                  [ir_1, jr_1 - delta_r_1, jr_2 - delta_r_2, ir_2])

        # clip the left part of ix
        if ix_2 > jx_2 + EPS_:
            # (b is ix_1 or ix_1 = jx_1) and c is jx_2
            return [jx_2, ix_2 - jx_2], [jr_2 - delta_r_2, ir_2]

        # clip the right part of ix
        if ix_1 < jx_1 - EPS_:
            # b is jx_1 and (c is ix_2 or ix_2 = jx_2)
            return [ix_1, jx_1 - ix_1], [ir_1, jr_1 - delta_r_1]

        # clip everything
        return [], []

    for l in range(len(i) - 2, -2, -2): # loop over base intervals
        i_l, i_l_r = i[l : l + 2], i_r[l : l + 2]
        for k in range(0, len(j), 2): # loop over intervals to be subtracted
            j_k, j_k_r = j[k : k + 2], j_r[k : k + 2]
            j_k_ = j_k[0] + .5 * j_k[1]
            for ll in range(len(i_l) - 2, -2, -2): # loop over local interval
                i_ll, i_ll_r = i_l[ll : ll + 2], i_l_r[ll : ll + 2]
                i_ll_ = i_ll[0] + .5 * i_ll[1]
                if np.abs(i_ll_ - 2. * np.pi - j_k_) < np.abs(i_ll_ - j_k_):
                    i_ll[0] -= 2. * np.pi
                if np.abs(i_ll_ + 2. * np.pi - j_k_) < np.abs(i_ll_ - j_k_):
                    i_ll[0] += 2. * np.pi
                i_ll, i_ll_r = interval_difference_simple(*i_ll, *j_k,
                                                          *i_ll_r, *j_k_r)
                i_l = i_l[: ll] + i_ll + i_l[ll + 2 :]
                i_l_r = i_l_r[: ll] + i_ll_r + i_l_r[ll + 2 :]
        i = i[: l] + i_l + i[l + 2 :]
        i_r = i_r[: l] + i_l_r + i_r[l + 2 :]
    angle_ref = - np.pi
    for l in range(0, len(i), 2):
        i[l] = maptoangle(i[l], angle_ref)
        if not l: angle_ref = i[0]
    return i, i_r

class domain:
    _points, _edges, _adjacency, bcs = [], [], None, None
    
    @property
    def points(self):
        return self._points
    @points.setter
    def points(self, points):
        self._adjacency = None
        self._points = points

    @property
    def edges(self):
        return self._edges
    @edges.setter
    def edges(self, edges):
        self._adjacency = None
        self._edges = edges

    @property
    def adjacency(self):
        if self._adjacency is None:
            self._adjacency = np.zeros((self.npoints,) * 2, dtype = int)
            for e in self.edges:
                self._adjacency[e[0], e[1]] = 1
                self._adjacency[e[1], e[0]] = -1
        return self._adjacency

    @property
    def npoints(self): return len(self.points)

    @property
    def nedges(self): return len(self.edges)

    def BCs(self, j):
        # get boundary conditions for edge j
        if self.bcs is None or len(self.bcs) != self.nedges: return None
        return self.bcs[j]

    def edge_exists(self, i, j):
        # check if edge [i, j] or [j, i] exists
        return self.adjacency[i, j] != 0

    def get_point_interior_angle(self, i):
        # get interior angle at point i
        # works only if i is endpoint of exactly two edges, with correct
        #   orientation
        idx_i = np.where(self.adjacency[i, :] == -1)[0]
        idx_o = np.where(self.adjacency[i, :] == 1)[0]
        if len(idx_i) == 0:
            raise Exception("no incoming edge at selected point")
        if len(idx_i) > 1:
            raise Exception("too many incoming edges at selected point")
        if len(idx_o) == 0:
            raise Exception("no outgoing edge at selected point")
        if len(idx_o) > 1:
            raise Exception("too many outgoing edges at selected point")
        e_i = ((self.points[idx_i[0]][0] - self.points[i][0])
             + 1j * (self.points[idx_i[0]][1] - self.points[i][1]))
        t_i = np.angle(e_i)
        dt_io = np.angle(((self.points[idx_o[0]][0] - self.points[i][0])
                        + 1j * (self.points[idx_o[0]][1] - self.points[i][1])
                          ) / e_i)
        return [t_i, maptoangle(dt_io, 0.)]

    def get_point_exterior_angle(self, i):
        # get exterior angle at point i
        # works only if i is endpoint of exactly two edges, with correct
        #   orientation
        [t_i, dt_io] = self.get_point_interior_angle(i)
        return [maptoangle(t_i + dt_io), 2 * np.pi - dt_io]

    def get_edge_length(self, j):
        # get length of edge j
        return np.abs(((self.points[self.edges[j][1]][0]
                      - self.points[self.edges[j][0]][0])
                     + 1j * ((self.points[self.edges[j][1]][1]
                            - self.points[self.edges[j][0]][1]))))

    def get_edge_angle(self, j):
        # get angle of edge j
        return np.angle(((self.points[self.edges[j][1]][0]
                        - self.points[self.edges[j][0]][0])
                       + 1j * ((self.points[self.edges[j][1]][1]
                              - self.points[self.edges[j][0]][1]))))

    def plot(self):
        # make a simple plot of the domain
        plt.figure(figsize = (5, 4))
        for e in self.edges:
            plt.plot([self.points[i][0] for i in e],
                     [self.points[i][1] for i in e])
        for p in self.points: plt.plot(*p, 'o')
        plt.show()

    def get_symmetric_point_off_edge(self, x, j_e):
        # perform symmetrizations of point x off edge j_e
        y0 = [self.points[self.edges[j_e][0]][0] - x[0],
              self.points[self.edges[j_e][0]][1] - x[1]]
        n = [self.points[self.edges[j_e][0]][1]
           - self.points[self.edges[j_e][1]][1],
             self.points[self.edges[j_e][1]][0]
           - self.points[self.edges[j_e][0]][0]]
        proj_coeff = 2 * (y0[0] * n[0] + y0[1] * n[1]) / (n[0] ** 2.
                                                        + n[1] ** 2.)
        return (x[0] + proj_coeff * n[0], x[1] + proj_coeff * n[1])

    def get_symmetric_angle_off_edge(self, x, t, j_e):
        # perform symmetrizations of angle t on point x off edge j_e
        if not t: return []
        y01 = ((self.points[self.edges[j_e][1]][0]
              - self.points[self.edges[j_e][0]][0])
             + 1j * (self.points[self.edges[j_e][1]][1]
                   - self.points[self.edges[j_e][0]][1]))
        a = np.angle(y01)
        out, ref_angle = [], - np.pi
        for k in range(len(t) - 2, -2, -2):
            angle_k = maptoangle(2 * a - t[k] - t[k + 1], ref_angle)
            if k == len(t) - 2: ref_angle = angle_k
            out += [angle_k, t[k + 1]]
        return out

    def is_diffraction_point(self, x, i):
        # check if point i is a potential diffraction point for wave from x
        #   returns [0] if not, i.e., convex with no shadow zones.
        #   returns [1, a1, a2] if concave with two reflect zones and no shadow
        #                       zones, where a1 and a2 are reflection angles
        #                       (if collapse_concave_diffraction, it is a
        #                        double diffraction point)
        #                       (if not collapse_concave_diffraction, it is not
        #                        a diffraction point).
        #   returns [2, a1, a2] if concave with one reflect zone and one shadow
        #                       zone, where a1 is reflection angle and a2 is sz
        #                       angle
        #                       (if collapse_concave_diffraction, it is a
        #                        double diffraction point)
        #                       (if not collapse_concave_diffraction, it is
        #                        a single diffraction point at angle a2).
        #   returns [3, a1] if convex with one shadow zone, where a1 is sz
        #                   angle and incident ray comes from outside
        #                   (if collapse_concave_diffraction, it is a simple
        #                    diffraction point only if not included in a prior
        #                    diffraction of type 1 or 2)
        #                   (if not collapse_concave_diffraction, it is a
        #                    simple diffraction point).
        #   returns [-3, a1] if convex with one shadow zone, where a1 is sz
        #                    angle and incident ray comes from outside, but
        #                    BCs of two support edges are equal and angle is
        #                    pi/n, with n an integer > 1
        #                   (if collapse_concave_diffraction, it is a simple
        #                    diffraction point only if not included in a prior
        #                    diffraction of type 1 or 2)
        #                   (if not collapse_concave_diffraction, it is a
        #                    simple diffraction point).
        #   all angles are counter-clockwise from in-edge.
        # this does not take into account shadow regions cast by edges
        t_i, dt_int = self.get_point_interior_angle(i)
        z = (x[0] - self.points[i][0]) + 1j * (x[1] - self.points[i][1])
        if dt_int < np.pi: # domain is locally convex
            idx_i = np.where(self.adjacency[i, :] == -1)[0]
            idx_o = np.where(self.adjacency[i, :] == 1)[0]
            if len(idx_i) == 0:
                raise Exception("no incoming edge at selected point")
            if len(idx_i) > 1:
                raise Exception("too many incoming edges at selected point")
            if len(idx_o) == 0:
                raise Exception("no outgoing edge at selected point")
            if len(idx_o) > 1:
                raise Exception("too many outgoing edges at selected point")
            e = - versor(np.angle(z))
            angle_i = np.angle(((self.points[idx_i[0]][0] - self.points[i][0])
                         + 1j * (self.points[idx_i[0]][1] - self.points[i][1])
                                ) / e)
            angle_o = np.angle(((self.points[idx_o[0]][0] - self.points[i][0])
                         + 1j * (self.points[idx_o[0]][1] - self.points[i][1])
                                ) / e)
            if angle_i < - EPS and angle_o > EPS:
                n_int = np.pi / dt_int
                for j, e in enumerate(self.edges):
                    if i == e[1]: e_i = j
                    if i == e[0]: e_o = j
                bc_i, bc_o = self.BCs(e_i), self.BCs(e_o)
                if (bc_i is not None and bc_i == bc_o
                and np.abs(n_int - np.round(n_int)) < EPS):
                    return [-3, - angle_i] # interior shadow zone
                return [3, - angle_i] # interior shadow zone
            return [0] # no interior shadow zone
        e_i = versor(t_i)
        angle_x = maptoangle(np.angle(z / e_i), 0.)
        if angle_x > 2 * np.pi - EPS: angle_x = EPS
        if angle_x < dt_int - np.pi: # e_i is lit, e_o is hidden
            return [2, np.pi - angle_x, np.pi + angle_x]
        elif angle_x > np.pi: # e_o is lit, e_i is hidden
            return [2, 2 * dt_int - angle_x - np.pi, angle_x - np.pi]
        else: # both e_i and e_o are lit
            return [1, np.pi - angle_x, 2 * dt_int - angle_x - np.pi]

    def get_shortest_points(self, x):
        # get polar coordinates of shortest path from x to all points
        # this ignores shadow regions cast by edges
        rs, ts = [], []
        for p in self.points:
            y = (p[0] - x[0]) + 1j * (p[1] - x[1])
            rs += [np.abs(y)]
            ts += [np.angle(y)]
        return rs, ts

    def _apply_force_mask(self, rs, ts, ft_edge = None, ft_angle = None,
                          ignore = None):
        if ignore is None: ignore = []
        ignore = list(ignore)
        if ft_edge is None and ft_angle is None:
            return ignore, None, None # should not be used for this!
        if ft_edge is not None:
            # get support on edge
            v_start, v_end = self.edges[ft_edge]
            ft_start, ft_width, switch = angleabstorel(ts[v_start],
                                                       ts[v_end], 1)
            if switch:
                rs_supp = [rs[v_end], rs[v_start]]
            else:
                rs_supp = [rs[v_start], rs[v_end]]
        if ft_angle is None:
            # only support on edge
            ts_ft = [ft_start, ft_width]
            rs_ft = rs_supp
        else:
            # also support on angle
            ts_ft = list(ft_angle)
            j = 0
            if ft_edge is None:
                ft_start = ts_ft[0]
                ft_width = ts_ft[-2] + ts_ft[-1] - ts_ft[0]
                if ft_width > PI_EPS:
                    warn(("ft edge width is fairly large: it might lead to "
                          "unstable results"))
                rs_supp = [EPS, EPS]
            else:
                # intersect support on edge and on angle
                while j < len(ts_ft):
                    dts_j = maptoangle(ts_ft[j] - ft_start)
                    if dts_j < 0.: # ts_ft[j] is too far left
                        dts_jp = maptoangle(ts_ft[j] + ts_ft[j + 1] - ft_start)
                        if dts_jp < 0.: # ts_ft[j+] is too far left
                            ts_ft.pop(j + 1)
                            ts_ft.pop(j)
                            continue
                        ts_ft[j] = ft_start
                        ts_ft[j + 1] += dts_j
                    else: # ts_ft[j] is right enough
                        dts_j = maptoangle(ts_ft[j] - ft_start - ft_width)
                        if dts_j > 0.: # ts_ft[j] is too far right
                            ts_ft.pop(j + 1)
                            ts_ft.pop(j)
                            continue
                        else: # ts_ft[j] is in range
                            dts_jp = maptoangle(ts_ft[j] + ts_ft[j + 1]
                                              - ft_start - ft_width)
                            if dts_jp > 0.: # ts_ft[j+] is too far right
                                ts_ft[j + 1] -= dts_jp
                    j += 2
            # figure out support lengths
            rs_ft = []
            for j in range(0, len(ts_ft), 2):
                r1 = compute_length_on_interval(ft_width, rs_supp,
                                                ts_ft[j] - ft_start)
                r2 = compute_length_on_interval(ft_width, rs_supp,
                                            ts_ft[j] + ts_ft[j + 1] - ft_start)
                rs_ft += [r1, r2]
        for j, e in enumerate(self.edges):
            ignore_j = 0
            # ignore any edge that one crosses to get to ft_edge
            if j != ft_edge:
                ts_e, dts_e, switch_e = angleabstorel(ts[e[0]], ts[e[1]], 1)
                rs_e = [rs[e[switch_e]], rs[e[1 - switch_e]]]
                try:
                    ts_j = interval_difference_polar(ts_ft, [ts_e, dts_e],
                                                     rs_ft, rs_e)[0]
                except SegmentIntersectionException as e:
                    warn(str(e))
                    ts_j = None
                # ignore_j is true if edge e shields the ft edge in any way
                ignore_j = (ts_j is None or len(ts_j) != len(ts_ft)
                          or np.any(np.abs(np.array(ts_j)
                                         - np.array(ts_ft)) > EPS__))
            if ignore_j and j not in ignore: ignore += [j]
        return ignore, rs_ft, ts_ft
    
    def _get_shield_from_force_mask(self, x, force_through_edge = None,
                                    force_through_angle = None,
                                    prevent_angle = None, ignore = None):
        rs, ts = self.get_shortest_points(x)
        shield_t, shield_r = [], []
        if prevent_angle:
            shield_t += [[prevent_angle[0] + EPS,
                          prevent_angle[0] + prevent_angle[1] - EPS]]
            shield_r += [[EPS] * 2]
            if shield_t[-1][1] - shield_t[-1][0] > PI_EPS:
                # if angle is too wide, split shield in 2
                shield_t += [[np.mean(shield_t[-1]), shield_t[-1][1]]]
                shield_t[-2][1] = shield_t[-1][0]
                shield_r += [[EPS] * 2]
        if force_through_edge is not None or force_through_angle is not None:
            ignore, rs_ft, ts_ft = self._apply_force_mask(rs, ts,
                                                          force_through_edge,
                                                          force_through_angle,
                                                          ignore)
            if force_through_edge not in ignore:
                ignore = ignore + [force_through_edge]
            t_mean = .5 * (ts_ft[0] + ts_ft[-2] + ts_ft[-1])
            shield_t += [[t_mean - np.pi, ts_ft[0]]]
            if shield_t[-1][1] - shield_t[-1][0] > PI_EPS:
                # if angle is too wide, split shield in 2
                shield_t += [[np.mean(shield_t[-1]), shield_t[-1][1]]]
                shield_t[-2][1] = shield_t[-1][0]
                shield_r += [[EPS] * 2]
            shield_r += [[EPS, rs_ft[0]]]
            for j in range(0, len(ts_ft) - 2, 2):
                shield_t += [[ts_ft[j] + ts_ft[j + 1], ts_ft[j + 2]]]
                shield_r += [[rs_ft[j + 1], rs_ft[j + 2]]]
            shield_t += [[ts_ft[-2] + ts_ft[-1], t_mean + np.pi]]
            shield_r += [[rs_ft[-1], EPS]]
            if shield_t[-1][1] - shield_t[-1][0] > PI_EPS:
                # if angle is too wide, split shield in 2
                shield_t += [[np.mean(shield_t[-1]), shield_t[-1][1]]]
                shield_t[-2][1] = shield_t[-1][0]
                shield_r += [[EPS] * 2]
        return rs, ts, shield_t, shield_r, ignore
    
    def get_effective_points(self, x, force_through_edge = None,
                             force_through_angle = None, prevent_angle = None,
                             ignore = None):
        # get polar coordinates of shortest path from x to all points
        # this takes into account shadow regions cast by edges
        # forces paths to go through force_through_edge, in the polar angle
        #   region specified by force_through_angle
        # prevents paths in the polar angle region specified by prevent_angle
        rs, ts, shield_t, shield_r, ignore = self._get_shield_from_force_mask(
                                                         x, force_through_edge,
                                                         force_through_angle,
                                                         prevent_angle, ignore)
        shield_ignore = [[]] * len(shield_t)
        for k, e in enumerate(self.edges):
            if ignore is None or k not in ignore:
                shield_t += [[ts[e[0]], ts[e[1]]]]
                shield_r += [[rs[e[0]], rs[e[1]]]]
                shield_ignore += [e]
        for j, p in enumerate(self.points):
            # first remove all points belonging to ignored edges
            for k, e in enumerate(self.edges):
                 if ignore is not None and k in ignore and j in e:
                    rs[j] = ts[j] = None
                    break
            else:
                p_x = tuple([p[j] - x[j] for j in range(len(x))])
                # then loop over shields
                for k in range(len(shield_ignore)):
                    if j not in shield_ignore[k]:
                        p_k1 = shield_r[k][0] * versor(shield_t[k][0], 1)
                        p_k2 = shield_r[k][1] * versor(shield_t[k][1], 1)
                        # path is unfeasible if it goes through shield
                        if segments_intersect((0.,) * len(x), p_x,
                                              p_k1, p_k2) != 0:
                            rs[j] = ts[j] = None
                            break
        return rs, ts

    def get_shortest_edges(self, x):
        # get polar coordinates of shortest path from x to all edges
        # this ignores shadow regions cast by edges
        rs, ts = [], []
        for e in self.edges:
            y0 = ((self.points[e[0]][0] - x[0])
                + 1j * (self.points[e[0]][1] - x[1]))
            n = ((self.points[e[0]][1] - self.points[e[1]][1])
               + 1j * (self.points[e[1]][0] - self.points[e[0]][0]))
            y = np.real(y0 * np.conj(n)) / np.abs(n) ** 2. * n
            rs += [np.abs(y)]
            ts += [np.angle(y)]
        return rs, ts

    def get_range_edges(self, x, force_through_edge = None,
                        force_through_angle = None, prevent_angle = None,
                        ignore = None):
        # get polar coordinate feasibility ranges of shortest paths from x to
        #   all edges
        # this takes into account shadow regions cast by edges
        # forces paths to go through force_through_edge, in the polar angle
        #   region specified by force_through_angle
        # prevents paths in the polar angle region specified by prevent_angle
        rs, ts, shield_t, shield_r, ignore = self._get_shield_from_force_mask(
                                                         x, force_through_edge,
                                                         force_through_angle,
                                                         prevent_angle, ignore)
        r_rs, r_ts = [], []
        for j, e in enumerate(self.edges):
            if ((ignore is not None and j in ignore)
             or (force_through_edge is not None and j == force_through_edge)):
                r_rs += [[]]
                r_ts += [[]]
            else:
                t_j, dt_j, switch_j = angleabstorel(ts[e[0]], ts[e[1]], 1)
                ts_j = [t_j, dt_j]
                rs_j = [rs[e[switch_j]], rs[e[1 - switch_j]]]
                for sh_t, sh_r in zip(shield_t, shield_r):
                    t_s, dt_s, switch_s = angleabstorel(* sh_t, 1)
                    s_t = [t_s, dt_s]
                    s_r = [sh_r[switch_s], sh_r[1 - switch_s]]
                    try:
                        ts_j, rs_j = interval_difference_polar(ts_j, s_t,
                                                               rs_j, s_r)
                    except SegmentIntersectionException as e:
                        warn(str(e))
                        ts_j, rs_j = [], []
                for k, f in enumerate(self.edges):
                    if (k != j and (ignore is None or k not in ignore)
                               and (force_through_edge is None
                                 or k != force_through_edge)):
                        ts_k, dts_k, switch_k = angleabstorel(ts[f[0]],
                                                              ts[f[1]], 1)
                        rs_k = [rs[f[switch_k]], rs[f[1 - switch_k]]]
                        try:
                            ts_j, rs_j = interval_difference_polar(ts_j,
                                                                 [ts_k, dts_k],
                                                                 rs_j, rs_k)
                        except SegmentIntersectionException as e:
                            warn(str(e))
                            ts_j, rs_j = [], []
                r_rs += [rs_j]
                r_ts += [ts_j]
        return r_rs, r_ts

    def get_effective_edges(self, x, force_through_edge = None,
                            force_through_angle = None, prevent_angle = None,
                            ignore = None):
        # get polar coordinates of shortest paths from x to all edges
        # this takes into account shadow regions cast by edges
        # forces paths to go through force_through_edge, in the polar angle
        #   region specified by force_through_angle
        # prevents paths in the polar angle region specified by prevent_angle
        s_rs, s_ts = self.get_shortest_edges(x)
        r_rs, r_ts = self.get_range_edges(x, force_through_edge,
                                          force_through_angle, prevent_angle,
                                          ignore)
        rs, ts = [], []
        for s_tj, s_rj, r_tj, r_rj in zip(s_ts, s_rs, r_ts, r_rs):
            if not r_tj:
                # no path is admissible
                rs += [None]
                ts += [None]
            else:
                # check if shortest path is admissible
                shortest = 0
                for k in range(0, len(r_tj), 2):
                    ts_k, dts_k = r_tj[k : k + 2]
                    s_jk = maptoangle(s_tj - ts_k)
                    if s_jk >= 0. and s_jk <= dts_k:
                        shortest = 1
                        break
                if shortest:
                    rs += [s_rj]
                    ts += [s_tj]
                else:
                    # find shortest path that is admissible
                    idx = np.argmin(r_rj)
                    rs += [r_rj[idx]]
                    if idx % 2:
                        ts += [r_tj[idx - 1] + r_tj[idx]]
                    else:
                        ts += [r_tj[idx]]
        return rs, ts

    def check_light_cone(self, x, x0, force_through_edge = None,
                         force_through_angle = None, prevent_angle = None,
                         ignore = None):
        # get light cone from x
        # this takes into account shadow regions cast by edges
        # forces paths to go through force_through_edge, in the polar angle
        #   region specified by force_through_angle
        # prevents paths in the polar angle region specified by prevent_angle
        r_rs, r_ts = self.get_range_edges(x0, force_through_edge,
                                          force_through_angle, prevent_angle,
                                          ignore)
        angle = np.angle((x[:, 0] - x0[0]) + 1j * (x[:, 1] - x0[1]))
        lit = np.zeros(len(x), dtype = bool)
        not_hidden = np.ones(len(x), dtype = bool)
        # first check that rays does not pass through non-force_through_edge
        for j in range(len(r_ts)):
            if not r_ts[j]: continue # empty support angle
            # find relevant angle interval
            for k in range(0, len(r_ts[j]), 2):
                r_ts_k, r_dts_k = r_ts[j][k : k + 2]
                if r_ts_k + r_dts_k > np.pi:
                    relevant = np.where(np.logical_or(r_ts_k <= angle,
                                     angle <= r_ts_k + r_dts_k - 2 * np.pi))[0]
                else:
                    relevant = np.where(np.logical_and(r_ts_k <= angle,
                                                 angle <= r_ts_k + r_dts_k))[0]
                i_relevant = segment_intersect_line(x[relevant], x0,
                                                 self.points[self.edges[j][0]],
                                                 self.points[self.edges[j][1]])
                lit[relevant] = True
                not_hidden[relevant[i_relevant >= 0]] = False
        not_hidden = np.logical_and(lit, not_hidden)
        if force_through_edge is None: return not_hidden
        # then check that rays pass through force_through_edge
        intersect = segment_intersect_line(x, x0,
                                self.points[self.edges[force_through_edge][0]],
                                self.points[self.edges[force_through_edge][1]])
        active = intersect >= 0
        return np.logical_and(not_hidden, active)
