from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
from domain import versor, maptoangle, EPS, EPS_, EPS__

@dataclass
class ray:
    c_x: Tuple[float] # x-location of ray center
    c_t: float # t-location of ray center
    s_t: float # start of light cone

@dataclass
class ray_r(ray):
    gen: int # index of generating ray
    edge: int # index of reflective surface
    ignore: List[int] # index of known edges to ignore
    angle: List[float] # angular support interval

@dataclass
class ray_d(ray):
    gen: int # index of generating ray
    point: int # index of diffraction point
    ignore: List[int] # index of known edges to ignore
    angle: List[float] # angular support interval
    ext_angle: List[float] # external angular support interval (to be ignored)
    diff: List[float] # angular information for diffraction

@dataclass(frozen = True)
class ray_list:
    rays: List[ray]
    
    def __getitem__(self, i:int): return self.rays[i]
        
    def __len__(self): return len(self.rays)
        
    def __iter__(self): return self.rays.__iter__()

    def __str__(self):
        def print_item(j):
            if not (isinstance(self.rays[j], ray_r)
                 or isinstance(self.rays[j], ray_d)):
                return "source"
            if isinstance(self.rays[j], ray_r):
                out = "R (e {}) ".format(self.rays[j].edge)
            else: #if isinstance(self.rays[j], ray_d):
                out = "D (p {}) ".format(self.rays[j].point)
            g = self.rays[j].gen
            out += "of [{}]    \t".format(g)
            return out + print_item(g)
        out = ""
        for i in range(len(self)):
            out += "[{}]  \t".format(i) + print_item(i) + "\n"
        return out

@dataclass(frozen = True)
class piecewise_linear_weights:
# weights are piecewise-linear functions stored as a tuple:
#    ([v1, v2, ..., vn], [p1, p2, ..., pn])
#    weight value at p is line segment value linking (vj, pj) and
#    (v{j+1}, p{j+1}) if pj <= p <= p{j+1}
#    (to account for periodicity, evaluation shifts p and all the pj's by
#       ( p1 + p_{end} ) / 2 - pi
#    (if p-list is empty and v-list has length 1, then the weight is constant)
    values: List[float]
    nodes: List[float]
    
    @property
    def S(self): return len(self.nodes)
    
    def __call__(self, p):
        if not self.S: return self.values[0] * np.ones_like(p)
        p = np.asarray(p)
        out = np.zeros_like(p)
        local_idx = np.abs(p - self.nodes[0] + .5 * EPS_) < EPS_
        if np.any(local_idx): # p slightly left of leftmost node
            out[local_idx] = out[local_idx] + self.values[0]
        for j in range(self.S):
            local_idx = p == self.nodes[j]
            if np.any(local_idx): # p at node
                half = 1. if j == 0 or j == self.S - 1 else .5
                out[local_idx] = out[local_idx] + half * self.values[j]
            if j < self.S - 1:
                local_idx = np.logical_and(p > self.nodes[j],
                                           p < self.nodes[j + 1])
                if np.any(local_idx): # p in-between nodes
                    out[local_idx] = out[local_idx] + (
                            (self.nodes[j + 1] - p[local_idx]) * self.values[j]
                          + (p[local_idx] - self.nodes[j]) * self.values[j + 1]
                                        ) / (self.nodes[j + 1] - self.nodes[j])
        local_idx = np.abs(p - self.nodes[-1] - .5 * EPS_) < EPS_
        if np.any(local_idx): # p slightly right of rightmost node
            out[local_idx] = out[local_idx] + self.values[-1]
        return out
    
    @property
    def shift(self):
        if len(self.nodes) == 0: return 0.
        return .5 * (self.nodes[0] + self.nodes[-1]) - np.pi

def build_angular_weight_reflect(D, ray, w_gen, ref_coeff):
    e_angle = D.get_edge_angle(ray.edge)
    vs_gen, ps_gen, ps_shift = w_gen.values, w_gen.nodes, w_gen.shift
    if len(ps_gen):
        # must define new piecewise-linear function nodes and values
        # -> start from support angles and subdivide if necessary
        vs, ps = [], []
        for k in range(0, len(ray.angle), 2):
            local_ps = [ray.angle[k], ray.angle[k] + ray.angle[k + 1]]
            local_vs = [None] * 2
            local_ps_sym = [maptoangle(2 * e_angle - local_ps[1])]
            local_ps_sym += [local_ps_sym[0] + ray.angle[k + 1]]
            lb = maptoangle(local_ps_sym[0], ps_shift)
            ub = local_ps_sym[1] + lb - local_ps_sym[0]
            for j in range(len(ps_gen)):
                # check if generating interval [j, j + 1] intersects local
                if j < len(ps_gen) - 1:
                    # try to reconstruct left value (right when reflected)
                    # and right value (left when reflected)
                    for idx, p in zip([-1, 0], [lb, ub]):
                        if local_vs[idx] is None:
                            if np.abs(p - ps_gen[j]) < EPS__:
                                # j must be 0 for being here
                                local_vs[idx] = vs_gen[j]
                            elif np.abs(p - ps_gen[j + 1]) < EPS__:
                                if j < len(ps_gen) - 2:
                                    local_vs[idx] = .5 * (vs_gen[j + 1]
                                                        + vs_gen[j + 2])
                                else:
                                    local_vs[idx] = vs_gen[j + 1]
                            elif (p > ps_gen[j] - EPS__
                              and p < ps_gen[j + 1] + EPS__):
                                local_vs[idx] = (
                                            (ps_gen[j + 1] - p) * vs_gen[j]
                                          + (p - ps_gen[j]) * vs_gen[j + 1]
                                            ) / (ps_gen[j + 1] - ps_gen[j])
                if lb < ps_gen[j] - EPS__ and ub > ps_gen[j] + EPS__:
                    # j-th generating point is inside local
                    ps_j = maptoangle(2 * e_angle - ps_gen[j],
                                      local_ps[0] - EPS)
                    vs_j = vs_gen[j]
                    local_ps = local_ps[: 1] + [ps_j] + local_ps[1 :]
                    local_vs = local_vs[: 1] + [vs_j] + local_vs[1 :]
            vs += local_vs
            ps += local_ps
    else:
        vs = list(vs_gen)
        ps = []
    # multiply by reflection coefficient of edge
    vs = [ref_coeff * v for v in vs]
    return vs, ps

def build_angular_weight_diffract_1(ray, r_i, r_o):
    if ray.angle is None: # concave
        beta = 2 * np.pi - ray.ext_angle[1] # wedge angle
    else: # convex
        beta = ray.angle[1] # wedge angle
    phi_3 = ray.diff[1]
    scale = - 1. + 2 * (ray.diff[0] == 3)
    v_in = ((r_o + 3) * (phi_3 - beta)
          / ((r_o + 3) * beta + (r_i - r_o) * phi_3)) * scale
    v_out = v_in + scale
    vs = [.5 * (r_i + 1) * v_in, v_in, v_out, .5 * (r_o + 1) * v_out]
    # in-angle, transition, out-angle
    ps = [0.] + [phi_3] * 2 + [beta]
    return vs, ps

def build_angular_weight_diffract_2(ray, r_i, r_o):
    dt_wedge = ray.ext_angle[1] # wedge angle
    beta = 2 * np.pi - dt_wedge
    phi_1, phi_2 = np.sort(ray.diff[1 :])
    is_really_grazing = np.abs(ray.diff[1] - ray.diff[2]) < EPS
    if ray.diff[0] == 1: # no shadow zone
        jump_i, jump_o = r_i, - r_o
    else:
        if ((not is_really_grazing and ray.diff[1] < ray.diff[2])
         or (is_really_grazing and np.abs(ray.diff[1] - np.pi) < EPS)):
            # shadow zone is second transition
            jump_i, jump_o = r_i, 1.
        else:#  ((not is_really_grazing and ray.diff[1] > ray.diff[2])
             #or (is_really_grazing and np.abs(ray.diff[1] - np.pi) >= EPS))
            # shadow zone is first transition
            jump_i, jump_o = - 1., - r_o
    v_in = (jump_i * (phi_1 - phi_2)) / (.5 * (r_i + 1) * phi_1 + phi_2)
    v_out = ((jump_o * (phi_2 - phi_1))
           / (.5 * (r_o + 1) * (beta - phi_2) + beta - phi_1))
    v_ti, v_to = v_in + jump_i, v_out - jump_o
    if is_really_grazing:
        # halve value since there is no reflection
        vs = [.25 * (r_i + 1) * v_in, .5 * v_in,
              .5 * v_out, .25 * (r_o + 1) * v_out]
        # in-angle, transition, out-angle
        ps = [0.] + [ray.diff[1]] * 2 + [2 * np.pi - dt_wedge]
    else:
        vs = [.5 * (r_i + 1) * v_in, v_in, v_ti,
              v_to, v_out, .5 * (r_o + 1) * v_out]
        # in-angle, closest transition, furthest trans, out-angle
        ps = ([0.] + [min(* ray.diff[1 :])] * 2
                   + [max(* ray.diff[1 :])] * 2 + [2 * np.pi - dt_wedge])
    return vs, ps

def build_angular_weight_diffract(ray, ray_gen, w_gen, r_i, r_o):
    if len(ray.diff) == 2: # 1-transition shadow zone
        vs, ps = build_angular_weight_diffract_1(ray, r_i, r_o)
    else: #if len(ray.diff) == 3: # 2-transition shadow zone
        vs, ps = build_angular_weight_diffract_2(ray, r_i, r_o)
    if ray.angle is None: # concave
        p_shift = ray.ext_angle[-2] + ray.ext_angle[-1]
    else: # convex
        p_shift = ray.angle[0]
    ps = [p + p_shift for p in ps]
    # rescale weights by value of generating wave at scattering point
    vs_source, ps_source = w_gen.values, w_gen.nodes
    if ps_source:
        e_i_negative = - versor(ps_source[0])
        t_source = np.angle(((ray.c_x[0] - ray_gen.c_x[0])
                      + 1j * (ray.c_x[1] - ray_gen.c_x[1])) / e_i_negative
                            ) + np.pi
        if t_source > 2 * np.pi - EPS: t_source = 0.
        ps_source = [p - ps_source[0] for p in ps_source]
        for j in range(len(vs_source)):
            if np.abs(t_source - ps_source[j]) < EPS:
                if j == 0 or j == len(vs_source) - 1:
                    scale = vs_source[j]
                else:
                    scale = .5 * (vs_source[j] + vs_source[j + 1])
                break
            if (j < len(vs_source) - 1
            and t_source > ps_source[j] - EPS
            and t_source < ps_source[j + 1] + EPS):
                scale = ((ps_source[j + 1] - t_source) * vs_source[j]
                       + (t_source - ps_source[j]) * vs_source[j + 1]
                         ) / (ps_source[j + 1] - ps_source[j])
                break
        else:
            scale = 0.
    else:
        scale = vs_source[0]
    vs = [v * scale for v in vs]
    return vs, ps

def build_angular_weights(D, rays):
    # D is domain (from polygon.domain class)
    # rays is list of rays (output of timetable.build_ray_sequence)
    # ref_coeffs is list of reflection coefficients of edges
    weights = []
    # for ray in rays:
    for i, ray in enumerate(rays):
        if isinstance(ray, ray_r): # reflected wave
            r_coeff = 2 * D.BCs(ray.edge) - 1 # -1 = Dirichlet, 1 = Neumann
            vs, ps = build_angular_weight_reflect(D, ray, weights[ray.gen],
                                                  r_coeff)
        elif isinstance(ray, ray_d): # diffracted wave
            # -1 = Dirichlet, 1 = Neumann
            r_coeff_i = 2 * D.BCs(ray.ignore[0]) - 1
            r_coeff_o = 2 * D.BCs(ray.ignore[len(ray.ignore) - 1]) - 1
            vs, ps = build_angular_weight_diffract(ray, rays[ray.gen],
                                                   weights[ray.gen],
                                                   r_coeff_i, r_coeff_o)
        else: # source wave
            vs, ps = [1.], []
        if ps:
            dp = maptoangle(ps[0]) - ps[0]
            if dp: ps = [p + dp for p in ps]
        # check if representation can be collapsed
        for j in range(len(ps) - 2, -2, -2):
            if np.abs(ps[j + 1] - ps[j]) < EPS:
                vs.pop(j + 1), ps.pop(j + 1)
                vs.pop(j), ps.pop(j)
        weights += [piecewise_linear_weights(vs, ps)]
    return weights
