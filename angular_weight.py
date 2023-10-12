from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
from domain import maptoangle, EPS

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
class constant_weight:
    phi_0: float # starting angle
    dphi: float # interior angle width
    clockwise: bool # whether to move clockwise
    scaling: float # scalar scaling factor

    def __call__(self, phi):
        phi = maptoangle((phi - self.phi_0) * (1 - 2 * self.clockwise), 0.)
        return self.scaling * (phi < self.dphi)
    
    @property
    def maxabs(self): return np.abs(self.scaling)

constant_one = constant_weight(0., 7, 0, 1.)

def fresnelScaled(x):
    # compute Fresnel integral using complex erf magic
    from scipy.special import erf
    return (np.pi * x) ** .5 * np.abs(erf((1 - 1j) * (.5 * x) ** .5) - 1)

@dataclass(frozen = True)
class utd_weight:
    phi_0: float # starting angle (angle of o face)
    dphi: float # interior angle width (angle of n face, measured from o face)
    clockwise: bool # whether to move clockwise
    scaling: float # scalar scaling factor
    phi_source: float # angular location of source point (measured from o face)
    bcs: List[float] # reflection coefficient at o and n faces
    ks: float # reference value of adimensional radial distance
    
    @property
    def nu(self): return np.pi / self.dphi
    
    def __call__(self, phi):
        phi = maptoangle((phi - self.phi_0) * (1 - 2 * self.clockwise), 0.)
        D_coeff = np.zeros_like(phi)
        scaling_base = - self.scaling * self.nu * (8 * np.pi * self.ks) ** -.5
        
        # collect coefficients for in, io, rn, ro
        Ns = [np.round(.5 * self.nu), np.round(- .5 * self.nu),
              np.round(.5 * (1 + self.nu)), np.round(.5 * (1 - self.nu))]
        beta_fre = ([phi - self.phi_source] * 2
                  + [phi + self.phi_source] * 2)
        beta_cot = [np.pi + phi - self.phi_source, np.pi - phi + self.phi_source,
                    np.pi + phi + self.phi_source, np.pi - phi - self.phi_source]
        signs = [1] * 2 + self.bcs[::-1]
        for j in range(4):
            theta_cot = .5 * self.nu * beta_cot[j]
            S, C = np.sin(theta_cot), np.cos(theta_cot)
            good_S = np.abs(S) > 1e-10
            bad_S = np.logical_not(good_S)
            theta_fre = Ns[j] * self.dphi - .5 * beta_fre[j][good_S]
            D_coeff[good_S] += (signs[j] * scaling_base * C[good_S] / S[good_S]
                              * fresnelScaled(2 * self.ks * np.cos(theta_fre) ** 2))
            # avoid risk of cancellation
            D_coeff[bad_S] += (- .5 * signs[j] * self.scaling
                               * np.sign(S[bad_S] * C[bad_S]))
        return D_coeff
    
    @property
    def maxabs(self):
        xtran = [self.phi_source - np.pi,
                 self.phi_source + np.pi,
                 np.pi - self.phi_source,
                 2 * self.dphi - self.phi_source - np.pi]
        xcheck = [x - EPS for x in xtran] + [x + EPS for x in xtran]
        xcheck = ([(1 - 2 * self.clockwise) * x for x in xcheck
                                                    if x > 0 and x < self.dphi]
                + [0, (1 - 2 * self.clockwise) * self.dphi])
        return np.max([np.abs(self(self.phi_0 + x)) for x in xcheck])

def build_angular_weight_reflect(D, ray, w_gen, ref_coeff):
    e_angle = D.get_edge_angle(ray.edge)
    phi_0_new = maptoangle(2 * e_angle - w_gen.phi_0)
    scaling_new = ref_coeff * w_gen.scaling
    if isinstance(w_gen, constant_weight):
        w_new = constant_weight(phi_0_new, w_gen.dphi, not w_gen.clockwise,
                                scaling_new)
    if isinstance(w_gen, utd_weight):
        w_new = utd_weight(phi_0_new, w_gen.dphi, not w_gen.clockwise,
                           scaling_new, w_gen.phi_source, w_gen.bcs[::-1],
                           w_gen.ks)
    return w_new

def build_angular_weight_diffract(D, ray, ray_gen, w_gen, ref_coeffs, ks):
    phi_0, dphi = D.get_point_interior_angle(ray.point)
    phi_source = maptoangle(np.angle((ray_gen.c_x[0] - ray.c_x[0])
                              + 1j * (ray_gen.c_x[1] - ray.c_x[1])) - phi_0, 0.)
    scaling = w_gen(phi_source + phi_0 + np.pi) # opposite angle
    if min(np.abs(phi_source), np.abs(dphi - phi_source)) < 1e-3: # grazing incidence
        scaling *= .5
    return utd_weight(phi_0, dphi, 0, scaling, phi_source, ref_coeffs, ks)

def build_angular_weights(D, rays, ks):
    # D is domain (from polygon.domain class)
    # rays is list of rays (output of timetable.build_ray_sequence)
    # ref_coeffs is list of reflection coefficients of edges
    weights = []
    # for ray in rays:
    for i, ray in enumerate(rays):
        if isinstance(ray, ray_r): # reflected wave
            r_coeff = 2 * D.BCs(ray.edge) - 1 # -1 = Dirichlet, 1 = Neumann
            weight = build_angular_weight_reflect(D, ray, weights[ray.gen], r_coeff)
        elif isinstance(ray, ray_d): # diffracted wave
            r_coeffs = [2 * D.BCs(ray.ignore[0]) - 1, # -1 = Dirichlet, 1 = Neumann
                        2 * D.BCs(ray.ignore[len(ray.ignore) - 1]) - 1]
            weight = build_angular_weight_diffract(D, ray, rays[ray.gen],
                                                   weights[ray.gen], r_coeffs, ks)
        else: # source wave
            weight = constant_one
        weights += [weight]
    return weights
