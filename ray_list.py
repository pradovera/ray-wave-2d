from dataclasses import dataclass
from typing import Tuple, List

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

