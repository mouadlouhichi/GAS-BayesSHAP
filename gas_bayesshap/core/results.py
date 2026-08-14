from dataclasses import dataclass,asdict
@dataclass
class GASResult:
 shapley_values:list; surrogate_shapley:list; residual_shapley:list; raw_confidence_widths:list; certified_projected_widths:list; posterior_std:list; status:str; certificate_is_rigorous:bool; range_bound_is_heuristic:bool
 def asdict(self):return asdict(self)
