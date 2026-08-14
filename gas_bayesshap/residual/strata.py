from dataclasses import dataclass
@dataclass
class ResidualRecord:
 feature:int; stratum:int; coalition:int; direction:str; residual_value:float; iteration:int; random_seed:int
