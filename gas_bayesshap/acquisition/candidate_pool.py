from ..game.subsets import random_coalition
def generate(rng,m,size):return [random_coalition(rng,m) for _ in range(size)]
