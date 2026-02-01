from abc import abstractmethod
from math import sqrt, ceil
from scipy.stats import poisson, norm, uniform, expon, gamma
from numpy.typing import NDArray
from environments.envClass import *


class BaseDist:
    """Abstract class for methods on probability distributions"""
    def __int__(self):
        pass

    @abstractmethod
    def sample(self):
        raise NotImplementedError("Method `sample` has to be implemented in class inherited from BaseDist.")

    @abstractmethod
    def get_cum_demand(self, invLevel: int):
        raise NotImplementedError("Method `get_cum_demand` has to be implemented in class inherited from BaseDist.")

    @abstractmethod
    def get_opt_bs_level(self, criticalRatio: float, L: int):
        raise NotImplementedError("Method `get_opt_bs_level` has to be implemented in class inherited from BaseDist.")

    @abstractmethod
    def get_exp_cost(self, environment: Environment):
        raise NotImplementedError("Method `get_exp_cost` has to be implemented in class inherited from BaseDist.")

    def __repr__(self):
        return str(self.__class__.__name__)


class Normal(BaseDist):
    """
    Normal distribution with parameters `mean` and `std_dev`
    """
    def __init__(self, mean: float, std_dev: float = 1.0, seed: int = 2):
        self.mean = mean
        self.std_dev = std_dev
        self.rng = np.random.default_rng(seed=seed)
        self.max_value = norm.ppf(q=0.99, loc=self.mean, scale=self.std_dev)
        np.random.seed(seed=seed)
        assert mean >= 0, f"Parameter `mean` of the Normal distribution must be non-negative but is {mean}."
        assert std_dev >= 0, f"Standard deviation of the Normal distribution must be non-negative but is {std_dev}."

    def __repr__(self) -> str:
        return "Normal"

    def sample(self) -> float:
        return float(max(0.0, min(self.rng.normal(loc=self.mean, scale=self.std_dev), self.max_value)))

    def get_cum_demand(self, invLevel: int) -> float:
        return float(norm.cdf(x=invLevel, loc=self.mean, scale=self.std_dev))

    def get_opt_bs_level(self, criticalRatio: float, L: int) -> float:
        mu_L = self.mean * (L + 1)
        sigma_L = self.std_dev * sqrt(L + 1)
        z = float(norm.ppf(criticalRatio))
        return mu_L + z * sigma_L
    
    def get_exp_cost(self, environment) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        L = environment.L
        num_steps = 1000
        demands = np.array([self.sample() for _ in range(num_steps)], dtype=np.float64)
        costs, pseudo_costs = get_cost_all_arms(demands, environment.bslevels, L, environment.model, environment.b, environment.h)
            
        return costs, pseudo_costs
    

class Uniform(BaseDist):
    """
    Continuous uniform distribution with parameters `mean` and `std_dev`
    """
    def __init__(self, low: float, scale: float, seed: int = 2):
        self.low = low
        self.scale = scale
        self.high = low + scale
        self.mean = (self.high + self.low) / 2
        self.std_dev = 1 / sqrt(12) * (self.high - self.low)
        # self.rng = np.random.default_rng(seed=seed)
        self.max_value = self.high
        np.random.seed(seed=seed)
        self.rng = np.random.default_rng(seed)
        assert low >= 0, f"Parameter `low` of the Uniform distribution must be non-negative but is {low}."
        assert scale >= 0, f"Parameter `scale` of the Uniform distribution must be non-negative but is {scale}."

    def __repr__(self) -> str:
        return "Uniform"

    def sample(self) -> float:
        return self.rng.uniform(low=self.low, high=self.high)

    def get_cum_demand(self, invLevel: int) -> float:
        return float(uniform.cdf(x=invLevel, loc=self.low, scale=self.scale))

    def get_opt_bs_level(self, criticalRatio: float, L: int):
        mu_L = (L + 1) * self.mean
        sigma_L = sqrt((L + 1) * (self.std_dev ** 2))

        z = norm.ppf(criticalRatio)
        return mu_L + z * sigma_L

    def get_exp_cost(self, environment: Environment) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        L = environment.L
        num_steps = 1000
        demands = np.array([self.sample() for _ in range(num_steps)], dtype=np.float64)
        costs, pseudo_costs = get_cost_all_arms(demands, environment.bslevels, L, environment.model, environment.b, environment.h)
        return costs, pseudo_costs


class Poisson(BaseDist):
    """
    Poisson distribution with parameter `mean`
    """
    def __init__(self, mean: float, seed: int = 2):
        self.mean = mean
        self.std_dev = sqrt(mean)
        self.max_value = ceil(poisson.ppf(q=0.99, mu=self.mean))
        np.random.seed(seed=seed)
        assert mean >= 0, f"Parameter `mean` of the Poisson distribution must be non-negative but is {mean}."

    def __repr__(self) -> str:
        return "Poisson"

    def sample(self) -> int:
        return poisson.rvs(mu=self.mean)

    def get_cum_demand(self, invLevel: int) -> int:
        return int(poisson.cdf(k=invLevel, mu=self.mean))

    def get_opt_bs_level(self, criticalRatio: float, L: int):
        opt_level = poisson.ppf(criticalRatio, mu=self.mean * (L + 1))
        return opt_level

    def get_exp_cost(self, environment: Environment) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        L = environment.L
        num_steps = 1000
        demands = np.array([self.sample() for _ in range(num_steps)], dtype=np.int32)
        costs, pseudo_costs = get_cost_all_arms(demands, environment.bslevels, L, environment.model, environment.b, environment.h)
        return costs, pseudo_costs
    

class Exponential(BaseDist):
    """
    Exponential distribution with parameter `mean`
    """
    def __init__(self, rate: float, seed: int = 2):
        assert rate > 0, f"Parameter `rate` of the Exponential distribution must be positive but is {rate}."
        self.rate = rate
        self.mean = 1/rate
        self.std_dev = 1/rate
        self.max_value = ceil(expon.ppf(q=0.99, scale=self.mean))
        np.random.seed(seed=seed)

    def __repr__(self):
        return "Exponential"

    def sample(self) -> float:
        return expon.rvs(scale=self.mean)

    def get_cum_demand(self, invLevel: float) -> float:
        return float(expon.cdf(x=invLevel, scale=self.mean))

    def get_opt_bs_level(self, criticalRatio: float, L: int):
        opt_level = gamma.ppf(criticalRatio, a=(L+1), scale=self.mean)
        return opt_level

    def get_exp_cost(self, environment: Environment) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        L = environment.L
        num_steps = 1000
        demands = np.array([self.sample() for _ in range(num_steps)], dtype=np.float64)
        costs, pseudo_costs = get_cost_all_arms(demands, environment.bslevels, L, environment.model, environment.b, environment.h)
        return costs, pseudo_costs


def get_cost_all_arms(
    demands: NDArray[np.float64] | NDArray[np.int32],
    bslevels: NDArray[np.float64] | NDArray[np.int32],
    lead_time: int,
    model: str,
    b: float,
    h: float,) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Monte-Carlo cost simulation for long-run costs of discrete set of base-stock policies.
    """
    T = len(demands)
    K = len(bslevels)

    on_hand = np.zeros(K)
    in_transit = np.zeros((K, lead_time))
    sum_inv = np.zeros(K)

    pseudo_cost = np.zeros(K)
    cost = np.zeros(K)

    for demand in demands:
        # inventory before demand
        if lead_time > 0:
            inventory_before = on_hand + in_transit[:, 0]

            # compute replenishment orders
            orders = np.maximum(bslevels - sum_inv, 0.0)

            # shift in-transit pipeline and insert new orders
            in_transit[:, 0] = orders
            in_transit[:] = np.roll(in_transit, shift=-1, axis=1)
        else:
            inventory_before = np.maximum(on_hand, bslevels)

        # demand fulfillment
        if model == "lost_sales":
            on_hand = np.maximum(0.0, inventory_before - demand)
        else:
            on_hand = inventory_before - demand

        # recompute total inventory
        if lead_time > 0:
            sum_inv = on_hand + np.sum(in_transit, axis=1)
        else:
            sum_inv = on_hand

        sales = np.minimum(inventory_before, demand)
        cost += h * (inventory_before - sales) + b * (demand - sales)
        pseudo_cost += h * (inventory_before - sales) - b * sales

    cost /= T
    pseudo_cost /= T
    return cost, pseudo_cost