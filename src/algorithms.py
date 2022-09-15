import logging
from typing import List

import numpy as np
from numpy.random import default_rng
from scipy import optimize

from data import Arm, Instance, Observation

def compute_c_one_resource(instance : Instance) -> float:
    """
    Compute the constant c for ControlBudget for one resource.
    """

    delta_drift = np.min([np.abs(arm.mean_drifts[0]) for arm in instance.arms])
    c = 6 / delta_drift**2

    return c

def compute_gamma(
    instance : Instance,
    lp_res : optimize.OptimizeResult,
    D : List[List[float]],
    verbose : bool = False
) -> float:
    """
    Compute the constant gamma for ControlBudget.
    """

    delta_slack = np.min([np.abs(slack) for slack in lp_res.slack \
        if np.abs(slack) > 1e-8] + [np.inf])
    delta_support = np.min([x for x in lp_res.x if x > 1e-8])
    sigma_min = np.linalg.svd(D, compute_uv=False)[-1]
    gamma = (sigma_min * min(delta_slack, delta_support)) / (4 * instance.num_resources)

    if verbose:
        logging.info(f'delta_slack: {delta_slack}')
        logging.info(f'delta_support: {delta_support}')
        logging.info(f'sigma_min: {sigma_min}')
        logging.info(f'gamma: {gamma}')

    return gamma

def compute_support(
    res : optimize.OptimizeResult
) -> List[int]:
    """
    Compute the support based on the given LP solution.
    """

    return [i for i in range(len(res.x)) if res.x[i] > 1e-8]

def compute_binding_resources(
    res : optimize.OptimizeResult,
    num_resources : int
) -> List[int]:
    """
    Compute the set of binding resources based on the given LP solution.
    """

    return [j for j in range(num_resources) if np.abs(res.slack[j]) <= 1e-8]

def compute_D(
    instance : Instance,
    support : List[int],
    binding_resources : List[int]
) -> np.ndarray:
    """
    Compute the constraint matrix corresponding to the sum-to-one constraint and
    resource drift constraints for the binding resources.
    """

    D = np.array([[1 for _ in support]])
    if len(binding_resources) > 0:
        _D = np.array([[
            instance.arms[i].mean_drifts[j] for i in support
        ] for j in binding_resources])
        D = np.concatenate((D, _D), axis = 0)

    return D

def compute_c_multiple_resources(
    instance : Instance,
    res : optimize.OptimizeResult,
) -> float:
    """
    Compute the constant c for ControlBudget for multiple resources.
    """

    support = compute_support(res)
    binding_resources = compute_binding_resources(res, instance.num_resources)
    D = compute_D(instance, support, binding_resources)
    gamma = compute_gamma(instance, res, D)
    c = 6 / gamma**2

    return c

def solve_lp(
    instance : Instance,
) -> optimize.OptimizeResult:
    """
    Solve the LP for the given instance using the true expectations.
    """

    T = instance.T
    B = instance.B
    arms = instance.arms
    num_resources = instance.num_resources

    c = [-arm.mean_reward for arm in arms]
    A_ub = [[-arm.mean_drifts[j] for arm in arms] for j in range(num_resources)]
    b_ub = [B/T for _ in range(num_resources)]
    A_eq = [[1 for _ in range(len(arms))]]
    b_eq = [1]
    bounds = (0, 1)

    res = optimize.linprog(
        c = c,
        A_ub = A_ub,
        b_ub = b_ub,
        A_eq = A_eq,
        b_eq = b_eq,
        bounds = bounds
    )

    res.fun *= -1

    return res

def solve_modified_lp(
    instance : Instance,
    mode : str,
    modify_arm : int = -1,
    modify_resource : int = -1
) -> optimize.OptimizeResult:
    """
    Solve the modified LP for the given instance using the UCB or LCB estimates.
    If modify_arm is nonnegative, then the constraints are modified to ensure
    this arm is played with zero probability. If modify_resource is nonnegative,
    then the objective function is modified to include a penalty term for the
    leftover budget for this resource.
    """

    T = instance.T
    B = instance.B
    arms = instance.arms
    num_resources = instance.num_resources

    # Using the actual confidence radius doesn't work well in practice. For our
    # simple examples, simply using the empirical mean or a very small
    # confidence radius works.
    _rad = 0

    if modify_resource > -1:
        # This should really be empirical mean + drifts, but it is mean - drifts
        # because of how c is computed below as a result of linprog minimizing
        # the objective instead of maximizing.
        empirical_means = np.array([
            arm.empirical_reward - arm.empirical_drifts[modify_resource] \
                for arm in arms
        ])
        rads = np.array([np.sqrt(8 * np.log(T) / len(arm.obs_rewards)) \
            for arm in arms])
        # Using a fixed confidence radius as described above.
        rads = np.array([_rad for arm in arms])
        if mode == 'ucb':
            c = -1 * (empirical_means + rads)
        else:
            c = -1 * (empirical_means - rads)
    else:
        empirical_means = np.array([arm.empirical_reward for arm in arms])
        rads = np.array([np.sqrt(8 * np.log(T) / len(arm.obs_rewards)) \
            for arm in arms])
        # Using a fixed confidence radius as described above.
        rads = np.array([_rad for arm in arms])
        if mode == 'ucb':
            c = -1 * (empirical_means + rads)
        else:
            c = -1 * (empirical_means - rads)
    """
    if modify_resource > -1:
        c = [-arm.mean_reward + arm.mean_drifts[modify_resource] for arm in arms]
    else:
        c = [-arm.mean_reward for arm in arms]
    """

    empirical_means = np.array([[
        arm.empirical_drifts[j] for arm in arms
    ] for j in range(num_resources)])
    rads = np.array([[
        np.sqrt(8 * np.log(T) / len(arm.obs_rewards)) for arm in arms
    ] for j in range(num_resources)])
    rads = np.array([[
        _rad for arm in arms
    ] for j in range(num_resources)])
    if mode == 'ucb':
        A_ub = -1 * (empirical_means + rads)
    else:
        A_ub = -1 * (empirical_means - rads)
    """
    A_ub = [[-arm.mean_drifts[j] for arm in arms] for j in range(num_resources)]
    """
    b_ub = [B/T for _ in range(num_resources)]

    if modify_arm > -1:
        A_eq = [
            [1 for _ in range(len(arms))],
            [0 if i != modify_arm else 1 for i in range(len(arms))]
        ]
        b_eq = [1, 0]
    else:
        A_eq = [[1 for _ in range(len(arms))]]
        b_eq = [1]

    bounds = (0, 1)

    res = optimize.linprog(
        c = c,
        A_ub = A_ub,
        b_ub = b_ub,
        A_eq = A_eq,
        b_eq = b_eq,
        bounds = bounds,
        options = {'presolve' : True}
    )

    res.fun *= -1

    if modify_resource > -1:
        res.fun += B/T

    return res

def sample_observation(
    arm : Arm
) -> Observation:
    """
    Sample an observation from the given arm.
    """

    rng = default_rng()
    reward = rng.binomial(1, arm.mean_reward)
    drifts = [
        rng.binomial(1, np.abs(mean_drift)) * np.sign(mean_drift) \
            for mean_drift in arm.mean_drifts
    ]
    return Observation(reward, drifts)

def update_budget(
    cur_budget : float,
    drift : float,
) -> float:
    """
    Update the current budget.
    """

    return cur_budget + drift

def controlbudget_one_resource(
    instance : Instance,
    c : float
) -> List[float]:
    """
    Compute the rewards obtained by ControlBudget (single resource).
    """

    T = instance.T
    B = instance.B
    arms = instance.arms
    num_resources = instance.num_resources

    # Solve the LP to obtain the support.
    lp_res = solve_lp(instance)
    support = compute_support(lp_res)
    assert(len(support) <= 2)

    rew = []

    # If the LP solution is supported on a positive drift arm.
    if len(support) == 1:
        x0, xp = arms[0], arms[support[0]]
        cur_budget = B
        for t in range(T):
            if cur_budget < 1:
                cur_arm = x0
            else:
                cur_arm = xp
            obs = sample_observation(cur_arm)
            rew.append(obs.reward)
            cur_budget = update_budget(cur_budget, obs.drifts[0])

    # If the LP solution is supported on the null arm and a negative drift arm.
    elif 0 in support:
        x0, xn = arms[0], arms[support[-1]]
        cur_budget = B
        for t in range(T):
            threshold = c * np.log(T - t)
            if cur_budget < max(1, threshold):
                cur_arm = x0
            else:
                cur_arm = xn
            obs = sample_observation(cur_arm)
            rew.append(obs.reward)
            cur_budget = update_budget(cur_budget, obs.drifts[0])

    # If the LP solution is supported on a positive drift arm and a negative
    # drift arm.
    else:
        x0 = arms[0]
        if arms[support[0]].mean_drifts[0] > 0:
            xp, xn = arms[support[0]], arms[support[-1]]
        else:
            xp, xn = arms[support[-1]], arms[support[0]]
        cur_budget = B
        for t in range(T):
            threshold = c * np.log(T - t)
            if cur_budget < 1:
                cur_arm = x0
            elif cur_budget < threshold:
                cur_arm = xp
            else:
                cur_arm = xn
            obs = sample_observation(cur_arm)
            rew.append(obs.reward)
            cur_budget = update_budget(cur_budget, obs.drifts[0])
    
    return rew

def controlbudget(
    instance : Instance,
    c : float
) -> List[float]:
    """
    Compute the rewards obtained by ControlBudget multiple resources.
    """

    rng = default_rng()

    T = instance.T
    B = instance.B
    arms = instance.arms
    num_resources = instance.num_resources

    # Solve the LP to obtain the support and set of binding resources.
    lp_res = solve_lp(instance)
    support = compute_support(lp_res)
    binding_resources = compute_binding_resources(lp_res, num_resources)

    rew = []

    cur_budgets = [B for _ in range(num_resources)]
    x0 = arms[0]
    D = compute_D(instance, support, binding_resources)
    gamma = compute_gamma(instance, lp_res, D)
    for t in range(T):
        # If there exists a resource whose budget is less than 1, pull the null
        # arm.
        if sum([1 for b in cur_budgets if b < 1]) > 0:
            cur_arm = x0

        # Otherwise, sample an arm from a probability distribution over arms as
        # described in the paper.
        else:
            threshold = c * np.log(T - t)
            st = np.array([1 if cur_budgets[j] < threshold else -1 \
                for j in binding_resources])
            b = np.array([-B/T for _ in binding_resources])
            rhs = b + gamma * st
            rhs = np.concatenate((np.array([1]), rhs))
            pt = np.linalg.solve(D, rhs)
            cur_arm = rng.choice([arms[i] for i in support], p = pt)
        obs = sample_observation(cur_arm)
        rew.append(obs.reward)
        cur_budgets = [
            update_budget(cur_budgets[j], obs.drifts[j]) \
                for j in range(num_resources)
        ]

    return rew

def explorethencontrolbudget(
    instance : Instance,
    gamma : float,
    c : float,
    verbose_interval : int = 5000
) -> List[float]:
    """
    Compute the rewards obtained by ExploreThenControlBudget.
    """

    rng = default_rng()

    T = instance.T
    B = instance.B
    arms = instance.arms
    num_resources = instance.num_resources

    rew = []

    num_arms = len(arms)
    cur_budgets = [B for _ in range(num_resources)]
    x0 = arms[0]
    support, binding_resources, nonbinding = set(), set(), set()

    # Solve the LP for debugging purposes.
    lp_sol = solve_lp(instance)
    logging.info(f'LP solution')
    logging.info(f'\tObjective: {lp_sol.fun}')
    logging.info(f'\tSolution: {lp_sol.x}')
    logging.info(f'\tSlack: {lp_sol.slack}')
    logging.info(f'\tSupport: {compute_support(lp_sol)}')
    logging.info(f'\tBinding resources: {compute_binding_resources(lp_sol, num_resources)}')

    num_plays = 0
    t = 1

    def round_robin(t, arm_idx, cur_budgets):
        """
        Play each arm in arm_idx once, playing the null arm if necessary.
        """

        for i in arm_idx:
            while t < T - len(arm_idx) and \
                sum([1 for b in cur_budgets if b < 1]) > 0:
                arm = x0
                obs = sample_observation(arm)
                arm.update(obs)
                cur_budgets = [
                    update_budget(cur_budgets[j], obs.drifts[j]) \
                        for j in range(num_resources)
                ]
                rew.append(obs.reward)
                t += 1
            arm = arms[i]
            obs = sample_observation(arm)
            arm.update(obs)
            cur_budgets = [
                update_budget(cur_budgets[j], obs.drifts[j]) \
                    for j in range(num_resources)
            ]
            rew.append(obs.reward)
            t += 1

        return t, cur_budgets

    # Play each arm num_plays_threshold times. This is technically phase 2 of
    # ETCB as described in the paper, but for practical purposes we execute this
    # phase so that we have many samples before we use UCB / LCB estimates in
    # the LP.
    num_plays_threshold = (32 * np.log(T)) / (gamma**2)
    logging.info(f'Threshold on number of plays: {num_plays_threshold}')
    while t < T - len(support) and num_plays < num_plays_threshold:
        t, cur_budgets = round_robin(t, list(range(num_arms)), cur_budgets)
        num_plays += 1

    logging.info(f'After warmup')
    logging.info(f'\tTotal number of rounds T: {T}')
    logging.info(f'\tIn round t: {t}')
    logging.info(f'\tNumber of plays: {num_plays}')
    logging.info(f'\tConfidence radius: {np.sqrt(8 * np.log(T) / num_plays)}')

    for i in range(num_arms):
        rad = np.sqrt(8 * np.log(T) / len(arms[i].obs_rewards))
        logging.info(f'Arm {i}')
        logging.info(f'\tMean reward: {arms[i].mean_reward}')
        logging.info(f'\tEmpirical reward: {arms[i].empirical_reward}')
        logging.info(f'\tUCB reward: {arms[i].empirical_reward + rad}')
        for j in range(num_resources):
            logging.info(f'\tMean drift for resource {j}: {arms[i].mean_drifts[j]}')
            logging.info(f'\tEmpirical drift for resource {j}: {arms[i].empirical_drifts[j]}')
            logging.info(f'\tUCB drift for resource {j}: {arms[i].empirical_drifts[j] + rad}')

    logging.info(f'Starting first phase in round {t}')

    # While the support and set of binding resources are not identified, play
    # each arm once in a round-robin fashion and solve the modified LPs as
    # described in the paper. Note that, as stated in solve_modified_lp, for our
    # simple examples simply using the empirical means instead of UCB / LCB
    # estimates works well.
    while t < T - num_arms and \
        len(support) + len(nonbinding) < num_resources + 1:
        if t % verbose_interval == 0:
            logging.info(f'\tRound {t}/{T}')
        t, cur_budgets = round_robin(t, list(range(num_arms)), cur_budgets)
        num_plays += 1

        lp = solve_modified_lp(instance, 'lcb', -1, -1)
        if t % verbose_interval == 0:
            logging.info(f'LCB of LP solution')
            logging.info(f'\tObjective: {lp.fun}')
            logging.info(f'\tSolution: {lp.x}')

        for i in range(num_arms):
            modified_lp = solve_modified_lp(instance, 'ucb', i, -1)
            if t % verbose_interval == 0:
                logging.info(f'UCB of LP solution upon removing arm {i}')
                logging.info(f'\tObjective: {modified_lp.fun}')
            if modified_lp.fun < lp.fun - 1e-8:
                support.add(i)

        for j in range(num_resources):
            modified_lp = solve_modified_lp(instance, 'ucb', -1, j)
            if t % verbose_interval == 0:
                logging.info(f'UCB of LP solution upon loosening resource {j}\'s constraint')
                logging.info(f'\tObjective: {modified_lp.fun}')
            if modified_lp.fun < lp.fun - 1e-8:
                nonbinding.add(j)

        if t % verbose_interval == 0:
            logging.info(f'Support: {support}')
            logging.info(f'Binding resources: {binding_resources}')
            logging.info(f'Nonbinding resources: {nonbinding}')

    binding_resources = list(set(range(num_resources)) - nonbinding)
    support = list(support)
    nonbinding = list(nonbinding)

    logging.info(f'Finished first phase')

    logging.info(f'Support: {support}')
    logging.info(f'Binding resources: {binding_resources}')
    logging.info(f'Nonbinding resources: {nonbinding}')

    logging.info(f'Starting second phase in round {t}')

    num_plays_threshold = (32 * np.log(T)) / (gamma**2)
    while t < T - len(support) and num_plays < num_plays_threshold:
        t, cur_budgets = round_robin(t, support, cur_budgets)
        num_plays += 1

    logging.info(f'Finished second phase')

    logging.info(f'Starting third phase in round {t}')

    optimization_fail_counter = 0

    while t <= T:
        if t % verbose_interval == 0:
            logging.info(f'\tRound {t}/{T}')

        # If there exists a resource whose budget is less than 1, pull the null
        # arm.
        if sum([1 for b in cur_budgets if b < 1]) > 0:
            cur_arm = x0

        # Otherwise, sample an arm from a probability distribution over arms as
        # described in the paper.
        else:
            threshold = c * np.log(T - t + 1)
            binding_below = [j for j in binding_resources \
                if cur_budgets[j] < threshold]
            binding_above = [j for j in binding_resources \
                if cur_budgets[j] >= threshold]

            # Solve the optimization problem as described in the paper to find
            # the probability distribution over arms.

            # Since scipy.optimize.minimize requires an objective function but
            # we only require finding a feasible solution, we use a constant
            # objective function.
            def min_fun(x):
                return 1
            
            inital_guess = np.array([1 / len(support) for _ in support])

            bounds = [(0, 1) for _ in support]

            # We initialize the constraint as below for implementation reasons
            # (avoiding initializing the matrices in the if / else conditions
            # below). This initial constraint is subsumed by the sum-to-one
            # constraint defined after the inequality constraints.
            A = np.array([[1 for _ in support]])
            lb = np.array([0])
            ub = np.array([1])
            if len(binding_below) > 0:
                empirical_means = np.array([[
                    arms[i].empirical_drifts[j] for i in support
                ] for j in binding_below])
                rads = np.array([[
                    np.sqrt(8 * np.log(T) / len(arms[i].obs_rewards)) \
                        for i in support
                ] for j in binding_below])
                _A = empirical_means - rads
                """
                _A = np.array([[
                    arms[i].mean_drifts[j] for i in support
                ] for j in binding_below])
                """
                _lb = np.array([gamma/8 for _ in binding_below])
                _ub = np.array([np.inf for _ in binding_below])
                A = np.concatenate((A, _A), axis = 0)
                lb = np.concatenate((lb, _lb))
                ub = np.concatenate((ub, _ub))
            if len(binding_above) > 0:
                empirical_means = np.array([[
                    arms[i].empirical_drifts[j] for i in support
                ] for j in binding_above])
                rads = np.array([[
                    np.sqrt(8 * np.log(T) / len(arms[i].obs_rewards)) \
                        for i in support
                ] for j in binding_above])
                _A = empirical_means + rads
                """
                _A = np.array([[
                    arms[i].mean_drifts[j] for i in support
                ] for j in binding_above])
                """
                _lb = np.array([-np.inf for _ in binding_above])
                _ub = np.array([-gamma/8 for _ in binding_above])
                A = np.concatenate((A, _A), axis = 0)
                lb = np.concatenate((lb, _lb))
                ub = np.concatenate((ub, _ub))
            if len(nonbinding) > 0:
                empirical_means = np.array([[
                    arms[i].empirical_drifts[j] for i in support
                ] for j in nonbinding])
                rads = np.array([[
                    np.sqrt(8 * np.log(T) / len(arms[i].obs_rewards)) \
                        for i in support
                ] for j in nonbinding])
                _A = empirical_means - rads
                """
                _A = np.array([[
                    arms[i].mean_drifts[j] for i in support
                ] for j in nonbinding])
                """
                _lb = np.array([gamma/8 for _ in nonbinding])
                _ub = np.array([np.inf for _ in nonbinding])
                A = np.concatenate((A, _A), axis = 0)
                lb = np.concatenate((lb, _lb))
                ub = np.concatenate((ub, _ub))
            A_eq = np.array([[1 for _ in support]])
            lb_eq = np.array([1])
            ub_eq = np.array([1])
            constraints = [
                optimize.LinearConstraint(A = A, lb = lb, ub = ub),
                optimize.LinearConstraint(A = A_eq, lb = lb_eq, ub = ub_eq)
            ]

            opt_res = optimize.minimize(
                fun = min_fun,
                x0 = inital_guess,
                bounds = bounds,
                constraints = constraints
            )

            if not opt_res.success:
                # logging.info(f'Optimization failed in round {t}')
                optimization_fail_counter += 1
                cur_arm = x0

            pt = opt_res.x

            cur_arm = rng.choice([arms[i] for i in support], p = pt)

        obs = sample_observation(cur_arm)
        rew.append(obs.reward)
        cur_budgets = [
            update_budget(cur_budgets[j], obs.drifts[j]) \
                for j in range(num_resources)
        ]

        t += 1

    logging.info(f'Finished third phase')
    logging.info(f'Optimization failed in {optimization_fail_counter} rounds')

    return rew