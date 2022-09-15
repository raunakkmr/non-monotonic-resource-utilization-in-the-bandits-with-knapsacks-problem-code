import argparse
import logging
import matplotlib.pyplot as plt
import os

import numpy as np
from scipy import optimize

import algorithms
import data

def controlbudget_one_resource(
    args : argparse.Namespace,
    instance : data.Instance
) -> np.ndarray:
    """
    Helper function to call control budget for one resource.
    """

    c = algorithms.compute_c_one_resource(instance)
    logging.info(f'\tValue of c: {c}')
    rewards = [algorithms.controlbudget_one_resource(instance, c) \
        for _ in range(args.trials)]
    cb_rew = np.cumsum(np.mean(rewards, axis = 0))

    return cb_rew

def controlbudget(
    args : argparse.ArgumentParser,
    instance : data.Instance,
    res : optimize.OptimizeResult
) -> np.ndarray:
    """
    Helper function to call control budget for multiple resources.
    """

    c = algorithms.compute_c_multiple_resources(instance, res)
    logging.info(f'\tValue of c: {c}')
    rewards = [algorithms.controlbudget(instance, c) \
        for _ in range(args.trials)]
    cb_rew = np.cumsum(np.mean(rewards, axis = 0))

    return cb_rew

def main():

    logging.basicConfig(
        format='%(asctime)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s',
        level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default='testing.txt')
    parser.add_argument('--learning', action='store_true')
    parser.add_argument('--trials', type=int, default=1)
    args = parser.parse_args()

    # Create an instance object from the data file.
    instance = data.parse_datafile(os.path.join('../data', args.data_filename))
    multiple_resources_flag = \
        True if len(instance.arms[0].mean_drifts) > 1 else False
    logging.info(f'Multiple resources: {multiple_resources_flag}')

    # Solve the linear program.
    logging.info(f'Solving the linear program.')
    res = algorithms.solve_lp(instance)
    support = algorithms.compute_support(res)
    binding_resources = algorithms.compute_binding_resources(
        res, instance.num_resources
    )
    lp_rew = [(res.fun * T) for T in range(1, instance.T + 1)]

    logging.info(f'\tObjective: {res.fun}')
    logging.info(f'\tSolution: {res.x}')
    logging.info(f'\tSlack: {res.slack}')
    logging.info(f'\tSupport: {support}')
    logging.info(f'\tBinding resources: {binding_resources}')
    # assert(len(support) <= 2)
    # if not multiple_resources_flag:
    #     if len(support) == 1:
    #         logging.info('LP solution is supported on one arm.')
    #     elif 0 in support:
    #         logging.info('LP solution is supported on null arm and negative drift arm.')
    #     else:
    #         logging.info('LP solution is supported on a positive drift arm and a negative drift arm.')

    # Play the MDP policy.
    logging.info(f'Playing the MDP policy ({args.trials} trials).')
    if not multiple_resources_flag:
        cb_rew = controlbudget_one_resource(args, instance)
    else:
        cb_rew = controlbudget(args, instance, res)

    # Difference between the LP and the MDP policy.
    logging.info(f'Regret')
    regret = [lp_rew[i] - cb_rew[i] for i in range(instance.T)]

    # Plot.
    logging.info(f'Plotting.')
    xvalues = list(range(1, instance.T + 1))
    plt.plot(xvalues, lp_rew, label = 'LP rewards', color = 'tab:green')
    plt.plot(xvalues, cb_rew, label = 'CB rewards', color = 'tab:blue')
    plt.plot(xvalues, regret, label = 'Regret', color = 'tab:red')
    plt.xlabel('Time horizon.')
    if not multiple_resources_flag:
        plt.title('LP vs ControlBudget (one resource).')
    else:
        plt.title('LP vs ControlBudget (multiple resources).')
    plt.legend()
    if not os.path.exists('../plots'):
        os.makedirs('../plots')
    plot_filename = args.data_filename.replace('.txt', '_cb' + '.png')
    plt.savefig(os.path.join('../plots', plot_filename))

    if not args.learning:
        return 0

    # Play the learning algorithm.
    logging.info(f'Playing the learning algorithm ({args.trials} trials).')
    D = algorithms.compute_D(instance, support, binding_resources)
    gamma = algorithms.compute_gamma(instance, res, D, verbose = True)
    c = algorithms.compute_c_multiple_resources(instance, res)
    logging.info(f'\tValue of c: {c}')
    rewards = [algorithms.explorethencontrolbudget(instance, gamma, c) \
        for _ in range(args.trials)]
    etcb_rew = np.cumsum(np.mean(rewards, axis = 0))

    # Difference between the LP and the learning algorithm.
    logging.info(f'Regret')
    regret = [lp_rew[i] - etcb_rew[i] for i in range(instance.T)]

    # Plot.
    logging.info(f'Plotting.')
    xvalues = list(range(1, instance.T + 1))
    plt.clf()
    plt.plot(xvalues, lp_rew, label = 'LP rewards', color = 'tab:green')
    plt.plot(xvalues, etcb_rew, label = 'ETCB rewards', color = 'tab:blue')
    plt.plot(xvalues, regret, label = 'Regret', color = 'tab:red')
    plt.xlabel('Time horizon.')
    if not multiple_resources_flag:
        plt.title('LP vs ETCB (one resource).')
    else:
        plt.title('LP vs ETCB (multiple resources).')
    plt.legend()
    if not os.path.exists('../plots'):
        os.makedirs('../plots')
    plot_filename = args.data_filename.replace('.txt', '_etcb' + '.png')
    plt.savefig(os.path.join('../plots', plot_filename))

if __name__ == '__main__':
    main()