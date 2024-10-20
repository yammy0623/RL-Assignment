import random
import numpy as np
import json
import matplotlib.pyplot as plt
import wandb

from algorithms import (
    MonteCarloPrediction,
    TDPrediction,
    NstepTDPrediction,
    MonteCarloPolicyIteration,
    SARSA,
    Q_Learning,
)
from gridworld import GridWorld

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# 2-1
STEP_REWARD     = -0.1
GOAL_REWARD     = 1.0
TRAP_REWARD     = -1.0
INIT_POS        = [0]
DISCOUNT_FACTOR = 0.9
POLICY          = None
MAX_EPISODE     = 300
LEARNING_RATE   = 0.01
NUM_STEP        = 3
# 2-2
EPSILON           = 0.2
BUFFER_SIZE       = 10000
UPDATE_FREQUENCY  = 200
SAMPLE_BATCH_SIZE = 500

def bold(s):
    return "\033[1m" + str(s) + "\033[0m"


def underline(s):
    return "\033[4m" + str(s) + "\033[0m"


def green(s):
    return "\033[92m" + str(s) + "\033[0m"


def red(s):
    return "\033[91m" + str(s) + "\033[0m"


def init_grid_world(maze_file: str = "maze.txt", init_pos: list = None):
    print(bold(underline("Grid World")))
    grid_world = GridWorld(
        maze_file,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
        init_pos=init_pos,
    )
    grid_world.print_maze()
    grid_world.visualize(title="Maze", filename="maze.png", show=False)
    print()
    return grid_world


def run_MC_prediction(grid_world: GridWorld,seed):
    print(f"Run MC prediction. Seed:{seed}")
    prediction = MonteCarloPrediction(
        grid_world,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        seed = seed
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"Monte Carlo Prediction",
        show=False,
        filename=f"MC_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()
    return prediction.get_all_state_values()


def run_TD_prediction(grid_world: GridWorld, seed):
    print(f"Run TD(0) prediction. Seed:{seed}")
    prediction = TDPrediction(
        grid_world,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        learning_rate=LEARNING_RATE,
        seed = seed
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"TD(0) Prediction",
        show=False,
        filename=f"TD0_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()
    return prediction.get_all_state_values()


def run_NstepTD_prediction(grid_world: GridWorld,seed):
    print(f"Run N-step TD prediction. Seed:{seed}")
    prediction = NstepTDPrediction(
        grid_world,
        learning_rate=LEARNING_RATE,
        num_step=NUM_STEP,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        seed=seed,
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"N-step TD Prediction",
        show=False,
        filename=f"NstepTD_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()
    return prediction.get_all_state_values()

def run_MC_policy_iteration(grid_world: GridWorld, iter_num: int):
    print(bold(underline("MC Policy Iteration")))
    policy_iteration = MonteCarloPolicyIteration(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            )
    policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"MC Policy Iteration",
        show=False,
        filename=f"MC_policy_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


def run_SARSA(grid_world: GridWorld, iter_num: int):
    print(bold(underline("SARSA Policy Iteration")))
    policy_iteration = SARSA(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            )
    policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"SARSA",
        show=False,
        filename=f"SARSA_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


def run_Q_Learning(grid_world: GridWorld, iter_num: int):
    print(bold(underline("Q_Learning Policy Iteration")))
    policy_iteration = Q_Learning(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            buffer_size=BUFFER_SIZE,
            update_frequency=UPDATE_FREQUENCY,
            sample_batch_size=SAMPLE_BATCH_SIZE,
            )
    policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"Q_Learning",
        show=False,
        filename=f"Q_Learning_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()

def analysis(model_name):
    v_gt = np.load("prediction_GT.npy")
    grid_world = init_grid_world("maze.txt",INIT_POS)
    all_state_value = []

    wandb.init(project=model_name + "diff")
    for seed in range(1, 51):
        if model_name == "MC_prediction_all":
            state_value = run_MC_prediction(grid_world,seed)
        elif model_name == "TD0_prediction_all":
            state_value = run_TD_prediction(grid_world,seed)   
        all_state_value.append(state_value)
        log_dict = {f"state_{i}": value for i, value in enumerate(state_value - v_gt)}
        wandb.log(log_dict)
        # wandb.log({"mean_state_value": np.mean(state_value)})

    wandb.finish()
    v_avg = np.mean(all_state_value, axis=0)
    bias = v_avg - v_gt
    var = np.var(all_state_value, axis=0)

    if model_name == "MC_prediction_all":
        grid_world.visualize(
            bias,
            title=f"MC Prediction (Bias for 50 seeds)",
            show=False,
            filename=f"MC_prediction_bias.png",
        )
        grid_world.visualize(
            var,
            title=f"MC Prediction (Variance for 50 seeds)",
            show=False,
            filename=f"MC_prediction_var.png",
        )
        print("Mean Bias: " + str(np.mean(bias)))
        print("Mean Var: " + str(np.mean(var)))

        np.save('MC_v_avg.npy', v_avg)
        np.save('MC_bias.npy', bias)

    elif model_name== "TD0_prediction_all":
        grid_world.visualize(
            bias,
            title=f"TD0 Prediction (Bias for 50 seeds)",
            show=False,
            filename=f"TD0_prediction_bias.png",
        )
        grid_world.visualize(
            var,
            title=f"TD0 Prediction (Variance for 50 seeds)",
            show=False,
            filename=f"TD0_prediction_var.png",
        )
        print("Mean Bias: " + str(np.mean(bias)))
        print("Mean Var: " + str(np.mean(var)))

        np.save('TD0_v_avg.npy', v_avg)
        np.save('TD0_bias.npy', bias)

    
def MC_curve_analysis(iter_num, epsilon_set):
    grid_world = init_grid_world("maze.txt")
    all_lr = []
    all_loss = []
    
    for e in epsilon_set:
        print(e)
        policy_iteration = MonteCarloPolicyIteration(
                grid_world, 
                discount_factor=DISCOUNT_FACTOR,
                learning_rate=LEARNING_RATE,
                epsilon= e,
        )
        policy_iteration.run(max_episode=iter_num)
        lr = policy_iteration.learning_curve
        loss = policy_iteration.loss_curve
        all_lr.append(lr)
        all_loss.append(loss)
    
    plt.figure()
    for i in range(epsilon_set):
        plt.plot(range(1, len(lr[i])+1), lr, label=f'epsilon = {epsilon_set[i]}')

    plt.xlabel('Episode')
    plt.ylabel('Average Non-discounted Episodic Reward')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig('MC_learning_curve.png', dpi=300, bbox_inches='tight') 
    plt.show()

    plt.figure()
    for i in range(epsilon_set):
        plt.plot(range(1, len(loss[i])+1), loss, label=f'epsilon = {epsilon_set[i]}')
    plt.xlabel('Episode')
    plt.ylabel('Average Non-discounted Episodic Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig('MC_loss_curve.png', dpi=300, bbox_inches='tight') 
    plt.show()

def wandb_analysis(iter_num, epsilon_set, model_name):
    # start a new wandb run to track this script
    
    grid_world = init_grid_world("maze.txt")

    for e in epsilon_set:
        print("epsilon = " + str(e))
        wandb.init(project=model_name + "_abs", name="epsilon = " + str(e))
        if model_name == "MC":

            policy_iteration = MonteCarloPolicyIteration(
                    grid_world, 
                    discount_factor=DISCOUNT_FACTOR,
                    learning_rate=LEARNING_RATE,
                    epsilon= e,
            )
        elif model_name == "SARSA":
            policy_iteration = SARSA(
                    grid_world, 
                    discount_factor=DISCOUNT_FACTOR,
                    learning_rate=LEARNING_RATE,
                    epsilon= e,
                    )
        elif model_name == "Q_Learning":

            policy_iteration = Q_Learning(
                    grid_world, 
                    discount_factor=DISCOUNT_FACTOR,
                    learning_rate=LEARNING_RATE,
                    epsilon= e,
                    buffer_size=BUFFER_SIZE,
                    update_frequency=UPDATE_FREQUENCY,
                    sample_batch_size=SAMPLE_BATCH_SIZE,
                    )
        policy_iteration.run(max_episode=iter_num)
        # lr = policy_iteration.learning_curve
        # loss = policy_iteration.loss_curve
        wandb.finish()
 

if __name__ == "__main__":
    # analysis("MC_prediction_all")
    # analysis("TD0_prediction_all")
    
    iter_num = 512000
    epsilon_set = [0.1, 0.2, 0.3, 0.4]
    # wandb_analysis(iter_num, epsilon_set, "MC")
    wandb_analysis(iter_num, epsilon_set, "SARSA")
    # wandb_analysis(iter_num, epsilon_set, "Q_Learning")
    # Matplot
    # MC_curve_analysis(iter_num, epsilon_set)
    
    # grid_world = init_grid_world("maze.txt",INIT_POS)
    # 2-1
    # run_MC_prediction(grid_world,seed)
    # run_TD_prediction(grid_world,seed)
    # run_NstepTD_prediction(grid_world,seed)

    # 2-2
    # grid_world = init_grid_world("maze.txt")
    # run_MC_policy_iteration(grid_world, 512000)
    # run_SARSA(grid_world, 512000)
    # run_Q_Learning(grid_world, 50000)