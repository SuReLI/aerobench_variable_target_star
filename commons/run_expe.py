import os
import signal
import time
import datetime
import yaml
import math
try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

import torch
import gym
import matplotlib.pyplot as plt
import numpy as np

from commons.utils import NormalizedActions, get_latest_dir

from cfd.starccm.CFDcommunication import CFDcommunication

def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    if type(config['GAME']) == str:
        config['GAME'] = {'id': config['GAME']}
    return config


def create_folder(algo_name, game, config):

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder = f'results/{algo_name}/{game}_{current_time}'

    # Create folder
    if not os.path.exists(f'{folder}/models/'):
        os.makedirs(f'{folder}/models/')
    if not os.path.exists(f'{folder}/variables/'):
        os.makedirs(f'{folder}/variables/')

    # Save config
    with open(f'{folder}/config.yaml', 'w') as file:
        yaml.dump(config, file)

    return folder


def train(Agent, args):

    if args.appli:
        if args.appli == 'flatplate':
            print('running flatplate environment')
            config = load_config(f'cfd/flatplate/config.yaml')
        elif args.appli == 'starccm':
            print('running starccm naca012 environment')
            config = load_config(f'cfd/starccm/config.yaml')
        else:
            print('the CFD environment is not properly defined')
    else:
        config = load_config(f'agents/{args.agent}/config.yaml')

    game = config['GAME']['id'].split('-')[0]
    folder = create_folder(args.agent, game, config)

    if args.load:
        config = load_config(f'{folder}/config.yaml')

    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"\033[91m\033[1mDevice : {device}\nFolder : {folder}\033[0m")

    # Create gym environment and agent
    if config["GAME"]["id"] == "STARCCMexternalfiles":
        env = NormalizedActions(CFDcommunication(config))
    elif config["GAME"]["id"] == "flatplate":
        env = NormalizedActions(FlatPlate(config))
    else:
        env = NormalizedActions(gym.make(**config['GAME']))
    model = Agent(device, folder, config)

    # Load model from a previous run
    if args.load:
        model.load(args.load)

    # Load replay memory from a previous run
    if args.loadrm:
        model.memory.load(args.loadrm)

    # Signal to render evaluation during training by pressing CTRL+Z
    def handler(sig, frame):
        model.evaluate(n_ep=1, render=True)
        # model.plot_Q(pause=True)
    signal.signal(signal.SIGTSTP, handler)

    nb_total_steps = 0
    nb_episodes = 0

    print("Starting training...")
    rewards = []
    eval_returns = []
    eval_scores = []
    lenghts = []
    time_beginning = time.time()
    best_score = - math.inf
    best_episode = 0

    try:
        for episode in trange(config["MAX_EPISODES"]):

            done = False
            step = 0
            episode_reward = 0

            state = env.reset(Btype=config['BTYPE'])

            while not done and step < config["MAX_STEPS"]:
                if args.appli and episode < config["PRE_FILL_EPISODES"]:
                    action = np.array([np.random.uniform(-1.0, 1.0)])
                else:
                    action = model.select_action(state, episode=episode)
                     
                if config["GAME"]["id"] == "STARCCMexternalfiles":
                    env.finishCFD()
                
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                if config["GAME"]["id"] == "STARCCMexternalfiles":
                    #set as done if the number of maximum steps is reached even if not
                    #reached the final position to avoid the simulation to continue
                    if not done and step == config["MAX_STEPS"] - 1:
                        done = True
                

                # Save transition into memory
                model.memory.push(state, action, reward, next_state, done)
                state = next_state

                if args.appli and episode >= config["PRE_FILL_EPISODES"]:
                    losses = model.optimize()

                step += 1
                nb_total_steps += 1

            rewards.append(episode_reward)
            lenghts.append(step)

            if args.appli:
                env.fill_array_tobesaved()

            if isinstance(config["FREQ_SAVE"], int):
                if episode % config["FREQ_SAVE"] == 0:
                    model.save()

            if episode % config["FREQ_EVAL"] == 0:
                eval_returns.append(model.evaluate(n_ep=args.nb_tests))
                eval_scores.append(np.mean(eval_returns[-1]))

                plt.cla()
                plt.title(folder.rsplit('/', 1)[1])
                absc = range(0, len(eval_scores*config["FREQ_EVAL"]), config["FREQ_EVAL"])
                plt.plot(absc, eval_scores)
                plt.savefig(f'{folder}/eval_scores.png')

                if isinstance(config["FREQ_SAVE"], str):
                    if eval_scores[-1] > best_score:
                        best_score = eval_scores[-1]
                        best_episode = episode 
                        model.save()

            if episode % config["FREQ_PLOT"] == 0:

                plt.cla()
                plt.title(folder.rsplit('/', 1)[1])
                plt.plot(rewards)
                plt.savefig(f'{folder}/rewards.png')

                plt.cla()
                plt.title(folder.rsplit('/', 1)[1])
                plt.plot(lenghts)
                plt.savefig(f'{folder}/lenghts.png')
                plt.close()

            nb_episodes += 1

    except KeyboardInterrupt:
        pass

    finally:
        # DUMP variables at the end of training
        filename = folder+'/variables/eval_returns.csv'
        np.savetxt(filename, eval_returns, delimiter=";")
        filename = folder+'/variables/lenghts.csv'
        np.savetxt(filename, lenghts, delimiter=";")

        if args.appli:
            env.print_array_in_files(folder+'/variables/')
            env.plot_training_output(rewards, eval_scores, config["FREQ_EVAL"], folder)

        env.close()
        if config["FREQ_SAVE"] != 'best':
            model.save()
        model.memory.save(folder)
        print('saved best model score is ', best_score)

        if config["GAME"]["id"] == "STARCCMexternalfiles":
            #end simulation of STARCCM+
            env.finishCFD(True)

    time_execution = time.time() - time_beginning

    print('---------------------------------------------------\n'
          '---------------------STATS-------------------------\n'
          '---------------------------------------------------\n',
          nb_total_steps, ' steps and updates of the network done\n',
          nb_episodes, ' episodes done\n'
          'Execution time : ', round(time_execution, 2), ' seconds\n'
          '---------------------------------------------------\n'
          'Average nb of steps per second : ', round(nb_total_steps/time_execution, 3), 'steps/s\n'
          'Average duration of one episode : ', round(time_execution/max(1, nb_episodes), 3), 's\n'
          '---------------------------------------------------')


def test(Agent, args):

    if args.folder is None:
        agent_folder = f'results/{args.agent}'
        args.folder = os.path.join(get_latest_dir(agent_folder))

    with open(os.path.join(args.folder, 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device('cpu')

    # Creating neural networks and loading models
    print(f"Testing \033[91m\033[1m{args.agent}\033[0m saved in the folder {args.folder}")
    model = Agent(device, args.folder, config)
    model.load()

    rewards = model.evaluate(n_ep=args.nb_tests, render=args.render, gif=args.gif, test=True, appli=args.appli)
    score = np.mean(rewards)
    print(f"Average score : {score}")
