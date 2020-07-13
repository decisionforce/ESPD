import gym
import random
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as opt

from replay_buffer import ReplayBuffer_imitation

env = gym.make("FetchPush-v1")

def select_action(action_mean, action_logstd, fctr):
    """
    given mean and std, sample an action from normal(mean, std)
    also returns probability of the given chosen
    """
    action_std = torch.exp(action_logstd) * fctr
    action = torch.normal(action_mean, action_std)
    return action

def eval_policy_50(fctr_used, args, network, device):
    # env = gym.make(args.env_name)
    reward_sum = 0
    succ_game = 0
    for display_i in range(50):
        env.reset()
        state = env.env._get_obs()
        state = np.concatenate(
            (state['observation'], state['desired_goal']))  # state_extended
        episode = []
        env_list = []
        Succ_in_env = 0
        for t in range(args.max_step_per_round):
            network.eval()
            action_mean, action_logstd, value = network(
                Tensor(state).unsqueeze(0).to(device))
            action_mean = action_mean.detach()
            action_logstd = action_logstd.detach()
            value = value.detach()

            action = select_action(action_mean, action_logstd, fctr_used)
            action = torch.clamp(action, -1, 1)
            action = action.data.cpu().numpy()[0]
            next_state, reward, done, _ = env.step(action)

            if _['is_success'] != 0:
                Succ_in_env = 1
                break

            next_state = np.concatenate(
                (next_state['observation'], next_state['desired_goal']))

            reward_sum += reward

            mask = 0 if done else 1

            if done:
                break
            state = next_state
        succ_game += Succ_in_env

    return succ_game / 50

def espd(args, network, device):
    def compute_cross_ent_error(batch_size, step_num):
        if ier_buffer.lenth(step_num) == 0:
            return None
        if batch_size > ier_buffer.lenth(step_num):
            return None
        state, action = ier_buffer.sample(batch_size, step_num)
        state = torch.FloatTensor(state).to(device)
        action_target = torch.FloatTensor(action).to(device)
        action_pred = model_imitation(state)[0]

        loss_func = nn.MSELoss()
        loss = loss_func(action_pred, action_target)
        optimizer_imitation.zero_grad()
        loss.backward()
        optimizer_imitation.step()
        return loss

    def test_isvalid_multistep(step_lenth, state_start, environment_start,
                               env):
        env_tim = env
        env_tim.sim.set_state(environment_start)
        env_tim.sim.forward()
        state_tim = deepcopy(state_start)
        for step_i in range(step_lenth):
            action_tim_mean, action_tim_logstd, value_tim = network(
                Tensor(state_tim).unsqueeze(0).to(device))
            action_tim_mean = torch.clamp(action_tim_mean, -1, 1)
            action_tim = action_tim_mean.cpu().data.numpy()[0]
            next_state_tim, reward, done, _ = env_tim.step(action_tim)
            next_state_tim = np.concatenate((next_state_tim['observation'],
                                             next_state_tim['desired_goal']))

            next_state_tim[-3:] = deepcopy(state_tim[-3:])

            rwd_sim = env_tim.compute_reward(next_state_tim[3:6],
                                             next_state_tim[-3:],
                                             {'is_success': 0.0})
            if rwd_sim == 0:
                if step_i <= step_lenth - 1:
                    return 1  # should not learn
                else:
                    return 0  # ok to learn
            state_tim = next_state_tim
        return 2  # learnable

    # env = gym.make(args.env_name)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # random.seed(args.seed)
    # env.seed(args.seed)

    Horizon_list = [i + 1 for i in range(args.Horizon_max)]
    Acceptance_rate = []
    FACTOR = args.factor

    model_imitation = network
    num_inputs = env.observation_space.spaces['observation'].shape[
        0] + env.observation_space.spaces['desired_goal'].shape[
            0]  # extended state
    num_actions = env.action_space.shape[0]

    optimizer_imitation = opt.RMSprop(model_imitation.parameters(),
                                      lr=args.lr_hid)

    reward_record = []
    global_steps = 0

    ier_buffer = ReplayBuffer_imitation(args.replay_buffer_size_IER)

    for i_episode in range(args.num_episode):
        episodic_pass_test_num = 0
        num_steps = 0
        reward_list = []
        len_list = []
        Succ_num = 0

        game_num = 0
        succ_game = 0

        Ret_2 = [0 * _ for _ in range(len(Horizon_list))]
        Ret_1 = [0 * _ for _ in range(len(Horizon_list))]
        Ret_0 = [0 * _ for _ in range(len(Horizon_list))]

        while num_steps < args.batch_size:
            '''interactions'''
            state = env.reset()

            game_num += 1
            state = np.concatenate((state['observation'],
                                    state['desired_goal']))  # state_extended

            reward_sum = 0
            episode = []
            env_list = []
            Succ_in_env = 0
            for t in range(args.max_step_per_round):
                action_mean, action_logstd, value = network(
                    Tensor(state).unsqueeze(0).to(device))
                action, logproba = network.select_action(action_mean,
                                                         action_logstd,
                                                         factor=FACTOR)

                action = torch.clamp(action, -1, 1)
                action = action.cpu().data.numpy()[0]
                logproba = logproba.cpu().data.numpy()[0]

                if len(Horizon_list) >= 2:
                    state_temp = env.env.sim.get_state()
                    env_list.append(state_temp)

                next_state, reward, done, _ = env.step(action)
                if reward == 0:
                    Succ_in_env = 1
                    reward = args.reward_pos
                    Succ_num += 1
                next_state = np.concatenate(
                    (next_state['observation'], next_state['desired_goal']))

                reward_sum += reward
                mask = 0 if done else 1

                episode.append(
                    (state, value, action, logproba, mask, next_state, reward))
                if done:
                    break

                state = next_state
            succ_game += Succ_in_env

            '''start learning'''
            for ind, (state, value, action, logproba, mask, next_state,
                      reward) in enumerate(episode):
                if len(Horizon_list) >= 2:
                    assert len(env_list) == len(episode)
                '''supervised learning'''
                for t_ in Horizon_list:
                    try:
                        episode[t_ + ind]
                    except:
                        break

                    target_state_ = deepcopy(episode[t_ + ind][-7])
                    state_ = deepcopy(state)
                    state_[-3:] = deepcopy(target_state_[3:6])
                    rwd_temp3 = np.linalg.norm(target_state_[3:6] -
                                               state_[3:6])
                    if rwd_temp3 > 0.05:
                        ret_tim = test_isvalid_multistep(
                            t_, state_, env_list[ind], env)
                        if ret_tim == 2:
                            ier_buffer.push(state_, action, '1step')
                            episodic_pass_test_num += 1
                            Ret_2[t_ - 1] += 1
                        elif ret_tim == 1:
                            Ret_1[t_ - 1] += 1
                        else:
                            Ret_0[t_ - 1] += 1

            num_steps += (t + 1)
            global_steps += (t + 1)
            reward_list.append(reward_sum)
            len_list.append(t + 1)
            Winrate = 1.0 * succ_game / game_num

        print('Return This Episode:', Ret_0, Ret_1, Ret_2)
        Acceptance_rate.append([
            round((Ret_2[_] /
                   (Ret_2[_] + Ret_1[_] + Ret_0[_] + 1e-6)) * 100.0) / 100.0
            for _ in range(len(Ret_2))
        ])

        reward_record.append({
            'episode': i_episode,
            'steps': global_steps,
            'meanepreward': np.mean(reward_list),
            'meaneplen': np.mean(len_list)
        })

        batch_size = episodic_pass_test_num

        SR = 1.0 * Succ_num / num_steps
        for i_epoch in range(
                int(args.num_epoch * batch_size / args.minibatch_size)):
            '''learning'''
            for h in [1]:
                flag = 0
                loss1 = compute_cross_ent_error(args.minibatch_size,
                                                str(h) + 'step')

        print('ier lenth', ier_buffer.lenth('1step'),
              ier_buffer.lenth('2step'), ier_buffer.lenth('3step'),
              ier_buffer.lenth('4step'), ier_buffer.lenth('5step'),
              ier_buffer.lenth('6step'), ier_buffer.lenth('7step'))

        eval_0_temp = eval_policy_50(0.0, args, network, device)
        eval_0p1_temp = eval_policy_50(0.1, args, network, device)
        eval_0p2_temp = eval_policy_50(0.2, args, network, device)
        print('Eval_sr:', eval_0_temp, eval_0p1_temp, eval_0p2_temp)
        print('Acceptance Rate ', Acceptance_rate[-1])
        print('Traj length in this episode', Ret_2)

        if i_episode % args.log_num_episode == 0:
            print('Finished episode: {} Reward: {:.4f} SuccessRate{:.4f} WinRate{:.4f}' \
                .format(i_episode, reward_record[-1]['meanepreward'],SR,Winrate))
            print('-----------------')

    return reward_record
