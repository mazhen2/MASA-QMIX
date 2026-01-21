"""
MARL训练和评估运行器

该模块负责管理多智能体强化学习的训练过程和评估过程。
支持多种MARL算法的训练，包括QMIX、VDN、COMA等。
"""

import numpy as np
import os
from MARL.common.rollout import RolloutWorker, CommRolloutWorker
from MARL.agent.agent import Agents, CommAgents
from MARL.common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import sys
from MARL.common import plot_utils


class Runner:
    """
    MARL算法的训练和评估运行器

    负责协调环境、智能体和经验回放缓冲区，执行训练循环和评估。
    支持on-policy和off-policy算法的训练。
    """
    def __init__(self, env, args):
        """
        初始化运行器

        Args:
            env: 船只调度环境实例
            args: 算法参数配置
        """
        self.env = env
        self.args = args

        # 根据算法类型选择相应的智能体和rollout工作器
        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:
            # 通信型智能体（支持智能体间通信）
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:
            # 普通智能体（无通信）
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)

        # 为off-policy算法创建经验回放缓冲区
        # COMA, Central V, Reinforce是on-policy算法，不需要缓冲区
        if args.learn and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:
            self.buffer = ReplayBuffer(args)

        # 训练统计信息
        self.win_rates = []      # 胜率记录（用于兼容其他环境）
        self.episode_rewards = [] # 奖励记录

        # 结果保存路径
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        """
        执行完整的训练过程

        Args:
            num: 运行编号（用于多算法并行运行时的标识）
        """
        train_steps = 0  # 总训练步数计数
        for_gantt_data = []  # 甘特图数据（暂时未使用）

        # 判断是否为on-policy算法
        is_on_policy = (self.args.alg.find('coma') > -1 or
                       self.args.alg.find('central_v') > -1 or
                       self.args.alg.find('reinforce') > -1)

        # 估算总训练步数（用于进度显示）
        estimated_total_train_steps = self.args.n_epoch * (1 if is_on_policy else max(1, self.args.train_steps))

        r_s = [0]  # 最近奖励列表，用于显示平均奖励

        # 主训练循环
        for epoch in range(self.args.n_epoch):
            # 显示训练进度
            text = '\rRun {run_id}, epoch {cur_epoch}/{total_epoch}, train_step {train_step}/{total_steps}, ave_rewards {ave:.3f}'
            sys.stdout.write(text.format(run_id=num,
                                         cur_epoch=epoch + 1,
                                         total_epoch=self.args.n_epoch,
                                         train_step=train_steps,
                                         total_steps=estimated_total_train_steps,
                                         ave=(sum(r_s) / len(r_s))))
            sys.stdout.flush()

            # 定期评估模型性能
            if epoch % self.args.evaluate_cycle == 0 and epoch != 0:
                win_rate, episode_reward = self.evaluate()
                self.win_rates.append(win_rate)
                print("\nepisode_reward:", episode_reward, "epoch:", epoch)
                self.episode_rewards.append(episode_reward)

                # 保存调度结果数据
                with open("./my_data_and_graph/historydata/scheduleresults.txt", "a") as f:
                    print(for_gantt_data, file=f)


            episodes = []
            r_s = []
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, _, _, for_gantt_data = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                r_s.append(sum(episode['r'][0])[0])
                # print(_)
            # 将多个episode的数据拼接成batch
            # episode数据结构：(1, episode_len, n_agents, feature_dim)
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            # 根据算法类型选择训练方式
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                # On-policy算法：直接使用episode数据训练
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                # Off-policy算法：先存入缓冲区，再采样训练
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1

        # save training curves and data
        try:
            self.plt(num)
            print("Training curves and numpy arrays saved to:", self.save_path)
        except Exception as e:
            print("Failed to save training curves:", e)

        # try to generate and save Gantt chart (best-effort)
        try:
            out_gantt = plot_utils.generate_and_save_gantt(self.save_path, out_filename="gantt.png")
            print("Gantt chart saved to:", out_gantt)
        except FileNotFoundError as e:
            # no scheduling records found
            print("No scheduling records found to generate Gantt chart:", e)
        except Exception as e:
            print("Failed to generate Gantt chart:", e)
    def evaluate(self):
        """
        评估当前模型的性能

        Returns:
            tuple: (平均胜率, 平均奖励)
        """
        win_number = 0      # 获胜次数
        episode_rewards = 0 # 总奖励

        # 进行多次评估episode
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, for_gant = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1

        # 打印甘特图数据
        print(for_gant)

        # 返回平均胜率和平均奖励
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        plt.figure()
        plt.axis([0, self.args.n_epoch, 0, 100])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rate')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)

