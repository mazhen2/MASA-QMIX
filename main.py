
"""
MASA-QMIX: Multi-Agent Scheduling Algorithm using QMIX
船只调度问题的多智能体强化学习解决方案

该项目实现了基于QMIX算法的多智能体强化学习，用于解决船只保障任务调度问题。
支持多种MARL算法：QMIX, VDN, COMA, Central V, Reinforce, CommNet, G2ANet等。
"""

import numpy as np
import pickle
from environment import ScheduleEnv
import sys
from os.path import dirname, abspath

# 添加项目根目录到Python路径
sys.path.append(dirname(dirname(abspath(__file__))))

# 导入MARL相关模块
from MARL.runner import Runner
from MARL.common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, \
    get_reinforce_args, \
    get_commnet_args, get_g2anet_args
from utils.PDRs.shortestDistence import SDrules

# 设置随机种子保证实验可重复性
np.random.seed(2)

# game_scores, rolling_scores的区别，一个是整个episode的reward
# ，一个是一个trial的reward，两个指标都不参与训练，trial是人为指定的一个固定长度的n个step的过程


def marl_agent_wrapper():
    """
    多智能体强化学习决策函数

    使用QMIX等MARL算法进行船只调度决策的包装函数。
    包含完整的训练流程：初始化环境、配置算法参数、训练模型、评估性能。
    """

    # 初始化数据记录文件
    # 清空历史数据文件，为新的训练运行做准备
    with open("./my_data_and_graph/historydata/accumulated_rewards.txt", "w") as f:
        print("----", file=f)  # 写入分隔符标记新的训练开始
    with open("my_data_and_graph/times.txt", "w") as f:
        print("----", file=f)  # 记录训练时间

    # 初始化其他数据记录文件
    with open("./my_data_and_graph/historydata/havealook.txt", "w") as f:
        pass  # 用于调试输出
    with open("./my_data_and_graph/historydata/loss.txt", "w") as f:
        pass  # 记录训练损失
    with open("./my_data_and_graph/historydata/scheduleresults.txt", "w") as f:
        pass  # 记录调度结果

    # 注释掉的代码：用于备份历史训练数据
    # import datetime, os, time
    # from shutil import copyfile
    # if os.path.exists("my_data_and_graph/marl.time_reward.txt"):
    #     tar = "my_data_and_graph/marlhisrtorydata/" + str(datetime.date.today()) + "-" + str(time.time()).split(".")[
    #         0] + "marl.time_reward.txt"
    #     with open(tar, "w") as f:
    #         pass
    #     copyfile("my_data_and_graph/marl.time_reward.txt", tar)
    #
    # with open("my_data_and_graph/marl.time_reward.txt", "w") as f:
    #     pass
    # for i in range(8):  # 因为一共8种marl算法

    # 获取算法的通用参数配置
    args = get_common_args()

    # 根据选择的算法获取特定的参数配置
    if args.alg.find('coma') > -1:  # COMA算法参数
        args = get_coma_args(args)
    elif args.alg.find('central_v') > -1:  # Central V算法参数
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:  # REINFORCE算法参数
        args = get_reinforce_args(args)
    else:  # 其他算法（QMIX, VDN等）使用mixer参数
        args = get_mixer_args(args)

    # 添加通信相关算法的参数
    if args.alg.find('commnet') > -1:
        args = get_commnet_args(args)
    if args.alg.find('g2anet') > -1:
        args = get_g2anet_args(args)

    # 初始化船只调度环境
    env = ScheduleEnv()

    # 重置环境到初始状态
    env.reset()

    # 获取环境信息，用于配置智能体参数
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]  # 每个智能体的动作空间大小
    args.n_agents = env_info["n_agents"]    # 智能体数量
    args.state_shape = env_info["state_shape"]  # 全局状态维度
    args.obs_shape = env_info["obs_shape"]      # 局部观测维度
    args.episode_limit = env_info["episode_limit"]  # 单个episode的最大步数

    # 打印配置信息
    print("是否加载模型（测试必须）：", args.load_model, "是否打印中间变量：", args.havelook, "是否训练：",args.learn)

    # 创建训练/评估运行器
    runner = Runner(env, args)

    # 根据模式选择训练或评估
    if args.learn:
        runner.run(0)  # 训练模式，0是算法ID（原来支持多种算法同时运行）
    else:
        _, reward = runner.evaluate()  # 评估模式
        print('The ave_reward of {} is  {}'.format(args.alg, reward))


def random_agent_wrapper():
    """
    随机决策函数，用于测试环境

    使用随机策略进行船只调度，作为基线方法进行对比。
    执行多个episode，记录调度结果和甘特图数据。
    """

    episodes = 50  # 测试的总episode数量

    env = ScheduleEnv()  # 初始化调度环境
    temp_save = [0]      # 临时保存变量
    EATs = []           # 存储每个episode的完成时间（Estimated Arrival Time）
    schedule_processes = []  # 存储调度过程记录

    # 运行多个episode进行测试
    for episode in range(episodes):

        s = env.reset()  # 重置环境
        is_terminal = False  # episode结束标志

        # 单episode内的决策循环
        while not is_terminal:
            actions = []  # 存储所有智能体的动作

            # 找出当前正在工作的智能体（非空闲状态）
            temp_not_idle_agents = []
            for m in range(len(env.sites)):
                if s[m] != 9:  # 9表示空闲状态
                    temp_not_idle_agents.append(s[m])

            # 为每个飞机选择动作
            for i in range(len(env.planes)):
                if i in temp_not_idle_agents:  # 飞机正在忙碌
                    actions.append(18)  # 等待动作
                else:
                    # 获取当前智能体的可用动作
                    avail_actions = env.get_avail_agent_actions(i)
                    tem_choose = []

                    # 找出所有可用的动作
                    if type(avail_actions) != str:
                        for k, eve in enumerate(avail_actions):
                            if eve == 1:  # 1表示该动作可用
                                tem_choose.append(k)

                        if tem_choose == []:  # 没有可用动作
                            action = 18  # 等待
                        else:
                            # 从可用动作中随机选择一个
                            action = np.random.choice(tem_choose, 1, False)[0]
                            env.has_chosen_action(action, i)  # 标记已选择该动作
                        actions.append(action)
                    else:
                        actions.append(18)  # 默认等待动作

            # 执行动作，获取下一个状态、奖励和终止标志
            s, r, is_terminal, dict = env.step(actions)
            # print(actions)
            # print(s[:18])
        EATs.append(dict["time"])
        schedule_processes.append(env.job_record_for_gant)
        print(env.job_record_for_gant)
        print(dict["time"], "-----------------------------------")
    print(sum(EATs)/len(EATs))
    # 存储中间结果
    with open("./my_data_and_graph/pickles/process.pk", "wb") as f:
        pickle.dump(schedule_processes, f)


def SDrules_agent_wrapper():
    """
    基于启发式规则的决策函数

    使用最短距离规则（Shortest Distance Rules）进行船只调度决策。
    包括FIFO（先进先出）、MLF（最长飞行时间优先）、LLF（最后期限优先）等规则。
    """
    EPISODES = 50  # 测试的总episode数量

    sd_rules = SDrules()  # 初始化启发式规则对象
    env = ScheduleEnv()   # 初始化调度环境
    sites_locations = env.sites_obj.sites_position  # 获取所有站位的地理位置

    actions = []  # 存储动作序列
    # 运行多个episode进行测试
    for episode in range(EPISODES):
        done = False  # episode结束标志
        env.reset()  # 重置环境

        # 单episode内的决策循环
        while not done:
            actions = []  # 当前步的动作序列

            # 使用FIFO规则生成智能体执行顺序（这里固定为8个智能体）
            agents_id_sequence = sd_rules.FIFO_generate_agents_sequence(8)
            # 其他可选的规则：
            # agents_id_sequence = sd_rules.MLF_generate_agents_sequence(env.planes)  # 最长飞行时间优先
            # agents_id_sequence = sd_rules.LLF_generate_agents_sequence(env.planes)  # 最后期限优先

            # 为每个智能体选择动作
            for agent_id in agents_id_sequence:
                avail_actions = env.get_avail_agent_actions(agent_id)  # 获取可用动作
                current_plane_location = env.planes[agent_id].position  # 当前飞机位置
                # 根据启发式规则选择最优动作
                action = sd_rules.choose_action(agent_id, avail_actions, current_plane_location, sites_locations)
                actions.append(action)
                if action < 18:  # 有效的调度动作（0-17）
                    env.has_chosen_action(action, agent_id)  # 标记动作已被选择

            # 重新排序动作，因为环境要求按智能体ID顺序提供动作
            reorder_actions = [-1 for i in range(8)]
            for i in range(8):
                reorder_actions[agents_id_sequence[i]] = actions[i]

            # 执行动作
            _, done, info = env.step(reorder_actions)

        print(info["time"])  # 打印当前episode的总完成时间
    print(info['episodes_situation'])


if __name__ == "__main__":
    marl_agent_wrapper()
