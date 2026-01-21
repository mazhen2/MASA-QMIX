"""
船只调度环境 (Boat Schedule Environment)

该环境模拟了船只保障任务的调度问题，是一个多智能体强化学习环境。
在这里：
- plane（飞机）对应job（作业任务）
- site（站位）对应machine/workstation（工作站）

环境基于OpenAI Gym框架实现，支持多智能体的协同调度决策。
"""

from utils.site import Sites      # 站位管理类
from utils.job import Jobs        # 作业管理类
from utils.task import Task       # 任务管理类
from utils.plane import Planes    # 飞机管理类
from utils import util            # 工具函数
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import math


class ScheduleEnv(gym.Env):
    """
    船只调度环境类

    继承自OpenAI Gym的Env类，实现多智能体船只调度问题的环境模拟。
    智能体需要协作决定如何分配船只到不同的保障站位，以最小化总完成时间。
    """
    environment_name = "Boat Schedule"

    def __init__(self):
        """
        初始化船只调度环境

        设置环境的基本参数、状态变量和动作空间。
        """
        # 环境组件初始化
        self.sites = []        # 站位列表
        self.jobs = []         # 作业列表
        self.task = []         # 任务序列
        self.planes_obj = Planes()  # 飞机对象管理器
        self.planes = []       # 飞机列表

        # 环境状态变量
        self.state = [[]]      # 当前环境状态
        self.done = False      # episode结束标志
        self.state_left_time = []     # 各站位剩余处理时间
        self.episode_time_slice = []  # 每个step消耗的时间序列
        self.plane_speed = 0   # 飞机飞行速度

        # 初始化环境参数
        self.initialize()

        # 动作空间定义：
        # 0-17: 前往指定站位 (len(self.sites)-1 = 17)
        # 18: 由于资源冲突需要等待
        # 19: 处于正忙（加工）动作
        # 20: 已经完成了动作
        # 注：19、20状态不参与训练决策
        self.action_space = spaces.Discrete(len(self.sites) + 3)  # 离散动作空间

        # 环境标识
        self.id = "Boat Schedule"

        # 环境参数（用于Gym兼容性）
        self.reward_threshold = -1000  # 奖励阈值
        self.trials = 50               # 类似于最大步数

        self.job_record_for_gant = []  # 用于存储调度中间过程四元组

        self.sites_state_global = None  # this para is utilized to indicate the current idle sites and their processing jobs

        # 一个全局状态，一个观测
        self.state4marl = None  # 维护全局state的变量
        self.obs4marl = None

    def initialize(self):
        """
        初始化环境的所有组件和状态变量

        创建站位、作业、任务和飞机的实例，并设置初始状态。
        """
        # 创建环境组件对象
        sites_obj = Sites()
        self.sites_obj = sites_obj
        jobs_obj = Jobs()
        task_obj = Task()
        self.planes_obj = Planes()

        # 获取组件列表
        self.sites = sites_obj.sites_object_list    # 站位对象列表
        self.jobs = jobs_obj.jobs_object_list       # 作业对象列表
        self.task = task_obj.simple_task_object     # 任务序列
        self.planes = self.planes_obj.planes_object_list  # 飞机对象列表

        # 初始化环境状态
        # 状态格式：[当前占用飞机ID, 资源可用性向量]
        # 9表示空闲，资源向量表示该站位能处理的作业类型（1=能处理，0=不能处理）
        self.state = [[9, [1 if j in self.sites[i].resource_ids_list else 0 for j in range(9)]]
                      for i in range(len(self.sites))]

        # 全局站位状态：-1表示未被安排保障任务
        self.sites_state_global = [-1 for i in range(len(self.sites))]

        # 用于记录甘特图数据的四元组：(开始时间, 作业ID, 站位ID, 飞机ID)
        self.job_record_for_gant = []

        # 重置环境控制变量
        self.done = False
        self.state_left_time = np.array([0 for i in range(len(self.sites))])  # 各站位剩余处理时间
        self.episode_time_slice = []    # 记录每个step的时间消耗
        self.plane_speed = self.planes_obj.plane_speed  # 飞机飞行速度

        # MARL相关变量初始化
        self.obs4marl = [[] for i in range(len(self.planes))]  # 各智能体的观测
        self.current_finishing_jobs = 0  # 当前已完成作业数
        self.step_count = 0              # 当前step计数

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        重置环境到初始状态

        返回：
            state: 全局状态向量，用于MARL算法
        """
        self.initialize()  # 重新初始化所有参数

        # 构造状态信息字典
        info = {
            "sites": [[self.sites[i].absolute_position,    # 站位绝对位置
                       self.state[i][0],                   # 当前占用飞机ID
                       self.state[i][1]                    # 资源可用性向量
                       ] for i in range(len(self.sites))],
            "planes": [[self.planes[i].left_job[0].index_id,                    # 当前作业ID
                        self.jobs[self.planes[i].left_job[0].index_id].time_span, # 作业处理时间
                        len(self.planes[i].left_job)                            # 剩余作业数量
                        ] if len(self.planes[i].left_job) != 0
                       else [9, 0, len(self.planes[i].left_job)]  # 空闲状态
                       for i in range(len(self.planes))],
            "planes_obj": self.planes
        }

        state = self.conduct_state(info)  # 构造全局状态向量
        return state

    def conduct_state(self, info):
        """
        构造全局状态向量和局部观测

        Args:
            info: 包含sites和planes信息的字典

        Returns:
            numpy.array: 全局状态向量
        """
        res = []
        temp = []

        for eve in info["sites"]:
            # res.append(eve[1])
            temp.append(eve[1])

        for eve in info["sites"]:

            res += eve[2]

        for i, eve in enumerate(info["planes"]):
            res += [eve[2]]
            temp_obs = []
            for l, eve_1 in enumerate(temp):
                if self.planes[i].left_job == []:
                    temp_obs.append(0)
                else:
                    if eve_1 == 9 and self.planes[i].left_job[0].index_id in self.sites[l].resource_ids_list:

                        temp_obs.append(util.count_path_on_road(self.planes[i].position, self.sites[l].absolute_position, self.plane_speed)/40)  # 代表处于空闲状态,下一步飞机可以去
                    else:
                        temp_obs.append(0)  # 否则处于加工状态，下一步飞机不能去
            self.obs4marl[i] = temp_obs + [eve[0], eve[2], eve[1]]

        current_working_plane_ids = []
        for eve in self.state:
            if eve[0] != 9:
                current_working_plane_ids.append(eve[0])
        if current_working_plane_ids == []:
            for k in range(len(self.obs4marl)):
                self.obs4marl[k].append(0)
        else:
            for k in range(len(self.obs4marl)):
                if k in current_working_plane_ids:
                    # 正忙状态下的观测也进行处理为除了最后三位都是0
                    self.obs4marl[k] = [0 if kk < len(self.obs4marl[k])-2 else self.obs4marl[k][kk] for kk in range(len(self.obs4marl[k]))]
                    self.obs4marl[k].append(1)
                else:
                    self.obs4marl[k].append(0)
        obslen = len(self.obs4marl[0])
        zero_obs = [0 for i in range(obslen)]


        for i, plane in enumerate(self.planes):
            if len(plane.left_job) == 0:
                self.obs4marl[i] = zero_obs

        self.state4marl = np.array(res)
        return np.array(res)

    def check_inflict_action(self, action):
        res = []

        for eve in action:
            if eve == len(self.sites) or eve == 19 or eve == 20:
                res.append(eve)
            else:
                if eve not in res:
                    res.append(eve)
                else:
                    raise Exception("sloppy error in actions", action)
                    # assert False
        return res

    # 进行动作的替换
    def action_replace(self, action):
        res = []
        real_conflict_num = 0
        for eve in action:
            if eve == 20 or eve == 19:
                res.append(18)
            else:
                if eve == 18:
                    real_conflict_num += 1

                res.append(eve)
        return res, real_conflict_num

    def step(self, action):
        """
        执行一步环境交互

        Args:
            action: 所有智能体的动作列表

        Returns:
            tuple: (reward, done, info)
        """
        self.step_count += 1

        # 预处理动作：将无效动作(20)替换为等待动作(18)
        action, real_conflict_num = self.action_replace(action)

        # 初始化变量
        count_break_rules = 0  # 规则违反计数
        assert len(action) == len(self.planes)  # 确保动作数量与智能体数量匹配

        rewards = [0 for eve in action]           # 各智能体的奖励
        max_time_on_roads = [0 for eve in action] # 各智能体的最大路途时间
        count_for_reward = 0                      # 用于奖励计算的计数
        action = self.check_inflict_action(action)
        time_span_increase = np.array([0 for eve in self.sites])
        for i, site_id in enumerate(action):
            if site_id == len(self.sites):
                pass
            else:  # 安排保障任务
                if self.planes[i].left_job[0].index_id in self.sites[site_id].resource_ids_list:  # 如果选择的战位有需要的保障资源
                    # time_on_road为0代表其留在了原地加工
                    time_on_road = util.count_path_on_road(self.planes[i].position,
                                                           self.sites[site_id].absolute_position.tolist(), self.plane_speed)

                    if type(site_id) == int:
                        self.save_env_info(
                            (sum(self.episode_time_slice), self.planes[i].left_job[0].index_id, site_id, i))
                    else:
                        self.save_env_info((sum(self.episode_time_slice), self.planes[i].left_job[0].index_id, site_id.item(), i))
                    temp_time = self.planes[i].execute_task(self.planes[i].left_job[0], self.sites[site_id])

                    time_span_increase[site_id] = temp_time + time_on_road
                    # self.state[site_id][0] = i  # 表示这个站位已经被占据了
                    count_for_reward += 1
                    # 为构造每个飞机的reward存储maxtime，方便归一化
                    max_time_on_roads[i] = time_on_road


                else:
                    raise Exception("不合理的动作没有mask", self.sites_state_global, i, site_id,action,self.planes[i].left_job[0].index_id,
                                    self.sites[site_id].resource_ids_list, self.state)

        real_did = 0
        for eve in action:
            if eve < 18:
                real_did += 1

        for i, site_id in enumerate(action):
            if site_id == len(self.sites):  # 代表这个飞机不安排保障任务

                rewards[i] = - 30  # 只传入因为资源冲突而等待的地方
            else:  # 安排保障任务
                if rewards[i] == 0:
                    # rewards[i] = -(max_time_on_roads[i]+0.1)/(max(max_time_on_roads)+0.1)-real_conflict_num
                    rewards[i] = -(max_time_on_roads[i]+0.1)/(max(max_time_on_roads)+0.1)

                else:
                    pass

        # 开始更新当前的状态剩余时间
        self.state_left_time = self.state_left_time + time_span_increase

        min_time = util.min_but_zero(self.state_left_time)
        # print("haoshi：", min_time)
        self.episode_time_slice.append(min_time)  # 这个step消耗的时间
        self.state_left_time = util.advance_by_min_time(min_time, self.state_left_time)  # step推进

        # 更新状态,主要是检查哪些状态用完了
        # state transition 2
        for i, eve_time in enumerate(self.state_left_time):
            if eve_time == 0:
                self.sites_state_global[i] = -1  # 更新做完的sites
                self.sites_obj.update_site_resources(self.sites_state_global)
                self.state[i][0] = 9
                self.state[i][1] = [1 if j in self.sites[i].resource_ids_list else 0 for j in range(9)]
            else:
                assert self.state[i][0] != 9  # 不为9的一定被占据了

        # 判断当前episode是否完成了
        is_all_done = [-1 for eve in self.planes]
        for i, plane in enumerate(self.planes):
            if len(plane.left_job) == 0:
                is_all_done[i] = 0
        # self.current_finishing_jobs = sum(is_all_done) + len(is_all_done) - self.current_finishing_jobs
        if sum(is_all_done) == 0:
            self.done = True
        else:
            self.done = False


        left_jobs, all_jobs = self.planes_obj.count_jobs()
        if self.done:
            reward = 6000 / (sum(self.episode_time_slice) + max(self.state_left_time))
            # print(11, reward)

        else:
            reward = real_did - self.step_count/60 - real_conflict_num*2

        info = {
            "sites": [[self.sites[i].absolute_position,
                       self.state[i][0],
                       self.state[i][1]
                       ] for i in range(len(self.sites))],
            "planes": [[self.planes[i].left_job[0].index_id,
                        len(self.planes[i].site_history),
                        len(self.planes[i].left_job)
                        ] if len(self.planes[i].left_job) != 0
                       else [
                        9,
                        len(self.planes[i].site_history),
                        len(self.planes[i].left_job)
                    ]for i in range(len(self.planes))],
            "planes_obj": self.planes
        }
        state = self.conduct_state(info)
        # print("min_time:", min_time)
        # print("left_time:", self.state_left_time)
        return reward, self.done, {"time": sum(self.episode_time_slice)+max(self.state_left_time),
                                          "left": self.state_left_time,
                                          "original_state": self.state,
                                          "planes_obj": self.planes,
                                          "rewards": rewards,
                                          "count_break_rules": count_break_rules,
                                          "sites_state_global": self.sites_state_global,
                                   "episodes_situation": self.job_record_for_gant
                                   }

    def get_avail_agent_actions(self, agent_id):
        """
        获取指定智能体的可用动作

        Args:
            agent_id: 智能体ID

        Returns:
            list: 可用动作掩码，1表示可用，0表示不可用
                 格式：[站位动作(0-17), 等待(18), 忙碌(19), 完成(20)]
        """
        # 检查飞机是否处于正忙状态（正在加工）
        for eve in self.state:
            if agent_id == eve[0]:  # 该飞机正在某站位工作
                return [0 for i in range(18)] + [0, 1, 0]  # 只有忙碌动作(19)可用

        # 如果飞机准备进行下一步操作
        res = [0 for eve in self.sites_state_global]  # 初始化站位动作可用性

        for i, eve in enumerate(self.sites_state_global):
            if eve == -1:  # 该站位空闲
                if len(self.planes[agent_id].left_job) != 0:
                    # 检查该飞机下一个任务是否能在该站位处理
                    if self.planes[agent_id].left_job[0].index_id in self.sites[i].resource_ids_list:
                        res[i] = 1  # 该站位动作可用
                else:  # 该飞机已完成所有任务
                    return [0 for i in range(18)] + [0, 0, 1]  # 只有完成动作(20)可用

        return res + [1, 0, 0]  # 返回：[站位动作, 等待动作(18), 忙碌(19), 完成(20)]

    # state transition 1
    def has_chosen_action(self, action_id, agent_id):
        assert self.planes[agent_id].left_job != []
        # print(action_id, agent_id)
        self.sites_state_global[action_id] = self.planes[agent_id].left_job[0].index_id  # 更新战位状态信息
        # 更新总资源列表的状态-这块不能马虎，注意这里是在选择合理的动作而不是已经做了动作
        self.sites_obj.update_site_resources(self.sites_state_global)
        self.state[action_id][0] = agent_id  # 表示这个站位已经被占据了,更新状态
        self.state[action_id][1] = [1 if j in self.sites[action_id].resource_ids_list else 0 for j in range(9)]  # 更新资源抢占状态


    def save_env_info(self, job_transition):
        self.job_record_for_gant.append(job_transition)

    def get_state(self):
        assert self.state4marl is not None
        return self.state4marl

    def get_obs(self):
        # assert self.obs4marl is not None and self.obs4marl != []
        agents_obs = [self.get_obs_agent(i) for i in range(len(self.planes))]
        return agents_obs

    def get_obs_agent(self, agent_id):
        return self.obs4marl[agent_id]

    def get_env_info(self):
        """
        获取环境的基本信息，用于配置MARL算法参数

        Returns:
            dict: 环境信息字典
                - n_actions: 动作空间大小
                - n_agents: 智能体数量
                - state_shape: 全局状态维度
                - obs_shape: 局部观测维度
                - episode_limit: 单个episode的最大步数
        """
        return {
            "n_actions": len(self.sites) + 3,  # 动作空间：站位动作 + 等待/忙碌/完成动作
            "n_agents": len(self.planes),       # 智能体数量（飞机数量）
            "state_shape": len(self.get_state()),     # 全局状态向量维度
            "obs_shape": len(self.get_obs()[0]),      # 单个智能体观测维度
            "episode_limit": 80  # episode最大长度，超过此长度未完成会报错
        }
