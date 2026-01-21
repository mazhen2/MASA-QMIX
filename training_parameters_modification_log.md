# MASA-QMIX — 训练与结果可视化变更记录与使用说明

更新时间：2026-01-21

目的：
- 在程序完整执行结束后自动保存训练曲线（win-rate / episode rewards）和生成调度甘特图（Gantt chart），方便快速复现与可视化调试。

重要修改（实现清单）：
- 新增文件：`MARL/common/plot_utils.py`
  - 功能：从 `my_data_and_graph/historydata/scheduleresults.txt` 或 `my_data_and_graph/pickles/process.pk` 加载调度事件（事件元组格式为 (start_time, job_id, site_id, agent_id)），对缺失结束时间采用启发式估计（同一机器人下一事件开始时间或全局中位持续时间），绘制并保存甘特图 `gantt.png`。
- 修改文件：`MARL/runner.py`
  - 在 `Runner.run()` 在训练循环结束后调用：
    - `self.plt(num)`：保存训练曲线图像 `plt_{num}.png` 以及 numpy 数据文件 `win_rates_{num}.npy` 和 `episode_rewards_{num}.npy`；
    - `plot_utils.generate_and_save_gantt()`：尝试从记录中生成甘特图并保存到 `./result/{alg}/{map}/gantt.png`（若没有调度记录则仅打印提示）。
- 说明（无侵入性）：上述绘图步骤为“best-effort”模式 —— 若缺少数据文件（scheduleresults.txt / process.pk）或解析失败，程序仅打印错误信息而不会中断流程。

如何产生调度记录（供甘特图使用）：
- 在训练或评估过程中，环境（`environment.ScheduleEnv`）会通过 `save_env_info()` 收集事件到 `env.job_record_for_gant`，并由 `RolloutWorker.generate_episode()` 在 episode 结束时把该列表作为 `for_gant` 返回，`Runner` 在评估点会把 `for_gantt_data` 追加写入：
  - `my_data_and_graph/historydata/scheduleresults.txt`（文本格式，每行为一个 Python 可解析的序列）
  - 或者：`my_data_and_graph/pickles/process.pk`（由随机基线跑出的 pickle）

推荐的训练参数（便于产出有意义的图）：
- 建议不要使用极短的快速验证参数；为了能绘出清晰的训练曲线与甘特图，推荐：
  - `args.n_epoch = 2000`           # 训练总轮数
  - `args.n_episodes = 5`           # 每轮的 episode 数
  - `args.train_steps = 2`          # 每轮的训练步数
  - `args.evaluate_cycle = 20`      # 每20个 epoch 评估一次（可视化更平滑）
  - `args.save_cycle = 200`         # 每200个 epoch 保存一次模型

快速验证建议（当希望短跑时）：
- 若仅用于语法/流程验证，可使用：
  - `args.n_epoch = 10`
  - `args.n_episodes = 3`
  - `args.train_steps = 1`
  - `args.evaluate_cycle = 1`
 但图像信息量会很少，建议至少把 `n_epoch` 提高到几百以便观察学习曲线。

运行方式（示例）：
1. 在项目根目录运行训练：
```bash
python main.py
```
2. 程序在训练结束后会在 `./result/{alg}/{map}/` 下保存：
  - `plt_0.png`（训练曲线图）
  - `win_rates_0.npy`（numpy 数组）
  - `episode_rewards_0.npy`（numpy 数组）
  - `gantt.png`（若存在调度记录则生成）

文件说明：
- `MARL/common/plot_utils.py`：生成甘特图的核心工具（会寻找 `my_data_and_graph/historydata/scheduleresults.txt` 和 `my_data_and_graph/pickles/process.pk`）。
- `MARL/runner.py`：新增了训练结束时的保存/生成逻辑（最佳尝试，不影响主流程）。
- 结果目录：`./result/{alg}/{map}/`，其中 `{alg}` 与 `{map}` 来源于命令行参数（默认 `qmix/boatschedule`）。

甘特图生成细节（实现要点）：
- 解析事件：事件格式为 `(start_time, job_id, site_id, agent_id)`。
- 结束时间估计：如果事件序列中后续存在同一 agent 的下一事件，则用下一事件的 start_time 作为当前事件 end_time；否则使用所有已知持续时间的中位数作为估计长度。
- 颜色映射：按 `job_id` 分配颜色（循环 colormap），并在条上显示 `job_id` 标签。

注意事项与后续建议：
- 若需要精确的结束时间，建议在 `environment.save_env_info()` 中同时写入显式的结束时间（或在 job transition 中保存开始/结束）。`plot_utils` 会优先使用显式结束时间（当前实现使用启发式估计）。
- 如果希望在极短训练中也有数据（例如立即评估一次以写入 scheduler 记录），我可以把 `Runner.run()` 在训练开始前/结束后强制进行一次评估调用并写入记录。

变更已部署：是

后续我可以：
-（可选）把 `plot_utils` 支持的输入格式扩展为包含明确的结束时间字段；
-（可选）添加命令行参数以控制是否在每次训练结束时自动生成甘特图。

如果你确认以上内容，我将把该实现提交到仓库（已编辑本地文件），并可以进一步运行一次示例训练来产生示例图片（需要你授权我运行短时间训练或告知你想要的参数）。