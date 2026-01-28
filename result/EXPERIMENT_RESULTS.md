# 实验结果与复现步骤（boatschedule）

下面列出已生成的图像、含义说明，以及如何从代码运行脚本复现这些图片。所有时间图均使用 **1 仿真时间单元 = 5 分钟**。

目录结构（关键输出）
- `result/qmix/boatschedule/gantt.png` — per-robot 甘特图（纵轴 Robot1..Robot8，横轴 时间（分钟））  
- `result/qmix/boatschedule/gantt_per_job.png` — per-job 甘特图（纵轴 Job1..JobN，横轴 时间（分钟），色块内标注执行该任务的机器人编号，颜色按机器人编号着色）  
- `result/qmix/boatschedule/gantt_job_legend.png` — 甘特图中 `job_id -> 颜色` 的图例  
- `result/qmix/boatschedule/gantt_agent_legend.png` — 甘特图中 `robot -> 颜色` 的图例  
- `result/overview_boatschedule_completion_time.png` — 跨算法（目录）总完工时间对比（箱线图 + 每次运行散点），Y 轴单位：分钟  
- `result/overview_boatschedule_training_stability.png` — 训练稳定性（每次运行曲线 + 均值 + 95% CI），横轴：训练步（episode index），纵轴：episode reward  
- `result/overview_boatschedule_completion_time.csv`、`result/overview_boatschedule_training_stability.csv` — 对应 CSV 数据

图像说明（横纵坐标与含义）
- 甘特图（`gantt.png` / `gantt_per_job.png`）  
  - 横轴：时间（分钟），刻度值 = 仿真时间 * 5 分钟（你可以在图片下方读到“plotted = simulation_time * 5 min”）  
  - 纵轴：`gantt.png` 为机器人（Robot1..Robot8），`gantt_per_job.png` 为作业（Job1..JobN）  
  - 色块：表示在该时间段某机器人执行的任务段（事件格式：`(start_time, job_id, site_id, agent_id)`）。  
  - 色块内部标签：在 per-robot 图中为 `job_id`（任务编号）；在 per-job 图中为执行该段的 `robot` 编号（1-based）。  
  - 颜色映射：甘特图颜色使用 `matplotlib` 的 `tab20` colormap；在 `gantt_job_legend.png` 和 `gantt_agent_legend.png` 中分别展示 `job_id->color` 与 `robot->color` 的对应关系。

- 总工期对比（`overview_boatschedule_completion_time.png`）  
  - 横轴：算法（目录名，例如 `qmix`）  
  - 纵轴：总完工时间（分钟） — 每个点代表一次独立运行/一次评估 episode 的完工时间（脚本会在每个算法的 `result/{alg}/{map}/` 下递归查找 `times.txt`，或回退到 `my_data_and_graph/historydata/times.txt`）  
  - 图中展示：箱线图（含中位数、四分位）与散点（每次运行），用于观察泛化/稳定性

- 训练稳定性（`overview_boatschedule_training_stability.png`）  
  - 横轴：训练步（episode index，整型）  
  - 纵轴：回合累计奖励（episode reward）  
  - 图中展示：灰色为每次运行的独立曲线，蓝色为均值曲线并带 95% 置信区间（shaded area）

如何复现这些图（命令）
1. 生成甘特图（per-robot & per-job，单位 5 分钟/仿真单元）：

```bash
python scripts/generate_gantt_results.py
```

2. 生成甘特图的图例（job 与 robot 对应颜色）：

```bash
python scripts/generate_gantt_legends.py
```

3. 生成跨算法总完工时间对比（箱线图 + 散点）：

```bash
python scripts/generate_alg_comparison.py --map boatschedule --time_unit 5 --outdir result
```

该脚本会自动在 `result/{alg}/{map}/` 目录下查找 `times.txt` 文件并汇总。若只处理当前 QMIX 结果，它会回退使用 `my_data_and_graph/historydata/times.txt`。

4. 生成训练稳定性图（每次运行曲线 + 均值 + 95% CI）：

```bash
python scripts/generate_training_stability.py
```

该脚本会尝试加载 `episode_rewards_*.npy`（任意 `result` 子目录），若不存在则回退使用 `my_data_and_graph/historydata/accumulated_rewards.txt`。

备注与建议
- 时间单位：所有时间相关图像均使用 `time_unit=5` 分钟（如需改为 10 分钟，运行脚本时传参 `--time_unit 10` 给 `generate_alg_comparison.py`，或修改 `generate_gantt_results.py` 的 `time_unit_minutes` 调用）。  
- 胜率（win_rate）：当前评估逻辑依赖环境在 `info` 中的 `battle_won` 字段，当前 `ScheduleEnv` 并未提供该字段，因此 `win_rate` 常为 0。若需要把“完成即视为胜利”，建议在 `environment.py` 的 `step()` 返回的 `info` 中加入 `battle_won` 标志（我可以为你实施这一改动）。  
- 若你将来添加 MASA/DASA 等算法结果，只需把各算法输出放在 `result/{alg}/{map}/` 下，运行步骤 3 即可自动生成跨算法对比图。

如需，我可以：
- 把所有生成的图再做一次“可视化风格统一化”的处理（统一字体、加上标题/注释框、把 legend 嵌入到图片里），或  
- 修改 `ScheduleEnv`，在 episode 结束时设置 `info['battle_won']`（按“完成即胜利”或基于阈值）。  

完成于：自动化脚本已放在 `scripts/` 下。欢迎下一步指定我把哪类图片风格/注释再细化。

