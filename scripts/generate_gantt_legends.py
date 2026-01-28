#!/usr/bin/env python3
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import MARL.common.plot_utils as plot_utils

def main():
    out_dir = os.path.join('result', 'qmix', 'boatschedule')
    os.makedirs(out_dir, exist_ok=True)

    txt_path = os.path.join("my_data_and_graph", "historydata", "scheduleresults.txt")
    pk_path = os.path.join("my_data_and_graph", "pickles", "process.pk")
    events = []
    events += plot_utils._load_scheduler_results_txt(txt_path)
    events += plot_utils._load_process_pk(pk_path)
    if not events:
        print("No events found for legend generation.")
        return

    job_ids = sorted({int(e[1]) for e in events})
    agent_ids = sorted({int(e[3]) for e in events})

    cmap = plt.get_cmap("tab20")
    job_color_map = {job: cmap(i % 20) for i, job in enumerate(job_ids)}
    agent_color_map = {agent: cmap(i % 20) for i, agent in enumerate(agent_ids)}

    # Job legend
    fig, ax = plt.subplots(figsize=(6, max(1, len(job_ids)*0.25)))
    patches = [mpatches.Patch(color=job_color_map[j], label=f'Job {j}') for j in job_ids]
    ax.axis('off')
    ax.legend(handles=patches, loc='center left', frameon=False, ncol=2)
    out_job = os.path.join(out_dir, 'gantt_job_legend.png')
    fig.tight_layout()
    fig.savefig(out_job, dpi=150)
    plt.close(fig)
    print("Saved job legend:", out_job)

    # Agent legend
    fig, ax = plt.subplots(figsize=(6, max(1, len(agent_ids)*0.25)))
    patches = [mpatches.Patch(color=agent_color_map[a], label=f'Robot {a+1}') for a in agent_ids]
    ax.axis('off')
    ax.legend(handles=patches, loc='center left', frameon=False, ncol=2)
    out_agent = os.path.join(out_dir, 'gantt_agent_legend.png')
    fig.tight_layout()
    fig.savefig(out_agent, dpi=150)
    plt.close(fig)
    print("Saved agent legend:", out_agent)

if __name__ == '__main__':
    main()

