import os
import ast
import pickle
import numpy as np
import matplotlib.pyplot as plt

def _load_scheduler_results_txt(path):
    """Load lines from scheduleresults.txt. Each line is expected to be a Python
    literal (list/tuple). We use ast.literal_eval to parse safely."""
    events = []
    if not os.path.exists(path):
        return events
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                val = ast.literal_eval(line)
                # val may be a list of tuples or nested; flatten if needed
                if isinstance(val, (list, tuple)):
                    for item in val:
                        if isinstance(item, (list, tuple)) and len(item) >= 4:
                            events.append(tuple(item[:4]))
                elif isinstance(val, dict):
                    # ignore
                    continue
            except Exception:
                # skip unparsable lines
                continue
    return events

def _load_process_pk(path):
    """Load pickled process data saved by other wrappers (list of schedules)."""
    events = []
    if not os.path.exists(path):
        return events
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        # data expected to be a list of schedule_processes; each is a list of tuples
        for schedule in data:
            if isinstance(schedule, (list, tuple)):
                for item in schedule:
                    if isinstance(item, (list, tuple)) and len(item) >= 4:
                        events.append(tuple(item[:4]))
    except Exception:
        return []
    return events

def generate_and_save_gantt(output_dir, out_filename="gantt.png"):
    """Generate and save a Gantt chart based on recorded scheduling events.

    Looks for:
    - ./my_data_and_graph/historydata/scheduleresults.txt
    - ./my_data_and_graph/pickles/process.pk

    Each event tuple format: (start_time, job_id, site_id, agent_id)
    If an event doesn't have an explicit end time we estimate it using the
    next start for the same agent or the median duration across events.
    """
    txt_path = os.path.join("my_data_and_graph", "historydata", "scheduleresults.txt")
    pk_path = os.path.join("my_data_and_graph", "pickles", "process.pk")

    events = []
    events += _load_scheduler_results_txt(txt_path)
    events += _load_process_pk(pk_path)

    if not events:
        raise FileNotFoundError("No scheduling events found in '{}' or '{}'".format(txt_path, pk_path))

    # events: list of (start_time, job_id, site_id, agent_id)
    # group by agent (agent_id)
    by_agent = {}
    for start_time, job_id, site_id, agent_id in events:
        try:
            start_time = float(start_time)
            agent_id = int(agent_id)
            job_id = int(job_id)
        except Exception:
            # best-effort: skip invalid rows
            continue
        by_agent.setdefault(agent_id, []).append((start_time, job_id, site_id, agent_id))

    # compute durations using next start_time per agent
    durations = []
    agent_bars = {}
    for agent_id, evs in by_agent.items():
        evs_sorted = sorted(evs, key=lambda x: x[0])
        bars = []
        for i, ev in enumerate(evs_sorted):
            start = ev[0]
            job = ev[1]
            if i + 1 < len(evs_sorted):
                end = evs_sorted[i + 1][0]
                if end <= start:
                    end = start + 1.0
            else:
                end = None
            bars.append({"start": start, "end": end, "job": job})
            if end is not None:
                durations.append(end - start)
        agent_bars[agent_id] = bars

    # global median duration fallback
    if durations:
        median_dur = float(np.median(np.array(durations)))
        if median_dur <= 0:
            median_dur = 1.0
    else:
        median_dur = 1.0

    # fill missing ends
    for agent_id, bars in agent_bars.items():
        for b in bars:
            if b["end"] is None:
                b["end"] = b["start"] + median_dur

    # Prepare plotting
    agents_sorted = sorted(agent_bars.keys())
    y_labels = ["Robot{}".format(a + 1) for a in agents_sorted]
    y_pos = np.arange(len(agents_sorted))

    # Collect unique job ids for color mapping
    job_ids = sorted({b["job"] for bars in agent_bars.values() for b in bars})
    cmap = plt.get_cmap("tab20")
    color_map = {job: cmap(i % 20) for i, job in enumerate(job_ids)}

    fig, ax = plt.subplots(figsize=(12, max(3, len(agents_sorted) * 0.35)))
    for idx, agent_id in enumerate(agents_sorted):
        bars = agent_bars[agent_id]
        for b in bars:
            start = b["start"]
            width = max(0.01, b["end"] - b["start"])
            job = b["job"]
            ax.barh(idx, width, left=start, height=0.6, color=color_map.get(job, "#333333"), edgecolor="k")
            ax.text(start + width * 0.02, idx, str(job), va='center', ha='left', color='white', fontsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time")
    ax.set_ylabel("Robots")
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, out_filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

