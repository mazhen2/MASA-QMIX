#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_episode_rewards_from_npy(paths):
    runs = []
    for p in paths:
        try:
            arr = np.load(p)
            # ensure 1D
            arr = np.asarray(arr).reshape(-1)
            runs.append(arr)
        except Exception:
            continue
    return runs

def load_accumulated_rewards_txt(path):
    vals = []
    if not os.path.exists(path):
        return vals
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                v = float(line)
                vals.append(v)
            except Exception:
                continue
    # treat the whole file as a single run
    return [np.array(vals)]

def main():
    outdir = os.path.join('result')
    os.makedirs(outdir, exist_ok=True)
    map_name = 'boatschedule'

    # search for episode_rewards_*.npy under result/
    npy_paths = glob.glob(os.path.join('result', '**', 'episode_rewards_*.npy'), recursive=True)
    runs = load_episode_rewards_from_npy(npy_paths)

    # fallback to historydata/accumulated_rewards.txt if no npy found
    if not runs:
        fallback = os.path.join('my_data_and_graph', 'historydata', 'accumulated_rewards.txt')
        runs = load_accumulated_rewards_txt(fallback)

    if not runs:
        print('No episode rewards data found.')
        return

    # align runs by padding with NaN to same length
    maxlen = max(len(r) for r in runs)
    data = np.full((len(runs), maxlen), np.nan)
    for i, r in enumerate(runs):
        data[i, :len(r)] = r

    x = np.arange(maxlen)  # training step = episode index

    fig, ax = plt.subplots(figsize=(10, 5))
    # plot individual runs
    for i in range(data.shape[0]):
        ax.plot(x, data[i], color='gray', alpha=0.4, linewidth=0.8)

    # compute mean and 95% CI ignoring NaNs
    mean = np.nanmean(data, axis=0)
    sem = np.nanstd(data, axis=0) / np.sqrt(np.sum(~np.isnan(data), axis=0))
    ci = 1.96 * sem

    ax.plot(x, mean, color='blue', linewidth=1.5, label='mean')
    ax.fill_between(x, mean - ci, mean + ci, color='blue', alpha=0.2, label='95% CI')

    ax.set_xlabel('Training step (episode index)')
    ax.set_ylabel('Episode reward')
    ax.set_title('Training stability — rewards vs training steps (map={})'.format(map_name))
    ax.legend()
    out_png = os.path.join(outdir, 'overview_{}_training_stability.png'.format(map_name))
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # save CSV of mean and CI
    out_csv = os.path.join(outdir, 'overview_{}_training_stability.csv'.format(map_name))
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'mean', 'ci_lower', 'ci_upper'])
        for i in range(maxlen):
            writer.writerow([i, float(mean[i]) if not np.isnan(mean[i]) else '', 
                             float(mean[i] - ci[i]) if not np.isnan(mean[i]) else '',
                             float(mean[i] + ci[i]) if not np.isnan(mean[i]) else ''])

    print('Saved:', out_png)
    print('CSV saved:', out_csv)

if __name__ == '__main__':
    main()

