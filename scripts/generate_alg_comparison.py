#!/usr/bin/env python3
"""
Generate total completion time comparison across algorithms.

Searches under `result/{alg}/{map}/` for `times.txt` files (recursively).
If none found for an algorithm and only one algorithm is being processed,
falls back to `my_data_and_graph/historydata/times.txt`.

Outputs:
 - result/overview_{map}_completion_time.png
 - result/overview_{map}_completion_time.csv
"""
import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def read_times_file(path):
    vals = []
    try:
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
    except Exception:
        pass
    return vals

def collect_times_for_alg(alg, map_name):
    base = os.path.join('result', alg, map_name)
    pattern = os.path.join(base, '**', 'times.txt')
    files = glob.glob(pattern, recursive=True)
    times = []
    for p in files:
        vals = read_times_file(p)
        # pick last positive value if exists
        pos_vals = [v for v in vals if v > 0]
        if pos_vals:
            times.append(pos_vals[-1])
    return times

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', default='boatschedule')
    parser.add_argument('--algorithms', nargs='*', default=None,
                        help='List of algorithm directory names under result/. If omitted, use all dirs in result/')
    parser.add_argument('--time_unit', type=float, default=5.0, help='minutes per simulation unit')
    parser.add_argument('--outdir', default='result')
    args = parser.parse_args()

    if args.algorithms:
        algs = args.algorithms
    else:
        # list directories under result/
        algs = [d for d in os.listdir('result') if os.path.isdir(os.path.join('result', d))]
    algs = sorted(algs)
    all_times = {}
    for alg in algs:
        times = collect_times_for_alg(alg, args.map)
        all_times[alg] = times

    # fallback: if only one alg selected and it has no data, try historydata/times.txt
    if len(algs) == 1 and (not all_times[algs[0]]):
        fallback = os.path.join('my_data_and_graph', 'historydata', 'times.txt')
        if os.path.exists(fallback):
            vals = read_times_file(fallback)
            all_times[algs[0]] = [v for v in vals if v > 0]

    # convert to minutes
    for alg in list(all_times.keys()):
        all_times[alg] = [v * args.time_unit for v in all_times[alg]]

    # prepare CSV
    out_csv = os.path.join(args.outdir, 'overview_{}_completion_time.csv'.format(args.map))
    os.makedirs(args.outdir, exist_ok=True)
    max_len = max((len(v) for v in all_times.values()), default=0)
    header = algs
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(max_len):
            row = []
            for alg in algs:
                lst = all_times.get(alg, [])
                row.append(lst[i] if i < len(lst) else '')
            writer.writerow(row)

    # plot boxplot + scatter of individual runs
    fig, ax = plt.subplots(figsize=(max(6, len(algs)*1.5), 6))
    data = [all_times.get(alg, []) for alg in algs]
    # boxplot
    ax.boxplot(data, labels=algs, showmeans=True)
    # overlay scatter
    for xi, lst in enumerate(data, start=1):
        xs = np.random.normal(xi, 0.04, size=len(lst))
        ax.scatter(xs, lst, alpha=0.7)
    ax.set_ylabel('Total completion time (minutes)')
    ax.set_xlabel('Algorithm')
    ax.set_title('Total completion time comparison (map={})'.format(args.map))
    out_png = os.path.join(args.outdir, 'overview_{}_completion_time.png'.format(args.map))
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print('Saved:', out_png)
    print('CSV saved:', out_csv)

if __name__ == '__main__':
    main()

