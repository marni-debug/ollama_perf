#!/usr/bin/env python3

import csv
import os
import re
import subprocess
from datetime import datetime

LOGFILE = os.path.expanduser("~/ollama-perf/generations.csv")

os.makedirs(os.path.dirname(LOGFILE), exist_ok=True)

if not os.path.exists(LOGFILE):
    with open(LOGFILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "task",
            "prompt_tokens",
            "prompt_seconds",
            "prompt_tps",
            "eval_tokens",
            "eval_seconds",
            "generation_tps",
            "total_seconds",
            "total_tokens",
        ])

pattern_prompt = re.compile(
    r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens.*?([\d.]+) tokens per second"
)

pattern_eval = re.compile(
    r"eval time =\s+([\d.]+) ms /\s+(\d+) tokens.*?([\d.]+) tokens per second"
)

pattern_total = re.compile(
    r"total time =\s+([\d.]+) ms /\s+(\d+) tokens"
)

pattern_task = re.compile(r"task (\d+)")

prompt = None
evaluation = None
task = None

process = subprocess.Popen(
    ["journalctl", "-u", "ollama", "-f", "-o", "cat"],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
    bufsize=1,
)

for line in process.stdout:
    m = pattern_task.search(line)
    if m:
        task = m.group(1)

    m = pattern_prompt.search(line)
    if m:
        prompt = (
            float(m.group(1)) / 1000,
            int(m.group(2)),
            float(m.group(3)),
        )

    m = pattern_eval.search(line)
    if m:
        evaluation = (
            float(m.group(1)) / 1000,
            int(m.group(2)),
            float(m.group(3)),
        )

    m = pattern_total.search(line)
    if m and prompt and evaluation:
        total_seconds = float(m.group(1)) / 1000
        total_tokens = int(m.group(2))

        with open(LOGFILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                task,
                prompt[1],
                prompt[0],
                prompt[2],
                evaluation[1],
                evaluation[0],
                evaluation[2],
                total_seconds,
                total_tokens,
            ])

        print(
            f"{datetime.now():%H:%M:%S} "
            f"task={task} "
            f"prompt={prompt[2]:.1f} t/s "
            f"generation={evaluation[2]:.2f} t/s "
            f"tokens={evaluation[1]}"
        )

        prompt = None
        evaluation = None
        task = None
