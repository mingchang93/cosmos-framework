#!/usr/bin/env python3
"""Extract iteration, loss, and time from an SFT training log.

Parses iter_speed log lines like:
  [09-09 12:12:02|INFO|.../iter_speed.py:116:on_training_step_end] Iteration 5: Hit counter: 5/50 | Loss: 3.2779 | Time: 29.61s

Usage:
  python3 tools/parse_training_log.py <logfile> [more_logfiles...]
  cat <logfile> | python3 tools/parse_training_log.py -

Prints CSV rows (header: iteration,loss,time). Loss is 'nan' when the line
has no Loss: field.
"""
import csv
import re
import sys

_ITER_WARMUP_RE = re.compile(r"Iteration\s+(\d+):")
_ITER_SPEED_RE = re.compile(r"\b(\d+)\s*:\s*iter_speed\b")
_LOSS_RE = re.compile(
    r"Loss:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf)", re.IGNORECASE
)
_TIME_RE = re.compile(
    r"Time:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*s\b", re.IGNORECASE
)
_SPEED_RE = re.compile(
    r"iter_speed\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+seconds\s+per\s+iteration",
    re.IGNORECASE,
)


def parse_line(line: str):
    """Return (iteration, loss, time) for a matching line, else None.

    Handles the warmup line
      "Iteration 5: Hit counter: 5/50 | Loss: 3.2779 | Time: 29.61s"
    and the post-warmup speed line
      "51 : iter_speed 28.90 seconds per iteration | Loss: 3.1021 | ...".
    Time is the wall-clock seconds ("Time:") for warmup lines, else the
    seconds-per-iteration from the speed line.
    """
    m = _ITER_WARMUP_RE.search(line)
    if m:
        iteration = int(m.group(1))
    else:
        m = _ITER_SPEED_RE.search(line)
        if not m:
            return None
        iteration = int(m.group(1))
    lm = _LOSS_RE.search(line)
    loss = float(lm.group(1)) if lm else float("nan")
    tm = _TIME_RE.search(line)
    if tm:
        time = float(tm.group(1))
    else:
        sm = _SPEED_RE.search(line)
        time = float(sm.group(1)) if sm else float("nan")
    return iteration, loss, time


def _iter_lines(paths):
    for path in paths:
        if path == "-":
            yield from sys.stdin
        else:
            with open(path) as f:
                yield from f


def main(argv):
    paths = argv[1:] or ["-"]
    writer = csv.writer(sys.stdout)
    writer.writerow(["iteration", "loss", "time"])
    for line in _iter_lines(paths):
        row = parse_line(line)
        if row:
            writer.writerow(row)


if __name__ == "__main__":
    main(sys.argv)
