#!/usr/bin/env python3
"""Replay a match JSON log without importing or rerunning student code."""

import argparse
import json
import sys
import time
from pathlib import Path

if __package__:
    from .arena import nonnegative_float
    from .visualizer import render_board
else:
    from arena import nonnegative_float
    from visualizer import render_board


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--delay", type=nonnegative_float, default=0.2)
    args = parser.parse_args(argv)
    try:
        replay = json.loads(args.log.read_text(encoding="utf-8"))
        if replay["format_version"] != 1:
            raise ValueError("Unsupported replay format")
        names = [player["name"] for player in replay["players"]]
        print(render_board(replay["initial_state"], names))
        for turn in replay["turns"]:
            if args.delay:
                time.sleep(args.delay)
            print("\nTurn {} | P{} | {} | {}".format(
                turn["turn"], turn["player"], turn["status"], turn["action"]))
            if turn.get("error"):
                print(turn["error"])
            print(render_board(turn["state"], names))
        winner = replay["winner"]
        print("\n" + ("Draw" if winner is None else "Winner: P{} {}".format(winner, names[winner])))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.error("Cannot read replay: {}".format(exc))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
