"""DISABLED (plan9 pre-registration, 10 Nov 2026): the public window is a pipeline check, never a score.

This script used to score a candidate on the public 5-minute window and log the result. It no longer reads the ground truth or
prints any score. The only permitted use of the window is scripts/final_check.py (two booleans, no scores).
"""
import sys

MESSAGE = ("public_read.py is disabled: the organisers' rule forbids using the public window to compare or tune versions "
           "(docs/reads/plan9-preregistration.md). Use scripts/final_check.py for the single pre-declared correctness check "
           "(prints two booleans only).")


def main():
    raise SystemExit(MESSAGE)


if __name__ == "__main__":
    main()
