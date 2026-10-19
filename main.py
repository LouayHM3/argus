"""Command-line entry point for ARGUS-Micro."""

from experiments.run_all import parse_args, run


if __name__ == "__main__":
    run(parse_args())
