"""One command interface, with lazy imports for optional integrations."""

import argparse
import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="prediction-markets", description="Run and inspect agent prediction markets.")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    descriptions = {
        "run": "Start a fresh experiment from YAML",
        "resume": "Continue a stopped run with its saved configuration",
        "download": "Download a Modal run and generate its trace export",
        "trace": "Read agent messages, tools, and timing",
        "export": "Export actions, book history, and raw market objects",
        "plot": "Compare internal fills with Kalshi prices",
    }
    for name, description in descriptions.items():
        commands.add_parser(name, help=description, add_help=False)
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command, arguments = argv[0], argv[1:]
    if command == "run":
        from .runtime.runner import main as execute
    elif command == "resume":
        from .runtime.runner import main as execute
        resume_parser = argparse.ArgumentParser(prog="prediction-markets resume", description=descriptions[command])
        resume_parser.add_argument("run", help="Run ID, directory, or run.db; no config overrides")
        options = resume_parser.parse_args(arguments)
        arguments = ["--resume-from", options.run]
    elif command == "download":
        from .storage.download import main as execute
    elif command == "trace":
        from .analysis.trace import main as execute
    elif command == "export":
        from .analysis.export import main as execute
    elif command == "plot":
        from .analysis.prices import main as execute
    else:
        parser.error(f"unknown command: {command}")
    try:
        return execute(arguments)
    except (OSError, ValueError) as exc:
        print(f"{command} failed: {exc}", file=sys.stderr)
        return 1
