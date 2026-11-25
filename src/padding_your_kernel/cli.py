"""Command line for the padding analysis and local workbench."""

import argparse
import json
from pathlib import Path

from . import __version__
from .model import analyze_adjacent_columns
from .presets import analyze_preset, list_presets
from .server import serve


def main(argv=None):
    parser = argparse.ArgumentParser(prog="padding-your-kernel", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    web = commands.add_parser("serve", help="Start the local Web workbench")
    web.add_argument("--port", type=int, default=8766)
    web.add_argument("--research-dir", type=Path, help="Folder containing the Experiment 23 CUDA runner")
    web.add_argument("--results-dir", type=Path, default=Path("results/live"),
                     help="Output directory for new GPU runs (default: results/live)")
    commands.add_parser("presets", help="Print the available research configurations as JSON")
    preset = commands.add_parser("preset", help="Analyze a fixed benchmark configuration")
    preset.add_argument("name", choices=[item.name for item in list_presets()])
    custom = commands.add_parser("adjacent-columns", help="Analyze a declared regular FP32 access pattern")
    custom.add_argument("--width", type=int, required=True, help="Logical row width in float elements")
    custom.add_argument("--rows", type=int, required=True, help="Logical number of rows")
    custom.add_argument("--reserved-lines", type=int, default=1, help="Maximum other distinct same-set lines (0–3)")
    custom.add_argument("--base-alignment-bytes", type=int, default=128)
    args = parser.parse_args(argv)
    if args.command == "serve":
        if not 0 <= args.port <= 65535:
            parser.error("--port must be in 0–65535")
        try:
            serve(args.port, research_dir=args.research_dir, results_dir=args.results_dir)
        except OSError as error:
            parser.exit(1, f"Could not start the local workbench: {error}\n")
        return 0
    try:
        if args.command == "presets":
            body = {"presets": [item.to_dict() for item in list_presets()]}
        elif args.command == "preset":
            body = analyze_preset(args.name).to_dict()
        else:
            body = analyze_adjacent_columns(args.width, args.rows,
                                            reserved_lines=args.reserved_lines,
                                            base_alignment_bytes=args.base_alignment_bytes).to_dict()
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(body, ensure_ascii=True, indent=2))
    return 0
