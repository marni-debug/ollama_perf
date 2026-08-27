#!/usr/bin/env python3
"""CLI entry: python prompt_compiler.py --fitness PATH --task PATH
             python prompt_compiler.py experiment --model MODEL --benchmark PATH ...
"""

from compiler.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
