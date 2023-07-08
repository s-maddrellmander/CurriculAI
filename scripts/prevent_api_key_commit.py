#!/usr/bin/env python3
import argparse
import re
import sys

API_KEY_REGEX = re.compile(
    r'([\'"])[a-zA-Z0-9]{30,}\1'
)  # Simple regex to match potential keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*", help="Filenames to run")
    args = parser.parse_args()

    for filename in args.filenames:
        with open(filename, "r") as file:
            for line_no, line in enumerate(file, start=1):
                if API_KEY_REGEX.search(line):
                    print(f"Potential API Key found in {filename} line {line_no}")
                    sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
