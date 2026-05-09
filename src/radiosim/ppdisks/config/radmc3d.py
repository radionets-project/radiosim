from typing import Any


def format_output_lines(inp: list[Any]) -> list[str]:
    output = list(map(lambda entry: str(entry) + "\n", inp))
    output[-1] = output[-1].removesuffix("\n")
    return output
