__all__ = ["Parser"]


class Parser:
    def __init__(self, bool_true: str = "yes", bool_false: str = "no"):
        self._type_parsers = [
            BoolParser(true_val=bool_true, false_val=bool_false),
            TypeParser(parse_type=int),
            TypeParser(parse_type=float),
            TypeParser(parse_type=str),
        ]

    def parse(self, value: str | None):
        if value is None:
            return None

        for parser in self._type_parsers:
            try:
                return parser.parse(value=value)
            except Exception:
                continue


class BoolParser:
    def __init__(self, true_val: str, false_val: str):
        self._true_val: str = true_val
        self._false_val: str = false_val

    def parse(self, value: str) -> bool:
        true_check = value == self._true_val
        false_check = value == self._false_val

        if true_check:
            return True
        if false_check:
            return False

        if not true_check and not false_check:
            raise ValueError(
                f"Invalid boolean value: '{value}'! "
                f"Valid values: {self._true_val} (True) or {self._false_val} (False)"
            )


class TypeParser:
    def __init__(self, parse_type: type):
        self._parse_type = parse_type

    def parse(self, value: str):
        return self._parse_type(value)
