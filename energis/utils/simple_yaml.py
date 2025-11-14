"""Minimal YAML loader for the limited configuration files used in the
project.

The real project depends on :mod:`yaml` from PyYAML, however that package
isn't available in the execution environment that powers the kata tests.
This module implements a very small subset of the YAML specification that is
powerful enough to parse the repository's configuration files.  It understands
mapping and list structures that are driven by indentation and supports the
scalar types that appear in the configs (strings, numbers, booleans and
``null``).

The implementation is intentionally conservative – it only accepts the syntax
we need which keeps the parser compact and easy to audit.  If a configuration
uses syntax that we don't understand we fail fast with a descriptive error so
that extending the parser is straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
from typing import Any, Iterable, List, Tuple


class YamlError(RuntimeError):
    """Exception raised when the lightweight parser encounters invalid input."""


def _strip_comments(line: str) -> str:
    """Remove YAML style ``#`` comments while respecting quoted substrings."""

    in_quote = False
    quote_char = ""
    escaped = False
    for idx, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if in_quote:
            if ch == quote_char:
                in_quote = False
            continue
        if ch in {'"', "'"}:
            in_quote = True
            quote_char = ch
            continue
        if ch == "#":
            return line[:idx]
    return line


def _normalise_lines(text: str) -> List[Tuple[int, str]]:
    """Return ``(indent, content)`` tuples for the relevant lines."""

    out: List[Tuple[int, str]] = []
    for raw in text.splitlines():
        cleaned = _strip_comments(raw).rstrip()
        if not cleaned:
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        if indent % 2:
            raise YamlError("Indentation must use multiples of two spaces.")
        out.append((indent, cleaned.strip()))
    return out


def _split_list(text: str) -> List[str]:
    """Split a YAML inline list (``[a, b, c]``) into the individual entries."""

    items: List[str] = []
    current = []
    depth = 0
    in_quote = False
    quote_char = ""
    escaped = False
    for ch in text:
        if escaped:
            current.append(ch)
            escaped = False
            continue
        if ch == "\\":
            current.append(ch)
            escaped = True
            continue
        if in_quote:
            current.append(ch)
            if ch == quote_char:
                in_quote = False
            continue
        if ch in {'"', "'"}:
            current.append(ch)
            in_quote = True
            quote_char = ch
            continue
        if ch == "[":
            depth += 1
            current.append(ch)
            continue
        if ch == "]":
            depth -= 1
            current.append(ch)
            continue
        if ch == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    if current:
        items.append("".join(current).strip())
    return [item for item in items if item]


def _parse_scalar(value: str) -> Any:
    """Parse the scalar ``value`` portion of a YAML statement."""

    value = value.strip()
    if value == "" or value == "~":
        return None
    lower = value.lower()
    if lower in {"null", "none"}:
        return None
    if lower in {"true", "yes"}:
        return True
    if lower in {"false", "no"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        return [_parse_scalar(part) for part in _split_list(value[1:-1])]
    # Try to interpret the scalar using Python's literal parser.  This covers
    # quoted strings as well as integers/floats in a robust fashion.  We catch
    # ValueError so that bare strings such as ``Import_Data.xlsx`` fall back to
    # plain text.
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    # Finally fall back to returning the string verbatim.
    return value


@dataclass
class _State:
    indent: int
    value: Any


def _parse_lines(lines: Iterable[Tuple[int, str]]) -> Any:
    """Convert the ``(indent, text)`` sequence into Python objects."""

    iterator = list(lines)
    idx = 0
    stack: List[_State] = [_State(indent=-2, value={})]

    while idx < len(iterator):
        indent, content = iterator[idx]
        idx += 1

        # Pop to the parent level if necessary
        while stack and indent < stack[-1].indent:
            stack.pop()
        if not stack:
            raise YamlError("Invalid indentation structure in YAML input.")

        parent = stack[-1].value

        if content.startswith("- "):
            if not isinstance(parent, list):
                raise YamlError("List item not associated with a list context.")
            item_text = content[2:].strip()
            if item_text:
                if ":" in item_text:
                    key, raw_val = item_text.split(":", 1)
                    key = key.strip()
                    raw_val = raw_val.strip()
                    new_dict: dict[str, Any] = {}
                    if raw_val:
                        new_dict[key] = _parse_scalar(raw_val)
                    else:
                        new_dict[key] = {}
                    parent.append(new_dict)
                    if idx < len(iterator) and iterator[idx][0] > indent:
                        stack.append(_State(iterator[idx][0], new_dict))
                else:
                    parent.append(_parse_scalar(item_text))
            else:
                # The list item opens a nested structure.
                next_indent = iterator[idx][0] if idx < len(iterator) else indent + 2
                if idx < len(iterator) and iterator[idx][0] > indent:
                    child_indent = iterator[idx][0]
                    if iterator[idx][1].startswith("- "):
                        new_list: List[Any] = []
                        parent.append(new_list)
                        stack.append(_State(child_indent, new_list))
                    else:
                        new_dict: dict[str, Any] = {}
                        parent.append(new_dict)
                        stack.append(_State(child_indent, new_dict))
                else:
                    parent.append(None)
            continue

        if ":" not in content:
            raise YamlError(f"Expected key/value pair, found: {content!r}")

        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        if raw_value == "":
            # Determine whether the nested structure is a list or a mapping by
            # peeking at the next relevant line.
            if idx < len(iterator) and iterator[idx][0] > indent:
                child_indent = iterator[idx][0]
                if iterator[idx][1].startswith("- "):
                    new_list = []
                    parent[key] = new_list
                    stack.append(_State(child_indent, new_list))
                else:
                    new_dict = {}
                    parent[key] = new_dict
                    stack.append(_State(child_indent, new_dict))
            else:
                parent[key] = {}
            continue

        parent[key] = _parse_scalar(raw_value)

    while len(stack) > 1:
        stack.pop()
    return stack[0].value


def loads(text: str) -> Any:
    """Parse ``text`` and return Python data structures."""

    lines = _normalise_lines(text)
    return _parse_lines(lines)


def load(path: str) -> Any:
    """Read ``path`` and parse the YAML contained within."""

    with open(path, "r", encoding="utf-8") as handle:
        return loads(handle.read())

