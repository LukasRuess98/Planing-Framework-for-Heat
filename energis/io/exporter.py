"""Utilities for exporting scenario inputs and optimization results."""

from __future__ import annotations

from collections import OrderedDict
from datetime import date, datetime
from numbers import Number
from typing import Iterable, Mapping, Sequence
import json

from energis.utils.timeseries import TimeSeriesTable

try:  # pragma: no cover - optional dependency
    from openpyxl import Workbook
    HAVE_OPENPYXL = True
except Exception:  # pragma: no cover
    Workbook = None
    HAVE_OPENPYXL = False

__all__ = ["write_timeseries_csv", "write_excel_workbook", "HAVE_OPENPYXL"]


def _ensure_lengths(data: Mapping[str, Sequence[float]], expected: int) -> None:
    for key, values in data.items():
        if len(values) != expected:
            raise ValueError(f"Serie {key!r} hat Länge {len(values)} statt {expected}")


def write_timeseries_csv(path: str, table: TimeSeriesTable, extra: Mapping[str, Sequence[float]]) -> None:
    """Write input and result time series to a semicolon separated CSV file."""

    n = len(table)
    _ensure_lengths(extra, n)

    columns = ["timestamp"] + table.columns + list(extra.keys())
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(";".join(columns) + "\n")
        for idx, ts in enumerate(table.index):
            base = [ts.isoformat() if isinstance(ts, datetime) else str(ts)]
            base.extend(_fmt_value(table[col][idx]) for col in table.columns)
            base.extend(_fmt_value(extra[name][idx]) for name in extra)
            handle.write(";".join(base) + "\n")


def _fmt_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return ("{0:.10f}".format(value)).rstrip("0").rstrip(".") or "0"
    return str(value)


def _normalize_excel_value(value: object) -> object:
    """Convert arbitrary values into a representation accepted by openpyxl."""

    if value is None:
        return None

    if isinstance(value, (str, Number, bool)):
        return value

    if isinstance(value, datetime):
        return value

    if isinstance(value, date):
        # openpyxl expects datetime for date-like objects
        return datetime.combine(value, datetime.min.time())

    iso_formatter = getattr(value, "isoformat", None)
    if callable(iso_formatter):
        try:
            return iso_formatter()
        except Exception:
            pass

    return str(value)


def _append_table(ws, headers: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    ws.append([_normalize_excel_value(header) for header in headers])
    for row in rows:
        ws.append([_normalize_excel_value(value) for value in row])


def write_excel_workbook(
    path: str,
    table: TimeSeriesTable,
    series: Mapping[str, Sequence[float]],
    summary_sections: Mapping[str, Mapping[str, object]],
    metadata_sections: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    """Create an Excel workbook containing inputs, results and summary tables."""

    if not HAVE_OPENPYXL:
        raise RuntimeError(
            "openpyxl ist nicht installiert – Excel-Export ist daher nicht möglich. Bitte openpyxl nachinstallieren."
        )

    n = len(table)
    _ensure_lengths(series, n)

    wb = Workbook()

    input_ws = wb.active
    input_ws.title = "input_timeseries"
    _append_table(
        input_ws,
        ["timestamp"] + table.columns,
        (
            [table.index[i].isoformat() if isinstance(table.index[i], datetime) else str(table.index[i])] +
            [table[col][i] for col in table.columns]
            for i in range(n)
        ),
    )
    input_ws.freeze_panes = "B2"

    results_ws = wb.create_sheet("results_timeseries")
    series_headers = ["timestamp"] + list(series.keys())
    _append_table(
        results_ws,
        series_headers,
        (
            [table.index[i].isoformat() if isinstance(table.index[i], datetime) else str(table.index[i])] +
            [series[name][i] for name in series]
            for i in range(n)
        ),
    )
    results_ws.freeze_panes = "B2"

    summary_ws = wb.create_sheet("summary")
    for section, metrics in summary_sections.items():
        summary_ws.append([_normalize_excel_value(section)])
        for key, value in metrics.items():
            summary_ws.append(
                [_normalize_excel_value(key), _normalize_excel_value(value)]
            )
        summary_ws.append([])

    if metadata_sections:
        meta_ws = wb.create_sheet("metadata")
        for section, entries in metadata_sections.items():
            meta_ws.append([_normalize_excel_value(section)])
            for key, value in entries.items():
                if isinstance(value, (dict, list)):
                    normalized_value = json.dumps(value, ensure_ascii=False)
                else:
                    normalized_value = _normalize_excel_value(value)
                meta_ws.append(
                    [_normalize_excel_value(key), _normalize_excel_value(normalized_value)]
                )
            meta_ws.append([])

    wb.save(path)
