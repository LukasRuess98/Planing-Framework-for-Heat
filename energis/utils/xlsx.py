"""Minimal helpers for reading and writing the tiny Excel files used in tests.

The implementation understands enough of the XLSX file format to round-trip
simple worksheets consisting of a single sheet with plain numbers, text and
datetimes.  It intentionally avoids the complexity of a full Excel parser –
the goal is merely to supply the functionality that used to be covered by the
``pandas`` dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import itertools
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, List, Sequence
from xml.sax.saxutils import escape
import zipfile

EXCEL_EPOCH = datetime(1899, 12, 30)
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
DATE_NUMFMT_IDS = {
    14,
    15,
    16,
    17,
    22,
    27,
    30,
    36,
    50,
    57,
    58,
    164,
    165,
}


def column_index_to_letter(index: int) -> str:
    letters = []
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters.append(chr(65 + remainder))
    return "".join(reversed(letters))


def column_letter_to_index(letter: str) -> int:
    acc = 0
    for ch in letter:
        acc = acc * 26 + (ord(ch.upper()) - 64)
    return acc - 1


def excel_number_to_datetime(value: float) -> datetime:
    return EXCEL_EPOCH + timedelta(days=float(value))


def datetime_to_excel_number(value: datetime) -> float:
    delta = value - EXCEL_EPOCH
    return delta.total_seconds() / 86400.0


def _load_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    try:
        xml = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(xml)
    result = []
    for si in root.findall(f"{NS}si"):
        text = "".join(t.text or "" for t in si.findall(f".//{NS}t"))
        result.append(text)
    return result


def _load_date_styles(zf: zipfile.ZipFile) -> set[int]:
    try:
        xml = zf.read("xl/styles.xml")
    except KeyError:
        return set()
    root = ET.fromstring(xml)
    custom_formats: Dict[int, str] = {}
    num_fmts = root.find(f"{NS}numFmts")
    if num_fmts is not None:
        for num in num_fmts.findall(f"{NS}numFmt"):
            num_id = int(num.attrib.get("numFmtId", "0"))
            fmt_code = num.attrib.get("formatCode", "")
            custom_formats[num_id] = fmt_code
    date_styles: set[int] = set()
    cell_xfs = root.find(f"{NS}cellXfs")
    if cell_xfs is None:
        return date_styles
    for idx, xf in enumerate(cell_xfs.findall(f"{NS}xf")):
        num_fmt_id = int(xf.attrib.get("numFmtId", "0"))
        fmt = custom_formats.get(num_fmt_id, "")
        if num_fmt_id in DATE_NUMFMT_IDS or any(token in fmt for token in ("yy", "dd", "mm")):
            date_styles.add(idx)
    return date_styles


def _iter_rows(sheet_root: ET.Element) -> Iterable[List[ET.Element]]:
    for row in sheet_root.findall(f".//{NS}row"):
        yield list(row.findall(f"{NS}c"))


def _cell_value(cell: ET.Element, shared_strings: Sequence[str], date_styles: set[int]):
    cell_type = cell.attrib.get("t")
    style_idx = int(cell.attrib.get("s", "0")) if "s" in cell.attrib else None
    if cell_type == "inlineStr":
        is_el = cell.find(f"{NS}is/{NS}t")
        return "" if is_el is None else is_el.text or ""
    if cell_type == "s":
        v = cell.find(f"{NS}v")
        if v is None or v.text is None:
            return ""
        return shared_strings[int(v.text)]
    v = cell.find(f"{NS}v")
    if v is None or v.text is None:
        return ""
    raw = v.text
    if style_idx is not None and style_idx in date_styles:
        try:
            return excel_number_to_datetime(float(raw))
        except Exception:
            return raw
    # try int/float conversion, fallback to string
    try:
        if "." in raw or "e" in raw.lower():
            return float(raw)
        return int(raw)
    except Exception:
        return raw


def read_xlsx(path: str, sheet_name: str | None = None) -> tuple[list[str], list[list[Any]]]:
    """Read ``path`` and return ``(header, rows)`` from the first sheet."""

    with zipfile.ZipFile(path, "r") as zf:
        shared_strings = _load_shared_strings(zf)
        date_styles = _load_date_styles(zf)
        sheet_xml = "xl/worksheets/sheet1.xml"
        if sheet_name:
            # Only a single sheet is used throughout the project; the parameter
            # exists solely for API compatibility.
            sheet_xml = f"xl/worksheets/{sheet_name}.xml"
        sheet_root = ET.fromstring(zf.read(sheet_xml))
        rows_iter = list(_iter_rows(sheet_root))
        if not rows_iter:
            return [], []
        header_cells = rows_iter[0]
        header: List[str] = []
        for cell in header_cells:
            value = _cell_value(cell, shared_strings, date_styles)
            header.append(str(value))
        data_rows: List[List[Any]] = []
        for r_idx, cells in enumerate(rows_iter[1:], start=2):
            row_values: Dict[int, Any] = {}
            max_index = -1
            for cell in cells:
                ref = cell.attrib.get("r", "")
                col_letters = "".join(itertools.takewhile(str.isalpha, ref))
                if not col_letters:
                    continue
                col_index = column_letter_to_index(col_letters)
                value = _cell_value(cell, shared_strings, date_styles)
                row_values[col_index] = value
                max_index = max(max_index, col_index)
            if max_index == -1:
                continue
            row = [row_values.get(i, "") for i in range(max_index + 1)]
            # normalise length to header
            if len(row) < len(header):
                row.extend([""] * (len(header) - len(row)))
            elif len(row) > len(header):
                row = row[: len(header)]
            data_rows.append(row)
        return header, data_rows


# --- Writing -----------------------------------------------------------------


CONTENT_TYPES_XML = """<?xml version='1.0' encoding='UTF-8'?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>
"""

RELS_XML = """<?xml version='1.0' encoding='UTF-8'?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
"""

WORKBOOK_XML = """<?xml version='1.0' encoding='UTF-8'?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="sheet1" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>
"""

WORKBOOK_RELS_XML = """<?xml version='1.0' encoding='UTF-8'?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>
"""

STYLES_XML = """<?xml version='1.0' encoding='UTF-8'?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="1">
    <numFmt numFmtId="165" formatCode="yyyy-mm-dd\ hh:mm"/>
  </numFmts>
  <fonts count="1">
    <font>
      <sz val="11"/>
      <color theme="1"/>
      <name val="Calibri"/>
      <family val="2"/>
    </font>
  </fonts>
  <fills count="1">
    <fill>
      <patternFill patternType="none"/>
    </fill>
  </fills>
  <borders count="1">
    <border>
      <left/><right/><top/><bottom/><diagonal/>
    </border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="3">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
  </cellXfs>
</styleSheet>
"""


def _build_shared_strings(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    strings = list(headers)
    for row in rows:
        for value in row:
            if isinstance(value, str) and value not in strings:
                strings.append(value)
    return strings


def _string_index_map(strings: Sequence[str]) -> Dict[str, int]:
    return {text: idx for idx, text in enumerate(strings)}


def _sheet_xml(headers: Sequence[str], rows: Sequence[Sequence[Any]], string_map: Dict[str, int]) -> str:
    lines = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">",
        "  <sheetData>",
    ]

    # header row with inline strings
    header_cells = []
    for idx, header in enumerate(headers):
        cell_ref = f"{column_index_to_letter(idx)}1"
        header_cells.append(
            f"      <c r=\"{cell_ref}\" t=\"inlineStr\"><is><t>{escape(header)}</t></is></c>"
        )
    lines.append("    <row r=\"1\">")
    lines.extend(header_cells)
    lines.append("    </row>")

    for r_idx, row in enumerate(rows, start=2):
        cell_xml: List[str] = []
        for c_idx, value in enumerate(row):
            if value is None or value == "":
                continue
            cell_ref = f"{column_index_to_letter(c_idx)}{r_idx}"
            if isinstance(value, datetime):
                excel_val = datetime_to_excel_number(value)
                cell_xml.append(
                    f"      <c r=\"{cell_ref}\" s=\"2\" t=\"n\"><v>{excel_val}</v></c>"
                )
            elif isinstance(value, (int, float)):
                cell_xml.append(
                    f"      <c r=\"{cell_ref}\" t=\"n\"><v>{value}</v></c>"
                )
            else:
                key = str(value)
                idx_str = string_map[key]
                cell_xml.append(
                    f"      <c r=\"{cell_ref}\" t=\"s\"><v>{idx_str}</v></c>"
                )
        lines.append(f"    <row r=\"{r_idx}\">")
        lines.extend(cell_xml)
        lines.append("    </row>")

    lines.append("  </sheetData>")
    lines.append("</worksheet>")
    return "\n".join(lines)


def _shared_strings_xml(strings: Sequence[str]) -> str:
    lines = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        f"<sst xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" count=\"{len(strings)}\" uniqueCount=\"{len(strings)}\">",
    ]
    for text in strings:
        lines.append(f"  <si><t>{escape(text)}</t></si>")
    lines.append("</sst>")
    return "\n".join(lines)


def write_simple_xlsx(path: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Write a tiny XLSX file to ``path``.

    The function intentionally emits the minimum required set of workbook
    components.  Only the features that are exercised in the unit tests are
    supported (one worksheet, inline/shared strings and numeric/datetime
    values).
    """

    strings = _build_shared_strings(headers, rows)
    string_map = _string_index_map(strings)
    sheet_xml = _sheet_xml(headers, rows, string_map)
    shared_xml = _shared_strings_xml(strings)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES_XML)
        zf.writestr("_rels/.rels", RELS_XML)
        zf.writestr("xl/workbook.xml", WORKBOOK_XML)
        zf.writestr("xl/_rels/workbook.xml.rels", WORKBOOK_RELS_XML)
        zf.writestr("xl/styles.xml", STYLES_XML)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        zf.writestr("xl/sharedStrings.xml", shared_xml)

