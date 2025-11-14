"""Plotting helpers for exporting standard scenario visualisations."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Callable, Iterable, Mapping, Sequence

from energis.utils.timeseries import TimeSeriesTable

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    HAVE_MATPLOTLIB = True
except Exception:  # pragma: no cover - matplotlib optional
    HAVE_MATPLOTLIB = False
    plt = None
    mdates = None

__all__ = ["HAVE_MATPLOTLIB", "export_plots"]


def export_plots(
    outdir: str,
    table: TimeSeriesTable,
    series: Mapping[str, Sequence[float]],
    summary_sections: Mapping[str, Mapping[str, object]] | None = None,
    *,
    dpi: int = 200,
) -> list[str]:
    """Generate a couple of high-quality figures for the export package.

    The plots intentionally follow a visual language often found in energy
    systems journals (e.g. Applied Energy, Applied Thermal Engineering).
    They are generated on best-effort basis – if required data or matplotlib
    are unavailable, the function degrades gracefully.
    """

    if not HAVE_MATPLOTLIB:
        print("[EXPORT] Matplotlib nicht verfügbar – Diagramme werden übersprungen.")
        return []

    if not table.index:
        print("[EXPORT] Keine Zeitschrittdaten vorhanden – Diagramme werden übersprungen.")
        return []

    os.makedirs(outdir, exist_ok=True)

    timestamps: Sequence[datetime] | Sequence[int]
    first = table.index[0]
    if isinstance(first, datetime):
        timestamps = table.index
    else:
        timestamps = list(range(len(table.index)))

    generated: list[str] = []

    def _configure_time_axis(ax) -> None:
        if isinstance(timestamps[0], datetime) and mdates is not None:
            locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            ax.figure.autofmt_xdate()
            ax.set_xlabel("Zeit")
        else:
            ax.set_xlabel("Zeitschritt")

    generated.extend(
        _heat_balance_plot(outdir, timestamps, table, series, dpi, _configure_time_axis)
    )
    generated.extend(
        _electric_balance_plot(outdir, timestamps, table, series, dpi, _configure_time_axis)
    )
    generated.extend(
        _storage_plot(outdir, timestamps, series, dpi, _configure_time_axis)
    )
    if summary_sections:
        generated.extend(
            _cost_breakdown_plot(outdir, summary_sections, dpi)
        )

    return generated


def _heat_balance_plot(
    outdir: str,
    timestamps: Sequence[datetime] | Sequence[int],
    table: TimeSeriesTable,
    series: Mapping[str, Sequence[float]],
    dpi: int,
    configure_axis: Callable[[Any], None],
) -> Iterable[str]:
    demand = table.data.get("waermebedarf_MWth")
    if not demand:
        return []

    supply_components: list[Sequence[float]] = []
    labels: list[str] = []

    for key, values in series.items():
        if key == "TES_charge_MW":
            continue
        if key.endswith("_Q_th_MW") or key == "TES_discharge_MW":
            if not _has_content(values):
                continue
            supply_components.append(values)
            labels.append(_prettify_label(key))

    if not supply_components:
        return []

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.stackplot(timestamps, *supply_components, labels=labels, alpha=0.7, linewidth=0)
    ax.plot(timestamps, demand, color="#1f77b4", linewidth=2.2, label="Wärmebedarf")
    ax.set_ylabel("Thermische Leistung [MW]")
    ax.set_title("Wärmebilanz")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    configure_axis(ax)
    ax.legend(loc="upper right", frameon=True, ncol=2)

    filename = os.path.join(outdir, "heat_balance.png")
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)
    return [filename]


def _electric_balance_plot(
    outdir: str,
    timestamps: Sequence[datetime] | Sequence[int],
    table: TimeSeriesTable,
    series: Mapping[str, Sequence[float]],
    dpi: int,
    configure_axis: Callable[[Any], None],
) -> Iterable[str]:
    pbuy = series.get("P_buy_MW")
    psell = series.get("P_sell_MW")
    if not pbuy and not psell:
        return []

    consumption_components: list[Sequence[float]] = []
    labels: list[str] = []
    for key, values in series.items():
        if key in {"P_buy_MW", "P_sell_MW"}:
            continue
        if key.endswith("_Pel_MW"):
            if not _has_content(values):
                continue
            consumption_components.append(values)
            labels.append(_prettify_label(key))

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    if consumption_components:
        ax.stackplot(
            timestamps,
            *consumption_components,
            labels=labels,
            alpha=0.6,
            linewidth=0,
        )
    if pbuy:
        ax.plot(timestamps, pbuy, color="#d62728", linewidth=2.0, label="Netzbezug")
    if psell:
        ax.plot(timestamps, psell, color="#2ca02c", linewidth=2.0, label="Netzeinspeisung")

    ax.set_ylabel("Elektrische Leistung [MW]")
    ax.set_title("Elektrische Bilanz")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    configure_axis(ax)
    ax.legend(loc="upper right", frameon=True, ncol=2)

    filename = os.path.join(outdir, "electric_balance.png")
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)
    return [filename]


def _storage_plot(
    outdir: str,
    timestamps: Sequence[datetime] | Sequence[int],
    series: Mapping[str, Sequence[float]],
    dpi: int,
    configure_axis: Callable[[Any], None],
) -> Iterable[str]:
    soc = series.get("TES_SOC_MWh")
    if not soc or not _has_content(soc):
        return []

    charge = series.get("TES_charge_MW")
    discharge = series.get("TES_discharge_MW")

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(timestamps, soc, color="#1f77b4", linewidth=2.2, label="Speicherfüllstand")
    ax.set_ylabel("Energieinhalt [MWh]")
    ax.set_title("Thermischer Speicher")
    ax.grid(True, which="both", axis="both", alpha=0.3)
    configure_axis(ax)

    if charge and _has_content(charge):
        ax2 = ax.twinx()
        ax2.fill_between(
            timestamps,
            0,
            charge,
            color="#ff7f0e",
            alpha=0.25,
            label="Beladung",
        )
        ax2.set_ylabel("Lade-/Entladeleistung [MW]")
        if discharge and _has_content(discharge):
            ax2.fill_between(
                timestamps,
                0,
                [-abs(v) for v in discharge],
                color="#2ca02c",
                alpha=0.25,
                label="Entladung",
            )
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=True)
    else:
        ax.legend(loc="upper right", frameon=True)

    filename = os.path.join(outdir, "storage_operation.png")
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)
    return [filename]


def _cost_breakdown_plot(
    outdir: str,
    summary_sections: Mapping[str, Mapping[str, object]],
    dpi: int,
) -> Iterable[str]:
    objective = summary_sections.get("objective")
    if not isinstance(objective, Mapping):
        return []

    entries: list[tuple[str, float]] = []
    for key, value in objective.items():
        if not key.endswith("_EUR") or key == "OBJ_value_EUR":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if abs(number) < 1e-6:
            continue
        entries.append((_prettify_label(key), number))

    if not entries:
        return []

    entries.sort(key=lambda item: abs(item[1]), reverse=True)
    labels, values = zip(*entries)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    positions = range(len(labels))
    colors = ["#4c72b0" if val >= 0 else "#dd8452" for val in values]
    ax.barh(positions, values, color=colors)
    ax.set_yticks(list(positions), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Kosten [EUR]")
    ax.set_title("Kostenaufteilung")
    ax.grid(True, axis="x", alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.8)

    filename = os.path.join(outdir, "cost_breakdown.png")
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)
    return [filename]


def _prettify_label(name: str) -> str:
    suffix_map = {
        "_Q_th_MW": ("Thermische Leistung", " [MW]"),
        "_Pel_MW": ("Elektrische Leistung", " [MW]"),
        "_fuel_MW": ("Brennstoffleistung", " [MW]"),
        "_MW": ("", " [MW]"),
        "_MWh": ("", " [MWh]"),
        "_EUR": ("", " [EUR]"),
    }

    pretty = name
    unit = ""
    for suffix, (alias, unit_label) in suffix_map.items():
        if pretty.endswith(suffix):
            pretty = pretty[: -len(suffix)]
            unit = unit_label
            if alias:
                pretty = f"{pretty}_{alias}" if pretty else alias
            break

    pretty = pretty.replace("_", " ").strip()

    lower = pretty.lower()
    special_map = {
        "p buy": "Netzbezug",
        "p sell": "Netzeinspeisung",
        "tes": "TES",
    }
    if lower in special_map:
        pretty = special_map[lower]

    replacements = {
        "charge": "Beladung",
        "discharge": "Entladung",
        "storage": "Speicher",
        "thermische": "Thermische",
        "elektrische": "Elektrische",
        "brennstoffleistung": "Brennstoffleistung",
        "soc": "Füllstand",
    }
    words = pretty.split()
    normalized_words: list[str] = []
    for word in words:
        repl = replacements.get(word.lower())
        if repl:
            normalized_words.append(repl)
        elif word.isupper() or any(char.isdigit() for char in word):
            normalized_words.append(word)
        else:
            normalized_words.append(word.capitalize())
    pretty = " ".join(normalized_words)

    return f"{pretty}{unit}".strip()


def _has_content(values: Sequence[float]) -> bool:
    return any(abs(float(v)) > 1e-9 for v in values)
