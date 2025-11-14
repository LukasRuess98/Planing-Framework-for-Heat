from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from energis.models.system_builder import _cop_series_from_table
from energis.utils.timeseries import TimeSeriesTable


def _base_table(hours: int, extra: dict[str, list[float]]) -> TimeSeriesTable:
    base = datetime(2023, 1, 1)
    index = [base + timedelta(hours=i) for i in range(hours)]
    data = {
        "strompreis_EUR_MWh": [0.0] * hours,
        "waermebedarf_MWth": [0.0] * hours,
        "grid_co2_kg_MWh": [0.0] * hours,
    }
    columns = list(data.keys())
    for key, values in extra.items():
        columns.append(key)
        data[key] = values
    return TimeSeriesTable(index=index, columns=columns, data=data)


def test_cop_series_bilinear_interpolation_with_clamping():
    table = _base_table(
        3,
        {
            "WRG1_T_K": [305.0, 315.0, 330.0],
            "sink_temp_K": [330.0, 335.0, 345.0],
        },
    )
    cfg = {
        "heat_pumps": {
            "cop": {
                "tables": {
                    "standard": {
                        "x": [300.0, 310.0, 320.0],
                        "y": [330.0, 340.0],
                        "values": [
                            [4.0, 4.5, 5.0],
                            [3.0, 3.5, 4.0],
                        ],
                        "x_column": "WRG1_T_K",
                        "y_column": "sink_temp_K",
                        "clamp": True,
                    }
                }
            }
        }
    }

    result = _cop_series_from_table(table, "WRG1_T_K", cfg, "standard")
    assert result == pytest.approx([4.25, 4.25, 4.0])


def test_cop_series_out_of_range_without_clamp_raises():
    table = _base_table(
        1,
        {
            "WRG1_T_K": [330.0],
            "sink_temp_K": [330.0],
        },
    )
    cfg = {
        "heat_pumps": {
            "cop": {
                "tables": {
                    "standard": {
                        "x": [300.0, 310.0, 320.0],
                        "y": [330.0, 340.0],
                        "values": [
                            [4.0, 4.5, 5.0],
                            [3.0, 3.5, 4.0],
                        ],
                        "x_column": "WRG1_T_K",
                        "y_column": "sink_temp_K",
                        "clamp": False,
                    }
                }
            }
        }
    }

    with pytest.raises(ValueError):
        _cop_series_from_table(table, "WRG1_T_K", cfg, "standard")
