# energis/io/loader.py
from __future__ import annotations
import os, re
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

def _normalize_col_spec(spec):
    """Nimmt YAML-Spezifikation (str oder Liste) und gibt ein Tupel mit Kandidaten zurück."""
    if spec is None:
        return tuple()
    if isinstance(spec, (list, tuple)):
        return tuple(spec)
    return (str(spec),)

# --------- kleine Utilities ---------
def _require(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)

_norm = lambda s: re.sub(r"[^a-z0-9]+", "", str(s).lower())

def _get_col(df: pd.DataFrame, *candidates: str) -> str:
    """Robuste Spaltensuche: exakte, dann fuzzy-Treffer."""
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    for cand in candidates:
        key = _norm(cand)
        for k, orig in norm_map.items():
            if k == key or k.startswith(key) or key in k:
                return orig
    raise RuntimeError("Fehlende Spalte. Gesucht eine von: " + " | ".join(candidates))

def _parse_and_localize_datetime(
    sr: pd.Series,
    *,
    tz: str = "Europe/Berlin",
    input_time_is_local: bool = True,
    ambiguous_policy: str = "infer_then_first"  # "infer_then_first" | "first" | "last" | "NaT" | "infer"
) -> pd.DatetimeIndex:
    ts = pd.to_datetime(sr, errors="raise", utc=False)
    if getattr(ts.dt, "tz", None) is None:
        if input_time_is_local:
            if ambiguous_policy == "infer_then_first":
                try:
                    ts = ts.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="infer")
                except Exception:
                    ts = ts.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous=True)
            else:
                ambiguous_arg = {"first": True, "last": False, "NaT": "NaT", "infer": "infer"}.get(ambiguous_policy, "infer")
                ts = ts.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous=ambiguous_arg)
        else:
            ts = ts.dt.tz_localize("UTC")
    # Naiv in UTC zurückgeben (Excel-freundlich)
    ts_utc_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return pd.DatetimeIndex(ts_utc_naive)

def _enforce_monotonic_unique_index(
    df: pd.DataFrame,
    duplicate_strategy: str = "drop_first"
) -> pd.DataFrame:
    _require(isinstance(df.index, pd.DatetimeIndex), "Index ist kein Datumsindex.")
    df = df.sort_index()
    dup_mask = df.index.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup:
        if duplicate_strategy == "drop_first":
            print(f"[Loader] Entferne {n_dup} doppelte Zeitstempel (behalte erste Vorkommen).")
            df = df[~dup_mask]
        else:
            raise RuntimeError(f"Doppelte Zeitstempel gefunden: {n_dup}.")
    _require(df.index.is_monotonic_increasing, "Zeitindex nicht monoton steigend.")
    _require(df.index.is_unique, "Zeitindex enthält weiterhin Duplikate.")
    return df

def _reindex_hourly_and_fill(
    df: pd.DataFrame,
    *,
    dt_hours: float = 1.0,
    gap_strategy: str = "ffill"
) -> pd.DataFrame:
    """Resample auf exakte Schrittweite; füllt Lücken je nach Strategie."""
    if df.empty:
        return df
    freq = pd.to_timedelta(dt_hours, unit="h")
    target_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    if len(target_index) != len(df):
        missing = len(target_index) - len(df)
        msg = f"[Loader] Es fehlen {missing} Zeitschritte (soll: {len(target_index)}, ist: {len(df)})."
        if gap_strategy == "error":
            raise RuntimeError(msg + " gap_strategy='error' → abbrechen.")
        print(msg + f" gap_strategy='{gap_strategy}' → fülle Lücken.")
    df2 = df.reindex(target_index)
    if gap_strategy == "ffill":
        df2 = df2.ffill().bfill()
    elif gap_strategy == "bfill":
        df2 = df2.bfill().ffill()
    elif gap_strategy == "interp":
        num_cols = df2.select_dtypes(include=[np.number]).columns
        non_num  = [c for c in df2.columns if c not in num_cols]
        df2[num_cols] = df2[num_cols].interpolate(method="time", limit_direction="both")
        if non_num:
            df2[non_num] = df2[non_num].ffill().bfill()
    steps = df2.index.to_series().diff().dropna().unique()
    _require(len(steps) == 1 and steps[0] == freq, f"Schrittweite ist nicht {freq}.")
    return df2

# --------- Hauptfunktion ---------
def load_input_excel(
    path: str,
    site_cfg: Dict[str, Any],
    *,
    dt_hours: Optional[float] = None,
    tz: Optional[str] = None,
    input_time_is_local: Optional[bool] = None,
    duplicate_strategy: str = "drop_first",
    gap_strategy: str = "ffill",
    ambiguous_policy: str = "first",  # robust bzgl. DST
) -> pd.DataFrame:
    """
    Erwartet site_cfg wie in configs/sites/*.site.yaml.
    Nutzt folgende Felder, wenn vorhanden:
      - sheet
      - year_target
      - time: { tz, input_local, ambiguous_policy }
      - columns: { price, heat, co2, wrgQ1..4, wrgT1..4 }
    """
    _require(os.path.exists(path), f"Datei nicht gefunden: {path}")
    xls = pd.ExcelFile(path)

    # --- Site-Zusätze ---
    sheet_name = site_cfg.get("sheet", None)
    year_target = site_cfg.get("year_target", None)

    time_cfg = site_cfg.get("time", {})
    tz = tz or time_cfg.get("tz", "Europe/Berlin")
    input_local = input_time_is_local if input_time_is_local is not None else bool(time_cfg.get("input_local", True))
    ambiguous_policy = time_cfg.get("ambiguous_policy", ambiguous_policy)

    # --- Einlesen ---
    sh = sheet_name if sheet_name is not None else xls.sheet_names[0]
    df_raw = pd.read_excel(path, sheet_name=sh)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    # --- Zeitspalte finden ---
    time_col = None
    for cand in ["Datum", "Date", "datetime", "Zeit", "Timestamp", "Time"]:
        if cand in df_raw.columns:
            time_col = cand
            break
    _require(time_col is not None, "Zeitspalte (z.B. 'Datum') fehlt in der Eingabedatei.")

    idx = _parse_and_localize_datetime(
        df_raw[time_col],
        tz=tz,
        input_time_is_local=input_local,
        ambiguous_policy=ambiguous_policy,
    )
    df = df_raw.set_index(idx).drop(columns=[time_col])

    if year_target is not None:
        df = df[df.index.year == int(year_target)]
        _require(len(df) > 0, f"Im Zieljahr {year_target} keine Daten.")

    df = _enforce_monotonic_unique_index(df, duplicate_strategy=duplicate_strategy)

    # --- Spaltenmapping: explizit aus site_cfg.columns, sonst heuristisch ---
    cols_cfg = site_cfg.get("columns", {}) or {}

    def pick(col_key: str, *fallbacks: str) -> str:
        if col_key in cols_cfg and cols_cfg[col_key]:
            # erlaubt: exakter Name oder Liste/Alternativen
            if isinstance(cols_cfg[col_key], (list, tuple)):
                return _get_col(df, *cols_cfg[col_key])
            return _get_col(df, cols_cfg[col_key])
        return _get_col(df, *fallbacks)

    col_price = pick("price", "Day_Ahead_Price €/MWh", "Day Ahead Price €/MWh", "strompreis", "price_eur_mwh")
    col_heat  = pick("heat",  "Wärmebedarf MW", "Waermebedarf MW", "waermebedarf", "heat_demand_mw")
    col_co2   = pick("co2",   "CO2_consumption_based kgCO2/MWh", "co2_intensity_kgco2_mwh", "co2 kgco2/mwh")


# WRG-Spalten robust behandeln (ohne *-Unpacking in Inline-Ifs)
df_normed = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in df.columns}

for i in [1,2,3,4]:
    # --- WRG Q ---
    col = None
    spec_tuple = _normalize_col_spec(wrgQ_cols[i])
    if spec_tuple:
        try:
            col = _get_col(df, *spec_tuple)
        except RuntimeError:
            col = None
    if col is None:
        # Fallback-Kandidaten
        for cand in [f"WRG{i}Q MW", f"WRG{i} Q MW", f"WRG{i}_Q MW", f"WRG{i}_Q MW_th", f"wrg{i}qmw"]:
            key = re.sub(r"[^a-z0-9]+", "", cand.lower())
            if key in df_normed:
                col = df_normed[key]
                break
    if col is not None:
        out[f"WRG{i}_Q_cap"] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )

    # --- WRG T ---
    colT = None
    spec_tuple_T = _normalize_col_spec(wrgT_cols[i])
    if spec_tuple_T:
        try:
            colT = _get_col(df, *spec_tuple_T)
        except RuntimeError:
            colT = None
    if colT is None:
        for cand in [f"WRG{i}_T °C", f"WRG{i} T °C", f"wrg{i} temp c", f"WRG{i}_T C"]:
            key = re.sub(r"[^a-z0-9]+", "", cand.lower())
            if key in df_normed:
                colT = df_normed[key]
                break
    if colT is not None:
        out[f"WRG{i}_T_K"] = pd.to_numeric(
            df[colT].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        ) + 273.15


    # --- Output-Frame aufbauen ---
    out = pd.DataFrame(index=df.index)

    _to_num = lambda s: pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )

    out["strompreis_EUR_MWh"] = _to_num(df[col_price])
    out["waermebedarf_MWth"]  = _to_num(df[col_heat])
    out["grid_co2_kg_MWh"]    = _to_num(df[col_co2])

    for i in [1,2,3,4]:
        # WRG Q
        if wrgQ_cols[i]:
            col = _get_col(df, wrgQ_cols[i] if isinstance(wrgQ_cols[i], str) else *wrgQ_cols[i])
        else:
            col = None
            for cand in [f"WRG{i}Q MW", f"WRG{i} Q MW", f"WRG{i}_Q MW", f"WRG{i}_Q MW_th", f"wrg{i}qmw"]:
                if _norm(cand) in {_norm(c) for c in df.columns}:
                    col = _get_col(df, cand); break
        if col:
            out[f"WRG{i}_Q_cap"] = _to_num(df[col])
        # WRG T
        if wrgT_cols[i]:
            colT = _get_col(df, wrgT_cols[i] if isinstance(wrgT_cols[i], str) else *wrgT_cols[i])
        else:
            colT = None
            for cand in [f"WRG{i}_T °C", f"WRG{i} T °C", f"wrg{i} temp c", f"WRG{i}_T C"]:
                if _norm(cand) in {_norm(c) for c in df.columns}:
                    colT = _get_col(df, cand); break
        if colT:
            out[f"WRG{i}_T_K"] = _to_num(df[colT]) + 273.15

    # Zahlen sauber auffüllen
    out = out.ffill().bfill()

    _require(not out.isna().any().any(), "NaN in Import-Daten nach Konvertierung. Bitte Quelle prüfen.")

    # Schrittweite erzwingen
    if dt_hours is None:
        # Fallback 1h
        dt_hours = 1.0
    out = _reindex_hourly_and_fill(out, dt_hours=float(dt_hours), gap_strategy=gap_strategy)

    print(f"[LOAD] {os.path.basename(path)} → {len(out)} Std (UTC-naiv) "
          f"von {out.index.min()} bis {out.index.max()}, dt={dt_hours}h")

    # ---------- SANITY STATS (jetzt gibt es 'out' sicher!) ----------
    def _series_stats(name, s: pd.Series):
        return dict(name=name, n=int(s.shape[0]), min=float(s.min()), max=float(s.max()), mean=float(s.mean()))

    stats = [
        _series_stats("strompreis_EUR_MWh", out["strompreis_EUR_MWh"]),
        _series_stats("waermebedarf_MWth",  out["waermebedarf_MWth"]),
        _series_stats("grid_co2_kg_MWh",    out["grid_co2_kg_MWh"])
    ]
    for i in [1,2,3,4]:
        qc = f"WRG{i}_Q_cap"
        tc = f"WRG{i}_T_K"
        if qc in out.columns: stats.append(_series_stats(qc, out[qc]))
        if tc in out.columns: stats.append(_series_stats(tc, out[tc]))
    print("[LOAD][sanity]", stats)

    # ---------- HARD GUARDS ----------
    if out["waermebedarf_MWth"].abs().max() == 0.0:
        raise RuntimeError(
            "Wärmebedarf ist überall 0. Bitte Spaltenzuordnung im Import prüfen "
            "(z.B. 'Wärmebedarf MW'). Ohne positive Nachfrage produziert das Modell rationalerweise nichts."
        )

    if (out["strompreis_EUR_MWh"].isna().any() or out["strompreis_EUR_MWh"].abs().max() == 0.0):
        print("[WARN] Strompreis ist leer oder 0 – das kann zu degenerierten Optima führen.")

    return out
