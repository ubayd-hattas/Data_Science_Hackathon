"""Loading and feature engineering for the building-age dataset.

Reimplements the Notebook 3 pipeline as importable functions, with two changes
that matter on a memory-limited machine:

* ``load_city`` reads only the 27 columns the pipeline uses. The parquet has 48;
  the extras are ``large_string`` (``scene_id_*``, ``dataset_*``, ``pixel_id``)
  and raw QA bitmasks that ``qa_valid_*`` already summarises. Dropping them cuts
  Madrid from 1259 MB to 359 MB.

* Gap filling reshapes each band to a dense ``(n_pixels, n_years)`` array and
  interpolates along the year axis with numpy, instead of a cross-join followed
  by ``groupby().transform(lambda ...)`` over 76k groups. Same result, no
  intermediate 3.2M-row merge.

Usage from a notebook in ``notebooks/``::

    import sys; sys.path.insert(0, '..')
    from src.data import build_city

    madrid = build_city('../data/madrid_train.parquet')
    X, y = madrid.X, madrid.y
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

BANDS = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
INDICES = ['NDVI', 'NDBI', 'UI', 'MNDWI', 'BSI']
PID = ['px_key', 'py_key']

#: Year separating age class 1 from class 2, per city (Notebook 1, section 1.4).
CLASS_EVENTS = {'Amsterdam': 1945, 'Madrid': 1960}

CLASS_LABELS = {
    'Amsterdam': {1: 'Pre-1945', 2: '1945-1984', 3: '1984-2004', 4: '2004-2024'},
    'Madrid':    {1: 'Pre-1960', 2: '1960-1984', 3: '1984-2004', 4: '2004-2024'},
}
CLASS_COLORS = {1: '#4393c3', 2: '#2ca25f', 3: '#fd8d3c', 4: '#de2d26'}

#: Blue reflectance above this marks haze the standard QA flags miss — chiefly
#: Landsat 7 scenes over Amsterdam around 2003. Normal urban Blue peaks ~10000.
BLUE_MAX = 15_000

#: Boundary between the "early" (pre-existing land use) and "late" (current
#: building) halves of the time series.
LATE_FROM = 2004


def lean_columns() -> list[str]:
    """The 27 parquet columns the pipeline actually reads."""
    cols = ['city', 'year', 'px_key', 'py_key', 'coverage', 'weighted_mean_year']
    for slot in (1, 2, 3):
        cols += [f'{b}_{slot}' for b in BANDS] + [f'qa_valid_{slot}']
    return cols


# ── Loading ──────────────────────────────────────────────────────────────────

def load_city(path: str) -> pd.DataFrame:
    """Read one city's parquet, keeping only the columns the pipeline needs."""
    return pd.read_parquet(path, columns=lean_columns())


def assign_age_class(df: pd.DataFrame, events: dict = CLASS_EVENTS) -> pd.DataFrame:
    """Cut ``weighted_mean_year`` into ordered classes 1-4, per city.

    The class 1/2 boundary is city-specific; 1984 and 2004 are shared.
    """
    out = df.copy()
    out['age_class'] = pd.NA
    for city, event in events.items():
        mask = out['city'] == city
        if not mask.any():
            continue
        out.loc[mask, 'age_class'] = pd.cut(
            out.loc[mask, 'weighted_mean_year'],
            bins=[-np.inf, event, 1984, 2004, np.inf],
            labels=[1, 2, 3, 4],
        ).astype('Int64')
    return out


def make_flat_view(df: pd.DataFrame, blue_max: float = BLUE_MAX) -> pd.DataFrame:
    """Collapse the three observation slots to one value per band per row.

    Takes the first QA-valid slot (obs_1, else obs_2, else obs_3), drops rows
    with no valid observation, and removes haze-contaminated rows.
    """
    flat = df[PID + ['year', 'age_class']].copy()

    for band in BANDS:
        col = np.full(len(df), np.nan, dtype=np.float32)
        for slot in (1, 2, 3):
            values = df[f'{band}_{slot}'].to_numpy(dtype=np.float32, copy=False)
            valid = df[f'qa_valid_{slot}'].to_numpy()
            take = np.isnan(col) & valid & np.isfinite(values)
            col[take] = values[take]
        flat[band] = col

    flat = flat.dropna(subset=BANDS, how='all')
    flat = flat[~(flat['Blue'] > blue_max)]
    return flat[flat['age_class'].notna()].reset_index(drop=True)


# ── Gap filling ──────────────────────────────────────────────────────────────

def _interp_rows(a: np.ndarray) -> np.ndarray:
    """Fill NaNs along axis 1: linear inside, edge-hold outside.

    ``a`` is ``(n_pixels, n_years)``. Interior gaps are linearly interpolated
    between the surrounding observed years; leading NaNs take the first observed
    value, trailing NaNs the last. All-NaN rows are left as NaN.
    """
    n_rows, n_years = a.shape
    valid = ~np.isnan(a)
    years = np.arange(n_years)

    # Index of the nearest valid year at or before each position (-1 if none).
    prev = np.where(valid, years, -1)
    np.maximum.accumulate(prev, axis=1, out=prev)

    # Index of the nearest valid year at or after each position (n_years if none).
    nxt = np.where(valid, years, n_years)
    nxt = np.minimum.accumulate(nxt[:, ::-1], axis=1)[:, ::-1]

    has_prev, has_next = prev >= 0, nxt < n_years
    rows = np.arange(n_rows)[:, None]
    v_prev = a[rows, np.where(has_prev, prev, 0)]
    v_next = a[rows, np.where(has_next, nxt, n_years - 1)]

    span = (nxt - prev).astype(np.float32)
    with np.errstate(invalid='ignore', divide='ignore'):
        weight = np.where(span > 0, (years - prev) / span, 0.0)

    filled = np.where(
        has_prev & has_next, v_prev + (v_next - v_prev) * weight,
        np.where(has_prev, v_prev, v_next),
    )
    return np.where(valid, a, filled).astype(np.float32)


@dataclass
class Panel:
    """A city's gap-filled time series as dense ``(n_pixels, n_years)`` arrays.

    Attributes:
        pixels: ``(n_pixels, 2)`` frame of ``px_key``/``py_key``, row-aligned.
        years:  the year for each column.
        cube:   ``{band_or_index: (n_pixels, n_years) float32}``.
        labels: age class per pixel, row-aligned.
        city:   city name.
        observed: ``(n_pixels, n_years)`` bool — True where a real observation
            existed before gap filling.
    """
    pixels: pd.DataFrame
    years: np.ndarray
    cube: dict[str, np.ndarray]
    labels: np.ndarray
    city: str
    observed: np.ndarray


def to_panel(flat: pd.DataFrame, city: str) -> Panel:
    """Reshape the flat view into dense per-pixel arrays and fill year gaps.

    Scatters observed values into a ``(n_pixels, n_years)`` grid per band, so no
    cross-join or row-wise merge is needed. At Madrid's size this is ~13 MB per
    band rather than a 3.2M-row intermediate frame.
    """
    pixels = flat[PID].drop_duplicates().sort_values(PID).reset_index(drop=True)
    pixels['_row'] = np.arange(len(pixels))

    years = np.arange(int(flat['year'].min()), int(flat['year'].max()) + 1)
    year_to_col = {y: i for i, y in enumerate(years)}

    idx = flat.merge(pixels, on=PID, how='left')
    row = idx['_row'].to_numpy()
    col = idx['year'].map(year_to_col).to_numpy()

    shape = (len(pixels), len(years))
    cube, observed = {}, np.zeros(shape, dtype=bool)
    for band in BANDS:
        grid = np.full(shape, np.nan, dtype=np.float32)
        grid[row, col] = idx[band].to_numpy(dtype=np.float32, copy=False)
        observed |= ~np.isnan(grid)
        cube[band] = _interp_rows(grid)

    labels = (idx.drop_duplicates('_row').sort_values('_row')['age_class']
                 .to_numpy(dtype=np.int8))

    return Panel(pixels[PID], years, cube, labels, city, observed)


def add_spectral_indices(panel: Panel) -> Panel:
    """Add NDVI, NDBI, UI, MNDWI and BSI to the panel, computed per pixel-year.

    Normalised band ratios are less sensitive than raw reflectance to
    illumination and sensor differences between scenes.
    """
    c, eps = panel.cube, 1e-6
    B, G, R = c['Blue'], c['Green'], c['Red']
    N, S1, S2 = c['NIR'], c['SWIR1'], c['SWIR2']

    c['NDVI'] = (N - R) / (N + R + eps)
    c['NDBI'] = (S1 - N) / (S1 + N + eps)
    c['UI'] = (S2 - N) / (S2 + N + eps)
    c['MNDWI'] = (G - S1) / (G + S1 + eps)
    c['BSI'] = ((S1 + R) - (N + B)) / ((S1 + R) + (N + B) + eps)
    return panel


# ── Features ─────────────────────────────────────────────────────────────────

@dataclass
class CityData:
    """One city, collapsed to a feature matrix ready for modelling."""
    X: np.ndarray               # (n_pixels, 60) float32
    y: np.ndarray               # (n_pixels,) int8, values 1-4
    feature_names: list[str]
    pixels: pd.DataFrame        # px_key / py_key, row-aligned with X
    city: str

    def __len__(self) -> int:
        return len(self.X)


def build_features(panel: Panel, late_from: int = LATE_FROM) -> CityData:
    """Collapse each pixel's time series into the 60-feature vector.

    Groups: overall mean/std per band (12) and per index (10), early- and
    late-period mean/std per band (24), year-on-year change mean/std per band
    (12), and two period-coverage indicators.

    The early/late split matters because 1984-2003 shows the *previous* land use
    at a location while 2004+ shows the current building. For classes 1 and 2
    both windows describe the same structure; for classes 3 and 4 they straddle
    the construction event, so their contrast is the most discriminative signal.
    """
    years, cube = panel.years, panel.cube
    early = years < late_from
    late = ~early

    columns, names = [], []

    def add(name: str, values: np.ndarray) -> None:
        columns.append(values.astype(np.float32))
        names.append(name)

    for key in BANDS + INDICES:
        add(f'{key}_mean', cube[key].mean(axis=1))
        add(f'{key}_std', cube[key].std(axis=1))

    for band in BANDS:
        add(f'{band}_early_mean', cube[band][:, early].mean(axis=1))
        add(f'{band}_early_std', cube[band][:, early].std(axis=1))
        add(f'{band}_late_mean', cube[band][:, late].mean(axis=1))
        add(f'{band}_late_std', cube[band][:, late].std(axis=1))

    for band in BANDS:
        diff = np.diff(cube[band], axis=1)
        add(f'd_{band}_mean', diff.mean(axis=1))
        add(f'd_{band}_std', diff.std(axis=1))

    add('has_early_data', panel.observed[:, early].any(axis=1))
    add('has_late_data', panel.observed[:, late].any(axis=1))

    return CityData(np.column_stack(columns), panel.labels, names,
                    panel.pixels, panel.city)


def build_city(path: str, late_from: int = LATE_FROM) -> CityData:
    """Run the whole pipeline for one city: parquet path in, features out."""
    df = assign_age_class(load_city(path))
    city = df['city'].iloc[0]
    flat = make_flat_view(df)
    del df

    panel = add_spectral_indices(to_panel(flat, city))
    del flat
    return build_features(panel, late_from=late_from)
