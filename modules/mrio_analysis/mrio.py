"""FIGARO MRIO loading, transformation, and impact-calculation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import warnings

from modules.latex_style import apply_house_style, save_and_close, tex_safe, set_house_style

from modules.config import HELPERS_DIR, MRIO_DATA_DIR, project_relative_path

# Default industry color map (fallback if industrial.COLOR_IND cannot be imported)
COLOR_IND_DEFAULT = {
    "B": "#6B6B6B",
    "C16": "#6A4F4F",
    "C18": "#3A2E2E",
    "C19": "#000000",
    "C20": "#8801AA",
    "C21": "#DC6BF9",
    "C22": "#E8ACF7",
    "C23": "#DEF3CE",
    "C24": "#BDF39C",
    "C25": "#28CF28DF",
    "C26": "#0D8C46C8",
    "C27": "#06492696",
    "C28": "#002315",
    "C30": "#A9ECE3",
    "C33": "#ACCCD8B5",
    "D35": "#7A1F4B",
    "E37T39": "#9137BA",
    "F": "#A84C78",
    "G45": "#533200",
    "H49": "#023B90",
    "H50": "#1D5BAC",
    "H52": "#1E7BEC",
    "H53": "#6BACFB",
    "K64": "#C3C002",
    "K65": "#848200",
    "K66": "#585500",
    "M69_70": "#CE4949",
    "M71": "#870101",
    "M72": "#330101",
    "M74_75": "#440000",
    "N78": "#B67102",
    "N80T82": "#935800",
    "Others": "#B3ADAD",
}

# --------------------------------------------------------------
# FIGARO dataset paths and source URLs
# --------------------------------------------------------------
DEFAULT_MRIO_YEAR = 2024
_KNOWN_MRIO_EDITIONS = {2023: "25ed", 2024: "26ed"}


def _mrio_file_path(year: int, *, industry_matrix: bool, suffix: str) -> Path:
    matrix_type = "ind-by-ind" if industry_matrix else "prod-by-prod"
    edition = _KNOWN_MRIO_EDITIONS.get(year, f"{year}ed")
    directory = MRIO_DATA_DIR / str(year)
    preferred = directory / f"flatfile_eu-ic-io_{matrix_type}_{edition}_{year}{suffix}"
    if preferred.exists():
        return preferred

    candidates = sorted(directory.glob(f"flatfile_eu-ic-io_{matrix_type}_*_{year}{suffix}"))
    return candidates[0] if candidates else preferred


def _mrio_cache_path(year: int, *, industry_matrix: bool) -> Path:
    suffix = "_ind_by_ind" if industry_matrix else ""
    return MRIO_DATA_DIR / str(year) / f"MRIO_{year}_CORE_ROW{suffix}.parquet"


FLATFILE_PATH = _mrio_file_path(
    DEFAULT_MRIO_YEAR,
    industry_matrix=False,
    suffix=".csv",
)
PARQUET_PATH = _mrio_file_path(
    DEFAULT_MRIO_YEAR,
    industry_matrix=False,
    suffix=".parquet",
)
MRIO_CACHE_PATH = _mrio_cache_path(DEFAULT_MRIO_YEAR, industry_matrix=False)

FLATFILE_PATH_IND = _mrio_file_path(
    DEFAULT_MRIO_YEAR,
    industry_matrix=True,
    suffix=".csv",
)
PARQUET_PATH_IND = _mrio_file_path(
    DEFAULT_MRIO_YEAR,
    industry_matrix=True,
    suffix=".parquet",
)
MRIO_CACHE_PATH_IND = _mrio_cache_path(DEFAULT_MRIO_YEAR, industry_matrix=True)

# CIRCABC source URLs for the current FIGARO release.
FLATFILE_URL = (
    "https://circabc.europa.eu/ui/group/cec66924-a924-4f91-a0ef-600a0531e3ba/"
    "library/84c6c8cf-16f0-4e5b-8c0b-7125f7aa7df1/details?download=true"
)
INDUSTRY_MATRIX_URL = (
    "https://circabc.europa.eu/ui/group/cec66924-a924-4f91-a0ef-600a0531e3ba/"
    "library/15c722eb-702b-42e0-b1c6-00e7dd3c3198/details?download=true"
)

LEVELS = (0, 1, 2)

# ==============================================================
# 1) CLASSIFICATIONS
# ==============================================================
def load_classifications(
    helpers_dir: Path,
    ind_ind: bool | None = None,
    *,
    use_industry_matrix: bool | None = None,
    set_globals: bool = True,
):
    """
    Load classification codes.
    - If ind_ind is provided: always use prd classifications and (optionally) strip CPA_.
    - If ind_ind is None: use_industry_matrix controls prd vs ind classifications.
    """
    if ind_ind is None:
        use_industry_matrix = bool(use_industry_matrix) if use_industry_matrix is not None else False
    else:
        use_industry_matrix = False  # keep prd base when ind_ind is explicitly provided
    code_prefix = "ind" if use_industry_matrix else "prd"
    class_codes = {}
    for code in ("use", "ava"):
        class_codes[code] = pd.read_csv(
            helpers_dir / f"class_{code_prefix}_{code}.csv",
            index_col=0,
        )

    ava = class_codes["ava"]
    use = class_codes["use"]

    ind_codes_out = ava.index[
        (ava["type"] == "Industries") &
        (ava["A*64"] == 1)
    ].astype(str).tolist()

    final_codes_out = use.index[
        use["type"] == "Final use"
    ].astype(str).tolist()

    primary_inputs_out = ava.index[
        ava["type"] == "Components of value added"
    ].astype(str).tolist()

    ind_labels_out = (
        ava
        .loc[ava["type"] == "Industries", ["description"]]
        .drop_duplicates()
        .squeeze()
        .to_dict()
    )

    if ind_ind:
        def _strip_cpa(x: str) -> str:
            s = str(x)
            return s[4:] if s.startswith("CPA_") else s
        ind_codes_out = [_strip_cpa(c) for c in ind_codes_out]
        final_codes_out = [_strip_cpa(c) for c in final_codes_out]
        primary_inputs_out = [_strip_cpa(c) for c in primary_inputs_out]
        ind_labels_out = {_strip_cpa(k): v for k, v in ind_labels_out.items()}

    if set_globals:
        global ind_codes, final_codes, primary_inputs, ind_labels
        ind_codes = ind_codes_out
        final_codes = final_codes_out
        primary_inputs = primary_inputs_out
        ind_labels = ind_labels_out

    return ind_codes_out, final_codes_out, primary_inputs_out, ind_labels_out

ind_codes, final_codes, primary_inputs, ind_labels = load_classifications(HELPERS_DIR, use_industry_matrix=False)



def _normalize_io_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize flatfile/parquet schemas to standard IO column names
    rename = {}

    # Origin/destination
    if "C_ORIG" in df.columns or "C_DEST" in df.columns:
        rename.update({"C_ORIG": "origin", "C_DEST": "destination"})
    if "refArea" in df.columns or "counterpartArea" in df.columns:
        rename.update({"refArea": "origin", "counterpartArea": "destination"})

    # Value column
    if "obsValue" in df.columns and "value" not in df.columns:
        rename["obsValue"] = "value"

    # Product/industry codes
    if "prd_ava" not in df.columns:
        if "rowPp" in df.columns:
            rename["rowPp"] = "prd_ava"
        elif "PRD_AVA" in df.columns:
            rename["PRD_AVA"] = "prd_ava"
        elif "rowIi" in df.columns:
            rename["rowIi"] = "prd_ava"
        elif "rowI" in df.columns:
            rename["rowI"] = "prd_ava"
        elif "rowInd" in df.columns:
            rename["rowInd"] = "prd_ava"
        elif "rowIndustry" in df.columns:
            rename["rowIndustry"] = "prd_ava"
        elif "IND_AVA" in df.columns:
            rename["IND_AVA"] = "prd_ava"

    if "prd_use" not in df.columns:
        if "colPp" in df.columns:
            rename["colPp"] = "prd_use"
        elif "PRD_USE" in df.columns:
            rename["PRD_USE"] = "prd_use"
        elif "colIi" in df.columns:
            rename["colIi"] = "prd_use"
        elif "colI" in df.columns:
            rename["colI"] = "prd_use"
        elif "colInd" in df.columns:
            rename["colInd"] = "prd_use"
        elif "colIndustry" in df.columns:
            rename["colIndustry"] = "prd_use"
        elif "IND_USE" in df.columns:
            rename["IND_USE"] = "prd_use"

    if rename:
        df = df.rename(columns=rename)

    return df

# ==============================================================
# 2) IO DATA LOADING
# ==============================================================
def load_data(flatfile_path: Path, parquet_path: Path, *, save_raw_parquet: bool = False) -> pd.DataFrame:
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        return _normalize_io_columns(df)

    if not flatfile_path.exists():
        raise FileNotFoundError(f"File not found: {flatfile_path}")

    with open(flatfile_path, "r", encoding="utf-8", errors="ignore") as f:
        sep = ";" if ";" in f.readline() else ","

    df = pd.read_csv(flatfile_path, sep=sep)
    df = _normalize_io_columns(df)

    if save_raw_parquet:
        df.to_parquet(parquet_path, index=False)

    return df

    if not flatfile_path.exists():
        raise FileNotFoundError(f"File not found: {flatfile_path}")

    with open(flatfile_path, "r", encoding="utf-8", errors="ignore") as f:
        sep = ";" if ";" in f.readline() else ","

    df = pd.read_csv(flatfile_path, sep=sep)
    df = df.rename(columns={
        "refArea": "origin", "counterpartArea": "destination",
        "rowPp": "prd_ava", "colPp": "prd_use", "obsValue": "value"
    })

    if save_raw_parquet:
        df.to_parquet(parquet_path, index=False)

    return df


def download_file(url: str, path: Path) -> None:
    """
    Download a file to the given path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    import urllib.parse
    import urllib.error
    import re

    def _try_download(u: str) -> bool:
        try:
            urllib.request.urlretrieve(u, path)
            return True
        except urllib.error.HTTPError:
            return False
        except urllib.error.URLError:
            return False

    if _try_download(url):
        return

    # Try simple URL variants (details <-> details?download=true)
    if url.endswith("?download=true"):
        alt = url.replace("?download=true", "")
        if _try_download(alt):
            return
        url = alt
    elif url.endswith("/details"):
        alt = url + "?download=true"
        if _try_download(alt):
            return

    # Fallback: try to parse HTML for a direct file link
    try:
        with urllib.request.urlopen(url) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        raise FileNotFoundError(
            f"Could not download file from: {url}\n"
            f"Please download manually and place it at: {path}"
        )

    # look for likely download links
    candidates = re.findall(r'href="([^"]+)"', html)
    # prioritize links that look like direct downloads
    candidates = [c for c in candidates if any(ext in c.lower() for ext in [".csv", ".zip", "download"])]
    for c in candidates:
        link = urllib.parse.urljoin(url, c)
        if _try_download(link):
            return

    raise FileNotFoundError(
        f"Could not download file from: {url}\n"
        f"Please download manually and place it at: {path}"
    )

def simplify_regions(
    df: pd.DataFrame,
    *,
    CORE: set[str],
    year: int = 2023,
) -> pd.DataFrame:
    # Keep the original data blocks shared: copying all 11M rows before adding
    # three derived columns creates a large, unnecessary peak in memory.
    out = df.copy(deep=False)

    def classify_series(values: pd.Series) -> pd.Series:
        # Vectorized string operations avoid Series.apply(), which makes
        # pandas/Arrow materialize an object array for the whole flatfile.
        normalized = values.astype("string").str.strip()
        return normalized.where(normalized.isin(CORE), "ROW")

    out["origin_region"] = classify_series(out["origin"])
    out["destination_region"] = classify_series(out["destination"])
    out["year"] = year
    return out


def build_mrio(df):
    aggregated = df.groupby([
        "year", "prd_ava", "prd_use",
        "origin_region", "destination_region"
    ])["value"].sum().reset_index()

    return aggregated.pivot_table(
        index=["origin_region", "prd_ava"],
        columns=["destination_region", "prd_use"],
        values="value",
        fill_value=0.0
    )


# ==============================================================
# 3) MRIO CACHE (long parquet for robust MultiIndex)
# ==============================================================
def save_mrio_cache(mrio: pd.DataFrame, path: Path) -> None:
    """
    Save MRIO (MultiIndex rows/cols) to a long parquet in a pandas-future-proof way.
    """
    # stack all column levels into rows
    df_long = mrio.stack(
        level=list(range(mrio.columns.nlevels)),
        future_stack=True
    ).reset_index()

    df_long.columns = list(mrio.index.names) + list(mrio.columns.names) + ["value"]
    df_long.to_parquet(path, index=False)

def load_mrio_cache(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.pivot_table(
        index=["origin_region", "prd_ava"],
        columns=["destination_region", "prd_use"],
        values="value",
        fill_value=0.0
    )

# ==============================================================
# 4) VALIDATION (no globals)
# ==============================================================
def check_industry_balance(IO: pd.DataFrame, ind_codes: list[str], tol=1e-6, verbose=True) -> pd.Series:
    row_sums = IO.sum(axis=1)
    col_sums = IO.sum(axis=0)

    idx_rows = IO.index.to_frame(index=False)
    idx_cols = IO.columns.to_frame(index=False)

    idx_rows.columns = ["orig_region", "prd_ava_code"]
    idx_cols.columns = ["dest_region", "prd_use_code"]

    row_inds = [
        (r, c) for r, c in zip(idx_rows["orig_region"], idx_rows["prd_ava_code"])
        if isinstance(c, str) and c in ind_codes and r != "ImportROW"
    ]
    col_inds = [
        (r, c) for r, c in zip(idx_cols["dest_region"], idx_cols["prd_use_code"])
        if isinstance(c, str) and c in ind_codes and r != "ExportROW"
    ]

    common = sorted(set(row_inds) & set(col_inds))
    diffs = pd.Series({k: float(row_sums[k] - col_sums[k]) for k in common}, name="row_minus_col")

    if verbose:
        abs_d = np.abs(diffs.values)
        print("[Industry balance]")
        print(f"  Pairs: {len(diffs)}, max |diff|: {abs_d.max() if len(abs_d) else np.nan:.2e}")
        print(f"  # > {tol}: {(abs_d > tol).sum()}")

    return diffs

# ==============================================================
# 5) MAIN (cache read default, cache write only when refresh_cache=True)
# ==============================================================
def build_or_load_mrio(
    *, CORE: set[str],
    mrio_year: int = DEFAULT_MRIO_YEAR,
    use_industry_matrix: bool = False,
    download_matrix: bool = False,
    use_cache: bool = True,
    refresh_cache: bool = False,
    save_raw_parquet: bool = False,   # explicit
    validate: bool = True,
) -> pd.DataFrame:
    global ind_codes, final_codes, primary_inputs, ind_labels
    ind_codes, final_codes, primary_inputs, ind_labels = load_classifications(
        HELPERS_DIR,
        use_industry_matrix=use_industry_matrix,
    )

    # ---- cache read
    cache_path = _mrio_cache_path(mrio_year, industry_matrix=use_industry_matrix)
    flat_path = _mrio_file_path(
        mrio_year,
        industry_matrix=use_industry_matrix,
        suffix=".csv",
    )
    parquet_path = _mrio_file_path(
        mrio_year,
        industry_matrix=use_industry_matrix,
        suffix=".parquet",
    )

    print(f"MRIO year: {mrio_year}")
    print(f"MRIO dataset directory: {project_relative_path(MRIO_DATA_DIR / str(mrio_year))}")

    if use_cache and cache_path.exists() and not refresh_cache:
        print(f"Loading cached MRIO: {cache_path.name}")
        mrio = load_mrio_cache(cache_path)
    else:
        print("Loading flatfile/parquet...")
        if download_matrix and (not flat_path.exists()):
            label = "industry-by-industry" if use_industry_matrix else "product-by-product"
            print(f"Downloading {label} flatfile -> {flat_path.name}")
            download_file(INDUSTRY_MATRIX_URL if use_industry_matrix else FLATFILE_URL, flat_path)
        df = load_data(flat_path, parquet_path, save_raw_parquet=save_raw_parquet)
        df = simplify_regions(
            df,
            CORE=CORE,
            year=mrio_year,
        )

        print("Building MRIO (groupby + pivot)...")
        mrio = build_mrio(df)

        # ---- cache write ONLY if explicitly requested
        if use_cache and refresh_cache:
            print(f"Saving MRIO cache: {cache_path.name}")
            save_mrio_cache(mrio, cache_path)

    if validate:
        print("\nVALIDATION")
        print(f"MRIO sum: {mrio.to_numpy().sum():,.0f}")
        print(f"MRIO shape: {mrio.shape}")
        regions = sorted({str(x) for x in mrio.index.get_level_values("origin_region") if str(x)})
        industries = sorted({str(x) for x in mrio.index.get_level_values("prd_ava") if str(x)})
        print(f"Regions loaded ({len(regions)}): {regions}")
        print(f"Industries loaded ({len(industries)}): {industries}")
        _ = check_industry_balance(mrio, ind_codes, verbose=True)

    return mrio

# ==============================================================
# 3. DATA LOADING & PROCESSING
# ==============================================================
def _row_from_v(v_nj: pd.DataFrame, prd_ava_code: str) -> pd.Series:
    """
    v_nj is a DataFrame indexed by MultiIndex (origin_region, prd_ava)
    and columns are industries (tuples) like (destination_region, prd_use)
    We want a 1D coefficient vector per industry (columns).
    """
    idx = v_nj.index
    mask = (idx.get_level_values("prd_ava").astype(str) == str(prd_ava_code))
    if not mask.any():
        raise KeyError(f"'{prd_ava_code}' not found in v_nj.index prd_ava.")
    return v_nj.loc[mask].sum(axis=0)  # sum across origin_region if multiple


# ==============================================================
# 7. IO ANALYSIS FUNCTIONS
# ==============================================================

def calculate_A(IO):
    """
    Technical coefficients matrix A:
        A_ij = Z_ij / X_j

    where:
    - Z_ij is the intermediate input supplied by industry i
      to industry j.
    - X_j is the gross output of purchasing industry j.
    """
    row_prod = IO.index.get_level_values("prd_ava")
    col_use = IO.columns.get_level_values("prd_use")

    primary_ext = list(primary_inputs) + ["*"]
    ind_rows = IO.index[~row_prod.isin(primary_ext)]

    final_ext = list(final_codes) + ["*"]
    ind_cols = IO.columns[~col_use.isin(final_ext)]

    Z = IO.loc[ind_rows, ind_cols].astype(float)

    Xj = IO.loc[:, ind_cols].astype(float).sum(axis=0)
    Xj = Xj.replace(0.0, np.nan)

    A = Z.div(Xj, axis=1).fillna(0.0)

    return A


def calculate_C(IO):
    """
    Final demand share matrix C_shares:
    For each (destination_region d, final code f),
    C[:, (d,f)] = shares over origin-region × industry rows.
    Columns are MultiIndex (destination_region, prd_use).
    """
    row_prod = IO.index.get_level_values("prd_ava").astype(str)
    col_use  = IO.columns.get_level_values("prd_use").astype(str)
    col_dest = IO.columns.get_level_values("destination_region").astype(str)

    # industry rows only
    ind_rows = IO.index[~row_prod.isin(primary_inputs)]

    valid_final_codes = [c for c in final_codes if c != "TU" and c in col_use.unique()]

    data = {}
    for d in col_dest.unique():
        for f in valid_final_codes:
            cols_df = IO.columns[(col_dest == d) & (col_use == f)]
            if len(cols_df) == 0:
                continue

            Y_df = IO.loc[ind_rows, cols_df].astype(float).sum(axis=1)
            tot  = float(Y_df.sum())
            data[(d, f)] = (Y_df * 0.0) if tot == 0 else (Y_df / tot)

    C = pd.DataFrame(data, index=ind_rows).fillna(0.0)
    C.columns = pd.MultiIndex.from_tuples(C.columns, names=["destination_region", "prd_use"])
    return C


def plot_lb_box_all_industries(
    series: pd.Series,
    *,
    industries: list[str] | None = None,
    europe_group: set[str] | None = None,
    countries: set[str] | None = None,
    stat: str = "median",
    highlight: tuple[str, ...] = (),
    show_fliers: bool = False,
    value_scale: float = 1.0,
    fig_w: float = 14.0,
    fig_h: float = 5.0,
    dpi: int = 150,
    ylabel: str | None = r"Employment coefficient ($v_{b,i}^{\,r}$)",
    filename: str = "lb_mrio_box_all_industries.pdf",
    save=None,
    close=None,
):
    """
    Boxplot of a coefficient Series by industry, with countries as variation.

    series: pd.Series with MultiIndex (destination_region, prd_use) -> value
    europe_group: if provided, splits EU vs RoW (two boxes per industry)
    """
    set_house_style()

    s = pd.to_numeric(series, errors="coerce")
    df = s.reset_index()
    val_col = df.columns[-1]
    if value_scale <= 0:
        raise ValueError("value_scale must be positive.")
    df["value"] = pd.to_numeric(df[val_col], errors="coerce") / value_scale
    df["destination_region"] = df["destination_region"].astype(str).str.strip()
    df["prd_use"] = df["prd_use"].astype(str)
    df = df[df["value"] > 0].copy()
    if countries is not None and europe_group is None:
        df = df[df["destination_region"].isin(countries)].copy()

    if industries is None:
        industries = sorted(df["prd_use"].unique().tolist())
    else:
        industries = [str(x) for x in industries]
        df = df[df["prd_use"].isin(industries)].copy()

    data = []
    positions = []
    xticks = []
    ind_order_clean = []
    pos = 1

    # no EU split, single box per industry (alphabetical)
    for ind in sorted(industries):
        sub = df[df["prd_use"] == ind]
        vals = sub["value"].to_numpy()
        if len(vals) == 0:
            continue
        data.append(vals)
        positions.append(pos)
        xticks.append(pos)
        ind_order_clean.append(ind)
        pos += 1.4

    if not data:
        raise ValueError("No data available to plot after filtering industries and regions.")

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    bp = ax.boxplot(
        data,
        positions=positions,
        showfliers=show_fliers,
        patch_artist=True,
    )

    ax.set_xticks(xticks)
    ax.set_xticklabels([tex_safe(i) for i in ind_order_clean], rotation=45, ha="right", fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontweight="bold")

    # style: white boxes with black edges
    for b in bp["boxes"]:
        b.set_facecolor("white")
        b.set_edgecolor("black")
    for median in bp["medians"]:
        median.set_color("#8B0000")

    # highlight selected countries
    if highlight:
        for i, ind in enumerate(ind_order_clean, start=1):
            sub = df[df["prd_use"] == ind]
            for c in highlight:
                v = sub.loc[sub["destination_region"] == c, "value"]
                if not v.empty:
                    val = float(v.iloc[0])
                    ax.scatter(xticks[i - 1], val, color="red", s=20, zorder=3)
                    ax.text(xticks[i - 1], val, c, color="red", fontsize=7, ha="left", va="bottom")

    apply_house_style(ax, grid=True)
    fig.tight_layout()

    if (save is not None) or (close is not None):
        save_and_close(fig=fig, filename=filename, save=save, close=close)
    return fig, ax


def calculate_vn_pj(IO):
    """
    Primary input coefficients v_nj:
        v_nj = VA_nj / X_j
    where:
    - rows n are primary inputs (components of value added)
    - columns j are industries (intermediate-use columns).
    This is the usual v_nj = value added component n per unit of
    industry j output (column-normalised).
    """
    row_prod = IO.index.get_level_values("prd_ava")
    col_use  = IO.columns.get_level_values("prd_use")

    primary_rows = IO.index[row_prod.isin(primary_inputs)]
    ind_rows     = IO.index[~row_prod.isin(primary_inputs)]
    inter_cols   = IO.columns[~col_use.isin(final_codes)]

    rows_for_xj = ind_rows.union(primary_rows)
    Xj = IO.loc[rows_for_xj, inter_cols].astype(float).sum(axis=0)

    V = IO.loc[primary_rows, inter_cols].astype(float)

    Xj = Xj.replace(0.0, np.nan)
    out = V.div(Xj, axis=1).fillna(0.0)
    return out


def validate_mrio_coefficients(
    A: pd.DataFrame,
    V: pd.DataFrame,
    VA: pd.Series,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-8,
) -> pd.DataFrame:
    """Validate column accounting without changing any coefficients."""
    if not A.columns.equals(V.columns):
        raise ValueError("A and V do not have identical industry-column indices.")
    if set(A.index) != set(A.columns):
        raise ValueError("A is not square on the same country-industry index.")
    if set(VA.index) != set(A.columns):
        raise ValueError("VA and MRIO coefficient columns do not have the same industry index.")

    va = pd.to_numeric(VA, errors="coerce").reindex(A.columns)
    if va.isna().any():
        missing = va.index[va.isna()].tolist()[:5]
        raise ValueError(f"VA is missing aligned industry coefficients, e.g. {missing}.")

    codes = V.index.get_level_values("prd_ava").astype(str)

    def _rows(code: str) -> pd.Series:
        mask = codes == str(code)
        return V.loc[mask].sum(axis=0) if mask.any() else pd.Series(0.0, index=V.columns)

    va_expected = sum((_rows(code) for code in ("D1", "B2A3G", "D29X39", "D21X31")),
                      start=pd.Series(0.0, index=V.columns))
    va_difference = va - va_expected
    if not np.allclose(va.to_numpy(), va_expected.to_numpy(), atol=atol, rtol=rtol):
        top = va_difference.abs().sort_values(ascending=False).head(5)
        raise ValueError(
            "VA_MRIO does not equal D1+B2A3G+D29X39+D21X31. "
            f"Largest differences: {top.to_dict()}"
        )

    op = _rows("OP_RES") + _rows("OP_NRES")
    if op.abs().max() > atol:
        warnings.warn(
            "OP_RES/OP_NRES are present in V but excluded from TiVA value added; "
            "this is intentional.",
            RuntimeWarning,
            stacklevel=2,
        )

    residual = 1.0 - A.sum(axis=0) - va
    nonzero_output = (A.abs().sum(axis=0) + V.abs().sum(axis=0)) > atol
    zero_output = residual.index[~nonzero_output]
    if len(zero_output):
        warnings.warn(
            f"{len(zero_output)} MRIO industry columns have zero gross output; "
            f"examples: {list(zero_output[:5])}",
            RuntimeWarning,
            stacklevel=2,
        )

    bad = residual[nonzero_output].abs() > (atol + rtol)
    if bad.any():
        raise ValueError(
            "Material MRIO coefficient-accounting residuals found. "
            f"Largest country-industries: {residual[nonzero_output][bad].abs().nlargest(5).to_dict()}"
        )

    return pd.DataFrame({"residual": residual, "nonzero_output": nonzero_output})

def calculate_L(A, V, C, system_type=1):
    """
    Leontief inverse:
    - Type I (system_type=1): open model
        L = (I - A)^(-1)
    - Type II (system_type=2): simple closure with:
        * 'D1' (compensation of employees) from V
        * 'P3_S14' (household consumption) from C

    Returns:
        L as a DataFrame with matching index/columns.
    """
    industries = [idx for idx in A.index if idx in A.columns]
    if len(industries) == 0:
        raise ValueError("No common industries between A.index and A.columns.")
    A_core = A.loc[industries, industries].astype(float)

    if system_type == 1:
        L = np.linalg.inv(np.eye(len(industries)) - A_core.values)
        return pd.DataFrame(L, index=A_core.index, columns=A_core.columns)

    if system_type == 2:
        # NOTE: Type II is only meaningful for a *single* economy table.
        # We keep it functional for domestic IO tables extracted from MRIO.

        A_ext = A_core.copy()

        va_codes = V.index.get_level_values("prd_ava").astype(str)
        d1_rows = V.index[va_codes == "D1"]
        if len(d1_rows) == 0:
            raise KeyError("No rows with prd_ava == 'D1' found in V.index")

        v = V.loc[d1_rows, industries].astype(float).sum(axis=0).fillna(0.0)
        A_ext.loc["Compensation of Employees"] = v.reindex(A_ext.columns).fillna(0.0).values

        # ---- household consumption shares ----
        if not isinstance(C.columns, pd.MultiIndex):
            raise TypeError("C.columns must be MultiIndex (destination_region, prd_use).")

        # if this is domestic IO, there should be exactly one destination_region != 'ExportROW'
        dests = [d for d in C.columns.get_level_values(0).unique() if str(d) != "ExportROW"]
        if len(dests) == 0:
            raise KeyError("No destination_region found in C columns (excluding ExportROW).")

        d0 = str(dests[0])  # the domestic country
        key = (d0, "P3_S14")
        if key not in C.columns:
            raise KeyError(f"Household column {key} not found in C.columns")

        c = C[key].reindex(industries).astype(float).fillna(0.0)
        A_ext.loc[:, "Con"] = 0.0
        A_ext.loc[industries, "Con"] = c.values
        A_ext.loc["Compensation of Employees", "Con"] = 0.0

        L_ext = np.linalg.inv(np.eye(A_ext.shape[0]) - A_ext.values)
        return pd.DataFrame(L_ext, index=A_ext.index, columns=A_ext.columns)

    raise ValueError(f"system_type {system_type} not recognized")

def calculate_L_truncated(A: pd.DataFrame, K: int = 3) -> pd.DataFrame:
    """
    Truncated Leontief series:
        L_K = I + A + A^2 + ... + A^K

    A must be square on the same industry index/columns (MultiIndex ok).
    """
    industries = [idx for idx in A.index if idx in A.columns]
    if len(industries) == 0:
        raise ValueError("No common industries between A.index and A.columns.")

    A_core = A.loc[industries, industries].astype(float)
    A_mat = A_core.to_numpy()

    I = np.eye(A_mat.shape[0])
    Lk = I.copy()
    An = I.copy()

    for _ in range(K):
        An = An @ A_mat
        Lk = Lk + An

    return pd.DataFrame(Lk, index=A_core.index, columns=A_core.columns)

# ======================================================
# Extract E (Compensations)
# ======================================================


def extract_gva_basic(IO: pd.DataFrame) -> pd.Series:
    """Return country-industry GVA at basic prices in FIGARO million EUR.

    This is an accounting-level amount, not a primary-input coefficient. It
    therefore remains suitable for transparent FTE40 allocation weights and
    C2 reporting. Product taxes (D21X31) are intentionally excluded.
    """
    col_use = IO.columns.get_level_values("prd_use")
    industry_columns = IO.columns[~col_use.isin(list(map(str, final_codes)) + ["*"])]
    row_codes = IO.index.get_level_values("prd_ava").astype(str).str.strip()
    result = pd.Series(0.0, index=industry_columns, dtype=float)
    for code in ("D1", "B2A3G", "D29X39"):
        mask = row_codes == code
        if mask.any():
            result = result.add(IO.loc[mask, industry_columns].sum(axis=0), fill_value=0.0)
    result.name = "FIGARO_GVA"
    return result



from typing import Mapping

def compute_full_effects_mrio(
    results_dict: dict | None,
    country: str | None,
    year: int | None,
    direct_output,
    drop_sectors: list | None = None,
    L1: pd.DataFrame | None = None,
    VA: pd.Series | None = None,
    LB: pd.Series | None = None,
    W: pd.Series | None = None,
):
    """Compute output, GVA, income, and employment effects.

    ``W`` is retained as the public argument name for compatibility, but it
    now contains FTE coefficients expressed as FTE per million EUR of gross
    output.  Employment is therefore obtained by converting output to million
    EUR and multiplying it by ``W``.
    """
    # --- allow MRIO direct mode ---
    if L1 is None or VA is None or LB is None:
        # fallback to old dict behavior
        L1 = results_dict[country]["L1"][year]
        VA = results_dict[country]["GVA"][year]
        LB = results_dict[country]["LB"][year]
        W  = results_dict[country]["W"].get(year, None)

    # normalize direct vector
    d = direct_output["Output"] if (isinstance(direct_output, dict) and "Output" in direct_output) else direct_output
    if isinstance(d, pd.DataFrame):
        d = d.iloc[:, 0]
    d = pd.to_numeric(d, errors="coerce").fillna(0.0).reindex(L1.columns).fillna(0.0)

    if drop_sectors:
        d = d.drop(labels=[s for s in drop_sectors if s in d.index], errors="ignore")

    direct   = d
    total_I  = L1.dot(direct)
    indirect = total_I - direct

    # induced removed but keep structure

    out_df = pd.DataFrame({"Direct": direct, "Indirect": indirect, "Total": total_I}, index=total_I.index)

    VA = pd.to_numeric(VA, errors="coerce").reindex(out_df.index).fillna(0.0)
    LB = pd.to_numeric(LB, errors="coerce").reindex(out_df.index).fillna(0.0)

    gva_df  = out_df.mul(VA, axis=0)
    earn_df = out_df.mul(LB, axis=0)

    if W is not None:
        fte_coefficients = (
            pd.to_numeric(W, errors="coerce")
            .reindex(out_df.index)
            .fillna(0.0)
        )
        # ``out_df`` is expressed in EUR, while the FTE coefficients are
        # expressed per million EUR of output.  Convert the output base before
        # applying the coefficient so Jobs remain in FTE per year.
        jobs_df = out_df.mul(fte_coefficients / 1_000_000.0, axis=0)
    else:
        jobs_df = pd.DataFrame(0.0, index=out_df.index, columns=["Direct","Indirect", "Total"])

    def _pack(df: pd.DataFrame) -> dict:
        return {"Direct": df["Direct"], "Indirect": df["Indirect"], "Total": df["Total"]}

    return {
        "Output":   _pack(out_df),
        "GVA":      _pack(gva_df),
        "Income": _pack(earn_df),
        "Jobs":     _pack(jobs_df),
    }

def summarize_deployments_mrio(
    y_by_deployment: dict[str, pd.Series],
    year: int,
    L1: pd.DataFrame,
    VA: pd.Series,
    LB: pd.Series,
    W: pd.Series | None = None,
) -> pd.DataFrame:
    rows = []

    for dep, y in y_by_deployment.items():
        eff = compute_full_effects_mrio(
            results_dict=None, country=None, year=None,
            direct_output=y,
            L1=L1, VA=VA, LB=LB, W=W
        )

        rows.append({
            "Country": dep,
            "Year": year,
            "Sector": "ALL",

            "Output_Direct":   eff["Output"]["Direct"].sum(),
            "Output_Indirect": eff["Output"]["Indirect"].sum(),
            "Output_Total": eff["Output"]["Total"].sum(),

            "GVA_Direct":      eff["GVA"]["Direct"].sum(),
            "GVA_Indirect":    eff["GVA"]["Indirect"].sum(),
            "GVA_Total":    eff["GVA"]["Total"].sum(),

            "Income_Direct":   eff["Income"]["Direct"].sum(),
            "Income_Indirect": eff["Income"]["Indirect"].sum(),
            "Income_Total": eff["Income"]["Total"].sum(),

            "Jobs_Direct":     eff["Jobs"]["Direct"].sum(),
            "Jobs_Indirect":   eff["Jobs"]["Indirect"].sum(),
            "Jobs_Total":   eff["Jobs"]["Total"].sum(),
        })

    cols_12 = [
        "Output_Direct","Output_Indirect","Output_Total",
        "GVA_Direct","GVA_Indirect", "GVA_Total",
        "Income_Direct","Income_Indirect","Income_Total",
        "Jobs_Direct","Jobs_Indirect", "Jobs_Total"
    ]

    return pd.DataFrame(rows)[["Country","Year","Sector"] + cols_12]


def summarize_deployments_by_origin_mrio(
    y_by_deployment: Mapping[str, pd.Series],
    year: int,
    L1: pd.DataFrame,
    VA: pd.Series,
    LB: pd.Series,
    W: pd.Series | None = None,
    origin_level: str | int | None = None,   # e.g. "region", "origin_region", 0, etc.
) -> pd.DataFrame:
    """
    Returns one row per (Deployment, CountryOrigin) with the same impact columns as
    summarize_deployments_mrio, but split by country/region of origin.

    origin_level:
      - If provided: used as the MultiIndex level name (str) or level position (int).
      - If None: will try to infer a reasonable region/country level name, otherwise uses level 0.
    """
    def _pick_origin_level(idx: pd.Index) -> str | int:
        if origin_level is not None:
            return origin_level

        if isinstance(idx, pd.MultiIndex):
            # Try common level names
            candidates = [
                "origin_region", "origin", "region", "country",
                "from_region", "src_region", "i_region"
            ]
            for c in candidates:
                if c in idx.names:
                    return c
            # Fallback to first level
            return 0

        # Not a MultiIndex: nothing to split beyond the flat labels
        return 0

    def _group_sum(x: pd.Series) -> pd.Series:
        if not isinstance(x, pd.Series):
            x = pd.Series(x)

        idx = x.index
        if isinstance(idx, pd.MultiIndex):
            lvl = _pick_origin_level(idx)
            return x.groupby(level=lvl).sum()
        else:
            # If flat index, treat each label as "origin"
            return x.groupby(idx).sum()

    rows = []

    for dep, y in y_by_deployment.items():
        eff = compute_full_effects_mrio(
            results_dict=None, country=None, year=None,
            direct_output=y,
            L1=L1, VA=VA, LB=LB, W=W
        )

        # Build origin-split series for each metric/component
        out_dir = _group_sum(eff["Output"]["Direct"])
        out_ind = _group_sum(eff["Output"]["Indirect"])
        out_tot = _group_sum(eff["Output"]["Total"])

        gva_dir = _group_sum(eff["GVA"]["Direct"])
        gva_ind = _group_sum(eff["GVA"]["Indirect"])
        gva_tot = _group_sum(eff["GVA"]["Total"])

        ear_dir = _group_sum(eff["Income"]["Direct"])
        ear_ind = _group_sum(eff["Income"]["Indirect"])
        ear_tot = _group_sum(eff["Income"]["Total"])

        job_dir = _group_sum(eff["Jobs"]["Direct"])
        job_ind = _group_sum(eff["Jobs"]["Indirect"])
        job_tot = _group_sum(eff["Jobs"]["Total"])

        # Union of all origins present across series
        origins = (
            out_dir.index.union(out_ind.index).union(out_tot.index)
                  .union(gva_dir.index).union(gva_ind.index).union(gva_tot.index)
                  .union(ear_dir.index).union(ear_ind.index).union(ear_tot.index)
                  .union(job_dir.index).union(job_ind.index).union(job_tot.index)
        )

        for origin in origins:
            rows.append({
                "Deployment": dep,
                "Country": origin,     # origin country/region
                "Year": year,
                "Sector": "ALL",

                "Output_Direct":   float(out_dir.get(origin, 0.0)),
                "Output_Indirect": float(out_ind.get(origin, 0.0)),
                "Output_Total": float(out_tot.get(origin, 0.0)),

                "GVA_Direct":      float(gva_dir.get(origin, 0.0)),
                "GVA_Indirect":    float(gva_ind.get(origin, 0.0)),
                "GVA_Total": float(gva_tot.get(origin, 0.0)),

                "Income_Direct":   float(ear_dir.get(origin, 0.0)),
                "Income_Indirect": float(ear_ind.get(origin, 0.0)),
                "Income_Total": float(ear_tot.get(origin, 0.0)),

                "Jobs_Direct":     float(job_dir.get(origin, 0.0)),
                "Jobs_Indirect":   float(job_ind.get(origin, 0.0)),
                "Jobs_Total": float(job_tot.get(origin, 0.0)),
            })

    cols_12 = [
        "Output_Direct","Output_Indirect","Output_Total",
        "GVA_Direct","GVA_Indirect","GVA_Total",
        "Income_Direct","Income_Indirect","Income_Total",
        "Jobs_Direct","Jobs_Indirect","Jobs_Total"
    ]
    return pd.DataFrame(rows)[["Deployment", "Country", "Year", "Sector"] + cols_12]


def compute_all_results(
    y_shock_lvl: dict,
    y_shock_sub_lvl: dict,
    year: int,
    L1_MRIO: pd.DataFrame,
    VA_MRIO: pd.Series,
    LB_MRIO: pd.Series,
    W_MRIO: pd.Series,
) -> tuple[dict, dict, dict, dict]:
    """
    Compute all MRIO results for component groups and subcomponents.

    Returns a tuple of 4 dictionaries:
        (results_lvl, results_sub_lvl, results_by_origin_lvl, results_by_origin_sub_lvl)

    Parameters
    ----------
    y_shock_lvl : dict
        Nested dict {level: {component_group: {deployment: y_series}}}
    y_shock_sub_lvl : dict
        Nested dict {level: {(component_group, subcomponent): {deployment: y_series}}}
    year : int
        Year for the analysis
    L1_MRIO, VA_MRIO, LB_MRIO, W_MRIO : pd.DataFrame or pd.Series
        MRIO matrices for Leontief, Value Added, Labor, and Wages
    levels : tuple
        Levels to process (default: LEVELS = (0, 1, 2))

    Returns
    -------
    results_lvl : dict
        {level: {component_group: DataFrame}} - Results aggregated across origins
    results_sub_lvl : dict
        {level: {(component_group, subcomponent): DataFrame}} - Subcomponent results
    results_by_origin_lvl : dict
        {level: {component_group: DataFrame}} - Results split by origin
    results_by_origin_sub_lvl : dict
        {level: {(component_group, subcomponent): DataFrame}} - Subcomponent results by origin
    """
    # ---- NOT BY ORIGIN ----
    results_lvl = {
        lvl: {cg: summarize_deployments_mrio(y, year, L1_MRIO, VA_MRIO, LB_MRIO, W_MRIO)
              for cg, y in y_shock_lvl[lvl].items()}
        for lvl in LEVELS
    }

    results_sub_lvl = {
        lvl: {k: summarize_deployments_mrio(y, year, L1_MRIO, VA_MRIO, LB_MRIO, W_MRIO)
              for k, y in y_shock_sub_lvl[lvl].items()}
        for lvl in LEVELS
    }

    # ---- BY ORIGIN ----
    results_by_origin_lvl = {
        lvl: {cg: summarize_deployments_by_origin_mrio(y, year, L1_MRIO, VA_MRIO, LB_MRIO, W_MRIO)
              for cg, y in y_shock_lvl[lvl].items()}
        for lvl in LEVELS
    }

    results_by_origin_sub_lvl = {
        lvl: {k: summarize_deployments_by_origin_mrio(y, year, L1_MRIO, VA_MRIO, LB_MRIO, W_MRIO)
              for k, y in y_shock_sub_lvl[lvl].items()}
        for lvl in LEVELS
    }

    return results_lvl, results_sub_lvl, results_by_origin_lvl, results_by_origin_sub_lvl


def validate_global_mrio_results(
    y_by_deployment: Mapping[str, pd.Series],
    results_total: pd.DataFrame,
    results_by_origin_total: pd.DataFrame,
    component_results: Mapping,
    L1: pd.DataFrame,
    A: pd.DataFrame,
    VA: pd.Series,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-9,
) -> None:
    """Validate aggregate Type I MRIO identities once per deployment."""
    if not isinstance(L1.index, pd.MultiIndex) or not isinstance(L1.columns, pd.MultiIndex):
        return
    regions = set(map(str, L1.columns.get_level_values(0)))
    if "ROW" not in regions or len(regions) < 2:
        return
    if list(A.index) != list(L1.index) or list(A.columns) != list(L1.columns):
        raise ValueError("A and L1 industry indices are not aligned.")

    va = pd.to_numeric(VA, errors="coerce").reindex(A.columns)
    residual = 1.0 - A.sum(axis=0) - va

    def _close(label: str, left: float, right: float) -> None:
        if not np.isclose(left, right, atol=atol, rtol=rtol):
            raise ValueError(f"Global MRIO accounting check failed for {label}: {left} vs {right}.")

    def _result_row(df: pd.DataFrame, deployment: str) -> pd.Series:
        rows = df[df["Country"].astype(str) == str(deployment)]
        if len(rows) != 1:
            raise ValueError(f"Expected one aggregate result row for deployment {deployment}, found {len(rows)}.")
        return rows.iloc[0]

    for deployment, y0 in y_by_deployment.items():
        y = pd.to_numeric(y0, errors="coerce").fillna(0.0).reindex(L1.columns).fillna(0.0)
        x = L1.dot(y)
        x_col = x.reindex(A.columns)
        x_from_a = A.dot(x_col) + y.reindex(A.columns).fillna(0.0)
        if not np.allclose(x.to_numpy(), L1.dot(y).to_numpy(), atol=atol, rtol=rtol):
            raise ValueError(f"Global MRIO check failed for x=L@y in deployment {deployment}.")
        if not np.allclose(x_col.to_numpy(), x_from_a.to_numpy(), atol=atol, rtol=rtol):
            raise ValueError(f"Global MRIO check failed for x=A@x+y in deployment {deployment}.")

        row = _result_row(results_total, deployment)
        _close("Output split", float(row["Output_Direct"] + row["Output_Indirect"]), float(row["Output_Total"]))
        _close("GVA split", float(row["GVA_Direct"] + row["GVA_Indirect"]), float(row["GVA_Total"]))
        _close("Output total", float(x.sum()), float(row["Output_Total"]))
        _close("GVA total", float((va * x_col).sum()), float(row["GVA_Total"]))

        direct_output = float(y.sum())
        gva_total = float((va * x_col).sum())
        gva_gap = direct_output - gva_total
        residual_gap = float((residual * x_col).sum())
        unexplained = gva_gap - residual_gap
        if not np.isclose(unexplained, 0.0, atol=atol, rtol=rtol):
            raise ValueError(
                f"Unexplained global GVA residual for {deployment}: "
                f"{unexplained:,.6f}."
            )
        if not np.isclose(gva_gap, 0.0, atol=atol, rtol=rtol):
            top = (residual * x_col).abs().sort_values(ascending=False).head(5)
            relative = abs(gva_gap) / abs(direct_output) if direct_output else 0.0
            warnings.warn(
                f"Global GVA reconciliation for {deployment}: absolute discrepancy="
                f"{abs(gva_gap):,.6f}, relative discrepancy={relative:.6%}; "
                f"explained by accounting residuals, largest country-industries={top.to_dict()}",
                RuntimeWarning,
                stacklevel=2,
            )

        regional = results_by_origin_total[
            results_by_origin_total["Deployment"].astype(str) == str(deployment)
        ]
        for col in ("Output_Direct", "Output_Indirect", "Output_Total", "GVA_Direct", "GVA_Indirect", "GVA_Total"):
            _close(f"regional sum {col}", float(regional[col].sum()), float(row[col]))

        component_frames = [
            frame for key, frame in component_results.items()
            if not (key == "Total" or (isinstance(key, tuple) and key and key[0] == "Total"))
        ]
        for col in ("Output_Direct", "Output_Indirect", "Output_Total", "GVA_Direct", "GVA_Indirect", "GVA_Total"):
            component_sum = sum(float(_result_row(frame, deployment)[col]) for frame in component_frames)
            _close(f"component sum {col}", component_sum, float(row[col]))


def make_final_demand_vector(
    MRIO: pd.DataFrame,
    final_cost_df: pd.DataFrame,
    deployment_country: str,
    component_group: str | None = None,
    sub_component: str | None = None,
    level: int = 1,
    lifetime_years: int=25,
    mode: str = "total",
    gross_output: pd.Series | None = None,
) -> pd.Series:
    if mode not in {"total", "annual"}:
        raise ValueError("mode must be either 'total' or 'annual'.")

    idx = MRIO.index
    if not isinstance(idx, pd.MultiIndex) or idx.nlevels < 2:
        raise ValueError("MRIO.index must be a MultiIndex with at least 2 levels (origin, industry).")

    origin = idx.get_level_values(0).astype(str)
    prd_use = idx.get_level_values(1).astype(str)
    if "ind_codes" in globals() and ind_codes:
        ind_set = {str(x) for x in ind_codes}
        prd_set = set(prd_use.unique())
        # If codes don't intersect (e.g., C10-12 vs C10T12), try a simple normalization.
        if not (ind_set & prd_set):
            canon = {c.replace("T", "-"): c for c in prd_set if "T" in c}
            ind_set = {canon.get(c, c) for c in ind_set}
        mold = idx[(origin != "ExportROW") & prd_use.isin(ind_set)]
        # Fallback: avoid empty mold if codes still don't match
        if len(mold) == 0:
            mold = idx[origin != "ExportROW"]
    else:
        mold = idx[origin != "ExportROW"]

    cg_col = f"component_group_L{level}"
    sc_col = f"sub_component_L{level}"
    if cg_col not in final_cost_df.columns or sc_col not in final_cost_df.columns:
        raise KeyError(f"final_cost_df must contain {cg_col} and {sc_col}")

    df = final_cost_df[final_cost_df["deployment_country"].astype(str) == str(deployment_country)].copy()

    if component_group not in (None, "Total"):
        df = df[df[cg_col].astype(str) == str(component_group)].copy()

    if sub_component is not None:
        df = df[df[sc_col].astype(str) == str(sub_component)].copy()

    df = df[df["value_alloc"].astype(float) != 0.0].copy()
    df["industry_code"] = df["industry_code"].astype(str).str.strip()
    df["origin_country"] = df["origin_country"].astype(str).str.strip()

    # Align industry codes with MRIO index (handle optional CPA_ prefix)
    mrio_codes = {str(x) for x in prd_use.unique()}
    if mrio_codes:
        def _align_ind_code(c: str) -> str:
            c = str(c).strip()
            if c in mrio_codes:
                return c
            if c.startswith("CPA_"):
                c2 = c[4:]
                if c2 in mrio_codes:
                    return c2
            else:
                c2 = "CPA_" + c
                if c2 in mrio_codes:
                    return c2
            return c
        df["industry_code"] = df["industry_code"].map(_align_ind_code)

    by_src_ind = df.groupby(["origin_country", "industry_code"])["value_alloc"].sum()

    allocated_total = float(by_src_ind.sum())
    if mode == "annual":
        by_src_ind = by_src_ind / float(lifetime_years)
        allocated_total /= float(lifetime_years)

    y = pd.Series(0.0, index=mold, name="final_demand")

    by_src_ind.index = pd.MultiIndex.from_tuples(
        by_src_ind.index, names=[idx.names[0], idx.names[1]]
    )
    missing = by_src_ind.index.difference(y.index)
    if len(missing):
        lost = float(by_src_ind.reindex(missing).sum())
        raise ValueError(
            f"Final-demand mapping lost {lost:,.6f} for deployment "
            f"{deployment_country}; missing country-industry combinations: "
            f"{list(missing[:5])}"
        )

    mapped = by_src_ind.reindex(y.index).fillna(0.0)
    if not np.isclose(float(mapped.sum()), allocated_total, atol=1e-6, rtol=1e-10):
        raise ValueError(
            f"Final-demand mapping total mismatch for {deployment_country}: "
            f"allocated={allocated_total:,.6f}, mapped={float(mapped.sum()):,.6f}."
        )
    y.update(mapped)

    if gross_output is None:
        gross_output_use = pd.to_numeric(MRIO.sum(axis=1), errors="coerce")
    else:
        gross_output_use = pd.to_numeric(gross_output, errors="coerce")
    gross_output_use = gross_output_use.reindex(y.index).fillna(0.0)
    zero_output = y.index[(y.abs() > 1e-12) & (gross_output_use.abs() <= 1e-12)]
    if len(zero_output):
        zero_value = float(y.reindex(zero_output).sum())
        warnings.warn(
            f"Final demand for {deployment_country} includes zero-output "
            f"country-industries totaling {zero_value:,.6f}: {list(zero_output[:5])}",
            RuntimeWarning,
            stacklevel=2,
        )
    return y


_ZERO_OUTPUT_INDUSTRY_FALLBACKS = {
    ("JP", "C33"): "C27",
    ("CA", "C33"): "C27",
}


def _apply_zero_output_industry_fallbacks(
    final_cost_df: pd.DataFrame,
    mrio: pd.DataFrame,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Apply the two explicit zero-output mappings before demand-vector construction."""
    if not isinstance(mrio.index, pd.MultiIndex) or mrio.index.nlevels < 2:
        raise ValueError("MRIO.index must be a country-industry MultiIndex for zero-output fallbacks.")

    country = mrio.index.get_level_values(0).astype(str).str.strip()
    industry = mrio.index.get_level_values(1).astype(str).str.strip()
    industry_base = industry.str.replace(r"^CPA_", "", regex=True)
    gross_output = pd.to_numeric(mrio.sum(axis=1), errors="coerce").fillna(0.0)
    code_values = set(final_cost_df["industry_code"].astype(str).str.strip())
    corrected = final_cost_df.copy()
    summaries = []

    for (country_code, source), target in _ZERO_OUTPUT_INDUSTRY_FALLBACKS.items():
        source_code = source if source in code_values else f"CPA_{source}" if f"CPA_{source}" in code_values else source
        target_code = f"CPA_{target}" if source_code.startswith("CPA_") else target
        source_output = float(gross_output[(country == country_code) & (industry_base == source)].sum())
        target_output = float(gross_output[(country == country_code) & (industry_base == target)].sum())

        if source_output > 1e-12:
            continue
        if target_output <= 1e-12:
            raise ValueError(
                f"Cannot apply fallback {country_code}-{source}->{target}: "
                "replacement output is not positive."
            )

        mask = (
            corrected["origin_country"].astype(str).str.strip().eq(country_code)
            & corrected["industry_code"].astype(str).str.strip().eq(source_code)
        )
        amount = float(corrected.loc[mask, "value_alloc"].sum())
        if mask.any():
            corrected.loc[mask, "industry_code"] = target_code
            summaries.append((country_code, source_code, target_code, amount))

    if summaries:
        keys = [column for column in corrected.columns if column != "value_alloc"]
        corrected = corrected.groupby(keys, as_index=False, dropna=False)["value_alloc"].sum()

    return corrected


def _lvl_cols(lvl: int):
    return f"component_group_L{lvl}", f"sub_component_L{lvl}"

def _pairs_by_level(df: pd.DataFrame, lvl: int):
    cg, sc = _lvl_cols(lvl)
    return [tuple(x) for x in df[[cg, sc]].astype(str).drop_duplicates().to_numpy()]

def _sum_y_by_dep(d: dict[str, dict[str, pd.Series]]) -> dict[str, pd.Series]:
    # d: cg -> {dep -> Series}
    deps = list(next(iter(d.values())).keys())
    # pick any Series as template for a zero start (keeps index/dtype)
    s0 = next(iter(next(iter(d.values())).values()))
    z = s0 * 0.0
    return {dep: sum((d[cg][dep] for cg in d), start=z.copy()) for dep in deps}


def build_y_shock_levels(
    df: pd.DataFrame,
    core: List[str],
    mrio: pd.DataFrame,
    *,
    verbose: bool = True,
):
    df = _apply_zero_output_industry_fallbacks(df, mrio, verbose=verbose)
    y_cg, y_sub = {}, {}
    for lvl in LEVELS:
        cg_col, sc_col = _lvl_cols(lvl)
        cgs = sorted(df[cg_col].astype(str).unique())

        y_cg[lvl] = {
            cg0: {dep: make_final_demand_vector(mrio, df, dep, cg0, None, level=lvl, mode="annual")
                  for dep in core}
            for cg0 in cgs
        }

        # ✅ add Total (sum of component-level shocks)
        y_cg[lvl]["Total"] = _sum_y_by_dep(y_cg[lvl])

        y_sub[lvl] = {
            (cg0, sub0): {dep: make_final_demand_vector(mrio, df, dep, cg0, sub0, level=lvl, mode="annual")
                          for dep in core}
            for (cg0, sub0) in _pairs_by_level(df, lvl)
        }

        # ✅ optional Total key for sub-level (one bucket)
        y_sub[lvl][("Total", "ALL")] = {dep: y_cg[lvl]["Total"][dep] for dep in core}

    return y_cg, y_sub


# ============================================================================
# MRIO Analysis Helpers (tables / matrices)
# ============================================================================






def cap_label(s: str, max_chars: int) -> str:
    s = str(s)
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"
