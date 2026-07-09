"""FIGARO MRIO loading, transformation, and impact-calculation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap
import re

from modules.latex_style import apply_house_style, save_and_close, tex_safe, tex_bold, set_house_style
from modules.country import COLOR_MAP

from modules.config import DATASETS_DIR, HELPERS_DIR

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
# Paths (centralized in modules.config)
# --------------------------------------------------------------
FLATFILE_PATH = DATASETS_DIR / "flatfile_eu-ic-io_prod-by-prod_25ed_2023.csv"
PARQUET_PATH = DATASETS_DIR / "flatfile_eu-ic-io_prod-by-prod_25ed_2023.parquet"
MRIO_CACHE_PATH = DATASETS_DIR / "MRIO_2023_CORE_ROW.parquet"

# Alternative: industry-by-industry flatfile/parquet + cache
FLATFILE_PATH_IND = DATASETS_DIR / "flatfile_eu-ic-io_ind-by-ind_25ed_2023.csv"
PARQUET_PATH_IND = DATASETS_DIR / "flatfile_eu-ic-io_ind-by-ind_25ed_2023.parquet"
MRIO_CACHE_PATH_IND = DATASETS_DIR / "MRIO_2023_CORE_ROW_ind_by_ind.parquet"

# Optional URL for industry-by-industry source (CIRCABC details page)
FLATFILE_URL = (
    "https://circabc.europa.eu/ui/group/cec66924-a924-4f91-a0ef-600a0531e3ba/"
    "library/ff0e2b4f-b572-424a-8f09-4b484ceabc7e/details?download=true"
)
INDUSTRY_MATRIX_URL = (
    "https://circabc.europa.eu/ui/group/cec66924-a924-4f91-a0ef-600a0531e3ba/"
    "library/56302478-664b-4a93-aa65-3d1ca73b1f4b/details?download=true"
)
FIGARO_DATABASE_URL = "https://ec.europa.eu/eurostat/web/esa-supply-use-input-output-tables/database"

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
        class_codes[code] = pd.read_excel(
            helpers_dir / f"class_{code_prefix}_{code}.xlsx",
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
            "CIRCABC direct links can change or require browser access.\n"
            f"Download the FIGARO 2025 flatfile manually from {FIGARO_DATABASE_URL}\n"
            f"and place it at: {path}"
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
        "CIRCABC direct links can change or require browser access.\n"
        f"Download the FIGARO 2025 flatfile manually from {FIGARO_DATABASE_URL}\n"
        f"and place it at: {path}"
    )

def simplify_regions(df: pd.DataFrame, *, CORE: set[str], EU_OFC: set[str]) -> pd.DataFrame:
    def classify(code):
        if pd.isna(code):
            return "ROW"
        c = str(code).strip()
        if c in EU_OFC:
            return "EU-OFC"
        return c if c in CORE else "ROW"

    out = df.copy()
    out["origin_region"] = out["origin"].apply(classify)
    out["destination_region"] = out["destination"].apply(classify)
    out["year"] = 2023
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

def collapse_row(mrio):
    row_mask = mrio.index.get_level_values("origin_region") == "ROW"
    if row_mask.any():
        mrio_no_row_rows = mrio[~row_mask]
        row_totals = mrio[row_mask].sum()
        row_index = pd.MultiIndex.from_tuples([("ImportROW", "*")], names=mrio.index.names)
        row_df = pd.DataFrame([row_totals], index=row_index, columns=mrio.columns)
        mrio = pd.concat([mrio_no_row_rows, row_df])
    
    col_mask = mrio.columns.get_level_values("destination_region") == "ROW"
    if col_mask.any():
        mrio_no_row_cols = mrio.loc[:, ~col_mask]
        col_totals = mrio.loc[:, col_mask].sum(axis=1)
        col_index = pd.MultiIndex.from_tuples([("ExportROW", "*")], names=mrio.columns.names)
        col_df = pd.DataFrame(col_totals, index=mrio.index, columns=col_index)
        mrio = pd.concat([mrio_no_row_cols, col_df], axis=1)
    
    return mrio

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
    *, CORE: set[str], EU_OFC: set[str],
    include_eu_ofc: bool = True,
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
    cache_path = MRIO_CACHE_PATH_IND if use_industry_matrix else MRIO_CACHE_PATH
    flat_path = FLATFILE_PATH_IND if use_industry_matrix else FLATFILE_PATH
    parquet_path = PARQUET_PATH_IND if use_industry_matrix else PARQUET_PATH

    if use_cache and cache_path.exists() and not refresh_cache:
        print(f"Loading cached MRIO: {cache_path.name}")
        mrio = load_mrio_cache(cache_path)
    else:
        print("Loading flatfile/parquet...")
        if download_matrix and (not flat_path.exists()):
            label = "industry-by-industry" if use_industry_matrix else "product-by-product"
            print(f"Downloading {label} flatfile -> {flat_path.name}")
            try:
                download_file(INDUSTRY_MATRIX_URL if use_industry_matrix else FLATFILE_URL, flat_path)
            except FileNotFoundError as exc:
                if use_cache and cache_path.exists():
                    print("Download failed; loading existing MRIO cache instead.")
                    print(f"Cache used: {cache_path.name}")
                    mrio = load_mrio_cache(cache_path)
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
                raise exc
        df = load_data(flat_path, parquet_path, save_raw_parquet=save_raw_parquet)
        df = simplify_regions(df, CORE=CORE, EU_OFC=(EU_OFC if include_eu_ofc else set()))

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
        A_ij = Z_ij / X_i
    where Z is the inter-industry block and X_i is total output
    of industry i (row i sum over all uses).
    """
    row_prod = IO.index.get_level_values("prd_ava")
    col_use  = IO.columns.get_level_values("prd_use")

    primary_ext = list(primary_inputs) + ["*"]
    ind_rows = IO.index[~row_prod.isin(primary_ext)]

    final_ext = list(final_codes) + ["*"]
    ind_cols = IO.columns[~col_use.isin(final_ext)]

    Z = IO.loc[ind_rows, ind_cols].astype(float)
    Xi = IO.loc[ind_rows].astype(float).sum(axis=1)
    Xi = Xi.replace(0.0, np.nan)  # avoid division by zero

    return Z.div(Xi, axis=0).fillna(0.0)

def select_A(
    A_global: np.ndarray,
    scope: Scope = "global",
    local_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Return the technical coefficient matrix to use.

    - scope='global': returns A_global unchanged.
    - scope='local': returns a submatrix A_local = A[local, local].

    local_indices: list/array of indices corresponding to the
    industries in the local country/region.
    """
    if scope == "global":
        return A_global

    if scope == "local":
        if local_indices is None:
            raise ValueError("local_indices must be provided for scope='local'")
        idx = np.asarray(local_indices)
        return A_global[np.ix_(idx, idx)]

    raise ValueError(f"Unknown scope: {scope}")

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


def compute_eu27_labor_capital_ratio(
    V_MRIO: pd.DataFrame,
    EU27: set[str],
    *,
    origin_level: str = "origin_region",
    dest_level: str = "destination_region",
    prd_use_level: str = "prd_use",
    prd_ava_level: str = "prd_ava",
    labor_code: str = "D1",
    capital_code: str = "B2A3G",
    dest_scope: str = "eu27",  # "eu27" or "all"
    agg: str = "mean",         # "mean" or "sum"
    include_prd_use: list[str] | None = None,
):
    """
    Build EU27-average labor/capital ratio by industry (prd_use).
    - Sum D1 and B2A3G across EU27 origin regions to get vectors over columns.
    - Optionally restrict destination_region to EU27, then aggregate by prd_use.
    Returns DataFrame with columns: labor, capital, ratio.
    """
    if not isinstance(V_MRIO.index, pd.MultiIndex):
        raise TypeError("V_MRIO.index must be a MultiIndex.")
    if not isinstance(V_MRIO.columns, pd.MultiIndex):
        raise TypeError("V_MRIO.columns must be a MultiIndex.")

    idx = V_MRIO.index
    cols = V_MRIO.columns

    if origin_level not in idx.names or prd_ava_level not in idx.names:
        raise KeyError(f"V_MRIO.index must include levels '{origin_level}' and '{prd_ava_level}'.")
    if dest_level not in cols.names or prd_use_level not in cols.names:
        raise KeyError(f"V_MRIO.columns must include levels '{dest_level}' and '{prd_use_level}'.")

    origin = idx.get_level_values(origin_level).astype(str)
    prd_ava = idx.get_level_values(prd_ava_level).astype(str)
    prd_ava_norm = prd_ava.str.strip().str.upper()

    def _sum_rows(code: str) -> pd.Series:
        code = str(code).strip().upper()
        eu_set = {str(x).strip() for x in EU27}

        # prefer EU27 origins if present; otherwise fall back to all origins
        has_eu_orig = origin.isin(eu_set).any()

        def _mk_mask(use_eu: bool):
            base = prd_ava_norm == code
            if not base.any():
                base = prd_ava_norm.str.startswith(code) | prd_ava_norm.str.contains(code)
            if use_eu:
                return base & origin.isin(eu_set)
            return base

        mask = _mk_mask(use_eu=has_eu_orig)
        if not mask.any():
            # fallback to all origins if EU27 filter was too strict
            mask = _mk_mask(use_eu=False)

        if not mask.any():
            sample = sorted(set(prd_ava_norm.unique()))[:30]
            raise KeyError(
                f"No rows found for prd_ava='{code}'. "
                f"Sample prd_ava values: {sample}"
            )
        return V_MRIO.loc[mask].sum(axis=0)

    labor_vec = _sum_rows(labor_code)
    cap_vec = _sum_rows(capital_code)

    df = pd.DataFrame({"labor": labor_vec, "capital": cap_vec})

    if dest_scope == "eu27":
        dest = df.index.get_level_values(dest_level).astype(str)
        df = df[dest.isin({str(x) for x in EU27})]

    if include_prd_use is not None:
        keep = {str(x) for x in include_prd_use}
        df = df[df.index.get_level_values(prd_use_level).astype(str).isin(keep)]

    df["ratio"] = df["labor"] / df["capital"].replace(0, np.nan)

    if agg == "sum":
        out = df.groupby(level=prd_use_level)[["labor", "capital"]].sum()
        out["ratio"] = out["labor"] / out["capital"].replace(0, np.nan)
    else:
        out = df.groupby(level=prd_use_level)[["labor", "capital", "ratio"]].mean()

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out
def plot_lb_box_all_industries(
    series: pd.Series,
    *,
    industries: list[str] | None = None,
    europe_group: set[str] | None = None,
    countries: set[str] | None = None,
    stat: str = "median",
    highlight: tuple[str, ...] = (),
    show_fliers: bool = False,
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
    df["value"] = pd.to_numeric(df[val_col], errors="coerce")
    df["destination_region"] = df["destination_region"].astype(str).str.strip()
    df["prd_use"] = df["prd_use"].astype(str)
    df = df[df["value"] > 0].copy()
    # Drop EU-OFC entirely
    df = df[~df["destination_region"].isin({"EU-OFC", "EU_OFC"})].copy()

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
def extract_E(IO: pd.DataFrame, row_name: str = "D1") -> pd.Series:
    """
    Extract 'Compensation of employees' (row prd_ava == row_name) over industry columns.
    Returns a Series indexed by MRIO industry columns (destination_region, prd_use, ...).
    """
    col_use = IO.columns.get_level_values("prd_use")
    final_ext = list(map(str, final_codes)) + ["*"]  # expects final_codes in scope

    ind_col_mask = ~col_use.isin(final_ext)
    industry_columns = IO.columns[ind_col_mask]

    row_prod = IO.index.get_level_values("prd_ava").astype(str).str.strip()
    row_mask = (row_prod == str(row_name).strip())
    if not row_mask.any():
        raise KeyError(f"No row found with prd_ava == '{row_name}'")

    E_data = IO.loc[row_mask, industry_columns]
    if len(E_data) > 1:
        E_data = E_data.sum(axis=0)
    if isinstance(E_data, pd.DataFrame):
        E_data = E_data.iloc[0] if len(E_data) == 1 else E_data.sum()

    return E_data

def deep_round_share_colsum(A: pd.DataFrame, K: int = 3) -> float:
    industries = [idx for idx in A.index if idx in A.columns]
    if len(industries) == 0:
        raise ValueError("No common industries between A.index and A.columns.")

    A_core = A.loc[industries, industries].astype(float)
    A_mat = A_core.to_numpy()
    I = np.eye(A_mat.shape[0])

    L_full = np.linalg.inv(I - A_mat)
    L_K = calculate_L_truncated(A_core, K=K).to_numpy()

    # total requirement per column (final demand in each industry)
    m_full = L_full.sum(axis=0)
    m_trunc = L_K.sum(axis=0)

    # share of mass beyond K using L1 (sum) over columns
    full = np.sum(m_full)
    trunc = np.sum(m_trunc)
    return float(max(0.0, 1.0 - trunc / full))


import pandas as pd
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
        wages = pd.to_numeric(W, errors="coerce").reindex(out_df.index).fillna(0.0).replace(0.0, np.nan)
        jobs_df = earn_df.divide(wages, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
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


def make_final_demand_vector(
    MRIO: pd.DataFrame,
    final_cost_df: pd.DataFrame,
    deployment_country: str,
    component_group: str | None = None,
    sub_component: str | None = None,
    level: int = 1,
    lifetime_years: int=25,
    mode: str = "total",
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

    if mode == "annual":
        by_src_ind = by_src_ind / float(lifetime_years)

    y = pd.Series(0.0, index=mold, name="final_demand")

    by_src_ind.index = pd.MultiIndex.from_tuples(
        by_src_ind.index, names=[idx.names[0], idx.names[1]]
    )
    y.update(by_src_ind.reindex(y.index).fillna(0.0))
    return y

def pick_lvl(d: dict, lvl: int):
    return {} if not d else (d[lvl] if isinstance(next(iter(d)), int) else d)

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


def build_y_shock_levels(df: pd.DataFrame, core: List[str],mrio: pd.DataFrame):
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


def industry_country_matrix(
    detailed_df: pd.DataFrame,
    *,
    deployment: str,
    metric: str,                    # "Output" | "GVA" | "Income" | "Jobs"
    effect: str,                    # "Direct" | "Indirect" | "Total"
    year: int | None = None,
    industry_col: str | None = None,  # "prd_ava_label" or "prd_ava"
    top_industries: int | None = None, # keep top N industries by sum (optional)
    top_countries: int | None = None,  # keep top N countries by sum (optional)
    round_0: bool = True,
    scale_money_to_kEUR: bool = True,
) -> pd.DataFrame:
    """
    Returns a pivot matrix with:
      - rows: industries
      - cols: origin countries
      - values: chosen metric & effect (absolute)
    """
    col = f"{metric}_{effect}"

    df = detailed_df.loc[detailed_df["Deployment"] == deployment].copy()
    if year is not None and "Year" in df.columns:
        df = df.loc[df["Year"] == year].copy()
    if df.empty:
        raise ValueError(
            f"No rows for Deployment == {deployment!r}" + (f" and Year == {year}" if year is not None else "")
        )

    ind_col = industry_col or ("prd_ava_label" if "prd_ava_label" in df.columns else "prd_ava")

    # Pivot: industry x country
    M = (
        df.pivot_table(
            index=ind_col,
            columns="origin_country",
            values=col,
            aggfunc="sum",
            fill_value=0.0,
            observed=False
        )
    )

    # Scale to k€ for monetary metrics
    if metric != "Jobs" and scale_money_to_kEUR:
        M = M / 1_000.0

    # Drop all-zero rows/cols
    M = M.loc[M.sum(axis=1) != 0, M.sum(axis=0) != 0].copy()

    # Optional trimming (top industries / countries by total contribution)
    if top_industries is not None and len(M.index) > top_industries:
        keep_i = M.sum(axis=1).sort_values(ascending=False).head(top_industries).index
        M = M.loc[keep_i]
    if top_countries is not None and len(M.columns) > top_countries:
        keep_c = M.sum(axis=0).sort_values(ascending=False).head(top_countries).index
        M = M[keep_c]

    # Round
    if round_0:
        M = M.round(0)

    # Sort rows/cols by totals (nice default)
    M = M.loc[M.sum(axis=1).sort_values(ascending=False).index,
              M.sum(axis=0).sort_values(ascending=False).index]

    return M


def detailed_impacts_by_origin_sector(
    y_by_deployment,
    *,
    year,
    L1,
    VA,
    LB,
    W,
    ind_labels=None
):
    rows = []
    for dep, y in y_by_deployment.items():
        eff = compute_full_effects_mrio(None, None, None, y, L1=L1, VA=VA, LB=LB, W=W)

        # stack metrics/types into one df indexed by (origin, prd_ava)
        s = {}
        for met in ["Output", "GVA", "Income", "Jobs"]:
            for t in ["Direct", "Indirect", "Total"]:
                s[f"{met}_{t}"] = eff[met][t]
        df = pd.DataFrame(s)
        df = df[df.abs().sum(1) > 0].reset_index()
        df.columns = ["origin_country", "prd_ava"] + list(df.columns[2:])

        if ind_labels is not None:
            df["prd_ava_label"] = df["prd_ava"].astype(str).map(ind_labels).fillna(df["prd_ava"].astype(str))

        df.insert(0, "Deployment", dep)
        df.insert(1, "Year", year)
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def cap_label(s: str, max_chars: int) -> str:
    s = str(s)
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def map_and_cap_industry(
    s: pd.Series,
    ind_labels: dict,
    max_chars: int = 32,
) -> pd.Series:
    return (
        s.map(ind_labels).fillna(s)
         .map(lambda x: cap_label(x, max_chars))
    )


def trace_indirect_sources(
    *,
    target_origin: str,
    target_cpa: str,
    deployment: str,
    y_by_deployment: dict[str, pd.Series],   # e.g. y_shock_lvl[lvl]["Total"]
    L1: pd.DataFrame,
    ind_labels: dict | None = None,
    top_n: int = 20,
    exclude_self: bool = True,
    max_chars: int = 34,
) -> pd.DataFrame:

    def _cap_label(x: str, n: int) -> str:
        x = str(x)
        return x if len(x) <= n else x[: n - 1] + "…"

    def pretty_ind(code: str) -> str:
        lab = ind_labels.get(code, code) if ind_labels else code
        return _cap_label(lab, max_chars)

    i_key = (str(target_origin), str(target_cpa))
    if i_key not in L1.index:
        raise ValueError(f"Target {i_key} not found in L1.index")

    y = y_by_deployment[deployment].reindex(L1.columns).fillna(0.0)
    if exclude_self and i_key in y.index:
        y = y.copy()
        y.loc[i_key] = 0.0

    contrib = (L1.loc[i_key] * y)
    out = (contrib[contrib > 0].nlargest(top_n).reset_index())
    out.columns = ["source_origin", "source_cpa", "output_contribution"]

    out["share_of_target_output"] = out["output_contribution"] / out["output_contribution"].sum()
    out["source_industry"] = out["source_cpa"].astype(str).map(pretty_ind)

    return out[["source_origin", "source_industry", "output_contribution", "share_of_target_output"]]
