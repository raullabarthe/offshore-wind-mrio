"""
Map plotting utilities for country-level supply-chain impacts.

The module contains Cartopy-based choropleth functions, project colormaps,
label formatting helpers, and PDF export routines used by the map section of
the analysis notebook.
"""

from __future__ import annotations

from pathlib import Path
import os
import os.path as op
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from shapely.ops import unary_union
from modules.cost_model.charts import PALETTE_CARDO

try:
    from modules.config import IMG_ROOT as DEFAULT_IMG_ROOT, SAVE_FIGURES, CLOSE_FIGURES
except Exception:
    DEFAULT_IMG_ROOT = None
    SAVE_FIGURES = True
    CLOSE_FIGURES = True

IMG_ROOT = Path(DEFAULT_IMG_ROOT) if DEFAULT_IMG_ROOT is not None else None

# Cardo colormaps (project palette)
def _cardo_cmap(name: str, mid: str, high: str):
    return LinearSegmentedColormap.from_list(
        name, [PALETTE_CARDO["light"], mid, high], N=256
    )

cardo_blue = _cardo_cmap("cardo_blue", PALETTE_CARDO["secondary"], PALETTE_CARDO["dark"])
cardo_red = _cardo_cmap("cardo_red", PALETTE_CARDO["highlight"], PALETTE_CARDO["dark"])


# ==============================================================
# 0) PATHS (default to config.IMG_ROOT; override by calling set_img_root)
# ==============================================================

def set_img_root(path=None):
    """Set the image output directory used by map exports."""
    global IMG_ROOT
    if path is None:
        if DEFAULT_IMG_ROOT is None:
            raise RuntimeError("IMG_ROOT is not set. Pass a path to set_img_root(...).")
        IMG_ROOT = Path(DEFAULT_IMG_ROOT)
    else:
        IMG_ROOT = Path(path)
    IMG_ROOT.mkdir(parents=True, exist_ok=True)
    return IMG_ROOT


def configure_paper_map_figures(
    *,
    img_root=None,
    eu_layout: dict | None = None,
    max_open_warning: int = 50,
) -> tuple[LinearSegmentedColormap, LinearSegmentedColormap]:
    """Apply paper map defaults and return the colormaps used by map figures."""
    set_img_root(img_root)
    set_default_layout(eu=eu_layout or dict(gap=0.1, cbar_width=0.12, right_pad=0.1))
    plt.rcParams["figure.max_open_warning"] = max_open_warning
    blue = LinearSegmentedColormap.from_list(
        "cardo_blue",
        [PALETTE_CARDO["light"], PALETTE_CARDO["secondary"], PALETTE_CARDO["dark"]],
        N=256,
    )
    red = LinearSegmentedColormap.from_list(
        "cardo_red",
        [PALETTE_CARDO["light"], PALETTE_CARDO["highlight"], PALETTE_CARDO["dark"]],
        N=256,
    )
    return blue, red

# ==============================================================
# 1) Map style and LaTeX-safe text helpers
# ==============================================================

def set_house_style():
    mpl.rcParams.update({
        # LaTeX / PGF output
        "backend": "pgf",
        "text.usetex": True,
        "pgf.texsystem": "lualatex",
        "pgf.rcfonts": False,
        "axes.unicode_minus": False,

        # Font sizes (12pt article scale)
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,

        # Lines
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,

        # Serif family (LuaLaTeX sets exact fonts)
        "font.family": "serif",

        # Force EXACT paper fonts via LuaLaTeX
        "pgf.preamble": r"""
\usepackage{fontspec}
\usepackage{unicode-math}

\setmainfont{Latin Modern Roman}
\setsansfont{Latin Modern Sans}
\setmonofont{Latin Modern Mono}
\setmathfont{Latin Modern Math}

\defaultfontfeatures{Ligatures=TeX}
"""
    })


def tex_safe(s: str) -> str:
    """Escape only LaTeX-breaking chars for normal text (PGF/LuaLaTeX safe)."""
    if s is None:
        return s
    s = str(s)
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def tex_bold(s: str) -> str:
    return r"\textbf{" + tex_safe(s) + "}"


def get_text_color(bg_color) -> str:
    """Pick black/white text based on background luminance."""
    if isinstance(bg_color, str):
        rgb = mcolors.to_rgb(bg_color)
    else:
        rgb = bg_color[:3]  # ignore alpha if present
    r, g, b = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.5 else "black"


DEFAULT_LAYOUT_EU = dict(
    map_width=6.0,
    map_height=3.8,
    left_pad=0.12,
    right_pad=0.10,
    top_pad=0.32,
    bottom_pad=0.24,
    cbar_width=0.12,
    gap=0.10,
)

DEFAULT_LAYOUT_WORLD = dict(
    map_width=6.0,
    map_height=2.9,
    left_pad=0.10,
    right_pad=0.10,
    top_pad=0.30,
    bottom_pad=0.18,
    cbar_width=0.12,
    gap=0.03,
)

def set_default_layout(*, eu=None, world=None):
    if eu:
        DEFAULT_LAYOUT_EU.update(eu)
    if world:
        DEFAULT_LAYOUT_WORLD.update(world)


# ==============================================================
# 2) MAP-SAFE SAVER (does NOT affect other plots)
# ==============================================================

def save_map_pdf_no_tight(
    fig=None,
    filename="map.pdf",
    img_root: Optional[Path] = None,
    *,
    save: bool = SAVE_FIGURES,
    close: bool = CLOSE_FIGURES,
):
    fig = fig or plt.gcf()

    if img_root is not None:
        outdir = Path(img_root)
    elif DEFAULT_IMG_ROOT is not None:
        outdir = Path(DEFAULT_IMG_ROOT)
    else:
        outdir = IMG_ROOT
    if outdir is None:
        raise RuntimeError("IMG_ROOT is not set. Call set_img_root(...) or define config.IMG_ROOT.")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / filename

    try:
        fig.set_layout_engine(None)
    except Exception:
        pass

    if save:
        fig.savefig(outpath, format="pdf")
        print(f"Saved to: {outpath}")

    if close:
        plt.close(fig)

    return outpath if save else None


# ==============================================================
# 3) DATA SHAPE + ISO helpers (same logic as your calibrated version)
# ==============================================================

def melt_results_by_origin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    vcols = [c for c in df.columns
             if "_" in c and c.split("_", 1)[0] in {"Output", "GVA", "Income", "Jobs"}]
    long = df.melt(["Deployment", "Country", "Year", "Sector"], vcols, "type_tier", "value")
    long[["type", "tier"]] = long["type_tier"].str.split("_", expand=True)[[0, 1]]
    return long.drop(columns="type_tier")


def aggregate_components_by_label(
    results_by_origin_lvl: dict,
    lvl: int,
    components_map: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    """
    Aggregate multiple component tables into labeled groups.

    components_map: {label: [component_keys...]}
    Returns: {label: aggregated_df}
    """
    if lvl not in results_by_origin_lvl:
        raise KeyError(f"Level {lvl} not found in results_by_origin_lvl.")
    base = results_by_origin_lvl[lvl]

    keys = ["Deployment", "Country", "Year", "Sector"]
    out: dict[str, pd.DataFrame] = {}

    for label, comps in components_map.items():
        missing = [c for c in comps if c not in base]
        if missing:
            raise KeyError(f"Missing components {missing} for label '{label}'.")
        dfs = [base[c] for c in comps]
        num_cols = [c for c in dfs[0].columns if c not in keys]
        agg = (
            pd.concat(dfs, ignore_index=True)
            .groupby(keys, as_index=False)[num_cols]
            .sum()
        )
        out[label] = agg

    return out


def _aggregate_components_from_base(
    base: dict,
    components_map: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    """Aggregate component dict (base level) into labeled groups."""
    keys = ["Deployment", "Country", "Year", "Sector"]
    out: dict[str, pd.DataFrame] = {}

    for label, comps in components_map.items():
        missing = [c for c in comps if c not in base]
        if missing:
            raise KeyError(f"Missing components {missing} for label '{label}'.")
        dfs = [base[c] for c in comps]
        num_cols = [c for c in dfs[0].columns if c not in keys]
        agg = (
            pd.concat(dfs, ignore_index=True)
            .groupby(keys, as_index=False)[num_cols]
            .sum()
        )
        out[label] = agg

    return out


_A3A2 = {
    "AUT":"AT","BEL":"BE","BGR":"BG","HRV":"HR","CYP":"CY","CZE":"CZ","DNK":"DK","EST":"EE","FIN":"FI","FRA":"FR",
    "DEU":"DE","GRC":"GR","HUN":"HU","IRL":"IE","ITA":"IT","LVA":"LV","LTU":"LT","LUX":"LU","MLT":"MT","NLD":"NL",
    "NOR":"NO","POL":"PL","PRT":"PT","ROU":"RO","SRB":"RS","RUS":"RU","SVK":"SK","SVN":"SI","ESP":"ES","SWE":"SE",
    "CHE":"CH","TUR":"TR","GBR":"GB","AUS":"AU","CAN":"CA","CHN":"CN","JPN":"JP","KOR":"KR","USA":"US","ZAF":"ZA",
}

def _fix_iso2(rec):
    a = rec.attributes
    iso2 = a.get("ISO_A2")
    if iso2 and iso2 != "-99":
        return "GB" if iso2 == "UK" else iso2
    a3 = a.get("ADM0_A3") or a.get("ISO_A3")
    return _A3A2.get(a3) if a3 else None


def _in_extent(x, y, e):
    xmin, xmax, ymin, ymax = e
    return xmin <= x <= xmax and ymin <= y <= ymax


# ==============================================================
# 4) COLOR + NORM + LABEL format (same as calibrated version)
# ==============================================================

def fmt_pct(v):
    return "1%" if v < 1 and round(v, 1) >= 1 else (f"{v:.1f}%" if v < 1 else f"{v:.0f}%")

def fmt_int(v):
    return f"{int(round(v)):,}"

def _rest_of_world_share_for_map(
    s_plot: pd.Series,
    *,
    explicit_codes: tuple[str, ...] = ("ROW", "RoW", "WRL_REST"),
) -> float | None:
    """Return the non-geographic Rest-of-the-World share for a percentage map."""
    if s_plot is None or len(s_plot) == 0:
        return None

    s_num = pd.to_numeric(s_plot.copy(), errors="coerce").fillna(0.0)
    s_num.index = s_num.index.astype(str).str.strip()

    for code in explicit_codes:
        if code in s_num.index:
            return float(s_num.loc[code])

    shp = shapereader.natural_earth("50m", "cultural", "admin_0_countries")
    drawable = {
        iso
        for iso in (_fix_iso2(rec) for rec in shapereader.Reader(shp).records())
        if iso is not None
    }
    drawn_total = float(s_num[s_num.index.isin(drawable)].sum())
    return max(0.0, 100.0 - drawn_total)

def make_norm(vmax, scale="linear", gamma=1.0):
    vmax = 1.0 if (vmax is None or vmax <= 0) else float(vmax)

    if scale == "linear":
        return mcolors.Normalize(0.0, vmax)

    if scale == "power":
        return mcolors.PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)

    if scale == "log":
        return mcolors.LogNorm(max(vmax * 1e-6, 1e-9), vmax)

    if scale == "piecewise":
        v_top = vmax

        low_max = min(5.0, v_top)
        mid_max = min(20.0, v_top)

        # Dense bins from 0 to 5 (0.1 steps gives 50-ish levels)
        b1 = np.arange(0.0, low_max + 1e-9, 0.1)

        # Medium bins 5 to 20 (0.5 steps)
        b2 = np.arange(5.0, mid_max + 1e-9, 0.5) if mid_max > 5.0 else np.array([])

        # Coarse bins 20 to vmax (5 steps)
        b3 = np.linspace(20.0, v_top, 15) if v_top > 20.0 else np.array([])

        boundaries = np.unique(np.concatenate([b1, b2, b3]))
        if boundaries[-1] < v_top:
            boundaries = np.append(boundaries, v_top)

        # BoundaryNorm maps values into discrete color indices
        # ncolors will be inferred from the cmap length in ScalarMappable usage
        return mcolors.BoundaryNorm(boundaries, ncolors=256, clip=True)

    raise ValueError("scale in {'linear','power','log','piecewise'}")


def round_up_nice(vmax: float, *, min_step: float = 10.0) -> float:
    """
    Round up to a "nice" maximum:
      - step size grows with magnitude (10, 100, 1000, ...)
      - min_step controls the smallest step (use 0.1 for multipliers)
    """
    if vmax is None or not np.isfinite(vmax) or vmax <= 0:
        return min_step
    vmax = float(vmax)
    exp = int(np.floor(np.log10(vmax)))
    step = 10 ** exp
    if step < min_step:
        step = min_step
    return float(np.ceil(vmax / step) * step)

def make_soft0_cmap(name_or_cmap, zero=None, start=0.35):
    """
    Colormap that starts at `start` of the base cmap.

    - If zero is None: the gradient simply begins at base(start), so small
      plotted values look colored (not like missing_color).
    - If zero is not None: prepend an explicit 'zero' color at index 0.
    """
    cmap = plt.get_cmap(name_or_cmap) if isinstance(name_or_cmap, str) else name_or_cmap

    # Use 256 so the returned cmap has a standard resolution
    base = cmap(np.linspace(start, 1, 256))

    if zero is None:
        return LinearSegmentedColormap.from_list("soft0", base, 256)

    z = np.array(mcolors.to_rgba(zero))
    return LinearSegmentedColormap.from_list("soft0", np.vstack([z, base[1:]]), 256)


# ==============================================================
# 5) CALIBRATED CHOROPLETH (your original layout/logic)
# ==============================================================

def plot_choropleth_with_labels(
    s: pd.Series, *,
    title,
    extent=None,
    projection=None,
    global_view=False,
    cmap="Accent",
    missing_color="lightgrey",
    vmax=None,
    fixed_color_by_iso: Optional[Dict[str, str]] = None,
    eu_fixed_color: Optional[str] = None,
    exclude_from_scale: Optional[set[str]] = None,
    label_exclude: Optional[set[str]] = None,
    skip_iso: Optional[set[str]] = None,
    min_plot_value: float = 0.0,
    label_formatter=fmt_pct,
    cbar_formatter=None,
    cbar_title: str | None = None,
    cbar_title_x: float = 3.00,
    cbar_title_rotation: float = 90.0,
    cbar_title_fontweight: str = "bold",
    label_eps=1e-3,
    absolute_bar_series: Optional[pd.Series] = None,
    absolute_bar_vmax: float | None = None,
    absolute_bar_scale_iso: Optional[set[str]] = None,
    absolute_bar_min_value: float = 0.0,
    absolute_bar_height: float = 4.0,
    absolute_bar_width: float = 0.90,
    absolute_bar_linewidth: float = 8.0,
    absolute_bar_headroom: float = 1.7,
    absolute_bar_label_offset: float = 0.25,
    absolute_bar_color: str = "#222222",
    absolute_bar_edgecolor: str = "white",
    absolute_bar_alpha: float = 0.85,
    absolute_bar_label_formatter=None,
    absolute_bar_label_fontsize: float = 6.0,
    forced_label_xy: Optional[Dict[str, Tuple[float, float]]] = None,
    force_text_color: Optional[Dict[str, str]] = None,
    label_fontsize=7,
    label_fontweight="normal",
    scale="linear",
    gamma=1.0,
    soft0_start=0.35,
    eu27=None,
    group_eu=False,
    dissolve_eu=False,
    only_iso: Optional[set[str]] = None,
    eu_label_xy=(10.0, 50.0),
    map_width=6.4,
    map_height=4.0,
    cbar_width=0.15,
    gap=0.1,
    left_pad=0.18,
    bottom_pad=0.28,
    top_pad=0.40,
    right_pad=0.15,
    left_adjust=0.01,
    cbar_text_x=1.35,
    cbar_text_ha="left",
    title_fontsize=10,
    dpi=180,
):
    s = pd.to_numeric(s.copy(), errors="coerce")
    s.index = s.index.astype(str).str.strip()
    absolute_bar_vmax_auto = absolute_bar_vmax is None
    if absolute_bar_series is not None:
        absolute_bar_series = pd.to_numeric(absolute_bar_series.copy(), errors="coerce")
        absolute_bar_series.index = absolute_bar_series.index.astype(str).str.strip()
        absolute_bar_vmax = float(absolute_bar_vmax or 0.0)
    forced_label_xy = forced_label_xy or {}
    force_text_color = force_text_color or {}
    fixed_color_by_iso = fixed_color_by_iso or {}
    exclude_from_scale = {str(c).strip() for c in (exclude_from_scale or set())}
    absolute_bar_scale_iso = {str(c).strip() for c in (absolute_bar_scale_iso or set())} or None
    label_exclude = {str(c).strip() for c in (label_exclude or set())}
    skip_iso = {str(c).strip() for c in (skip_iso or set())}
    only_iso = {str(c).strip() for c in (only_iso or set())} if only_iso else None

    fig_width = left_pad + map_width + gap + cbar_width + right_pad
    fig_height = bottom_pad + map_height + top_pad
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

    ax = fig.add_axes(
        [(left_pad - left_adjust) / fig_width, bottom_pad / fig_height,
         map_width / fig_width, map_height / fig_height],
        projection=projection or ccrs.PlateCarree(),
    )

    if global_view:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.35, zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.25, zorder=2)

    shp = shapereader.natural_earth("50m", "cultural", "admin_0_countries")

    data = s.dropna()
    if exclude_from_scale:
        data = data[~data.index.isin(exclude_from_scale)]
    vmax = float(data.max()) if vmax is None and len(data) else (1.0 if vmax is None else float(vmax))
    vmax = 1.0 if vmax <= 0 else vmax

    norm = make_norm(vmax, scale, gamma)

    # IMPORTANT: gradient starts at soft0_start (no "missing grey" inside the cmap)
    cmap_obj = make_soft0_cmap(cmap, zero=None, start=soft0_start)

    def in_view(x, y):
        return True if (global_view or extent is None) else _in_extent(x, y, extent)

    if absolute_bar_series is not None and absolute_bar_vmax_auto:
        visible_bar_vals = []
        for rec in shapereader.Reader(shp).records():
            iso2, geom = _fix_iso2(rec), rec.geometry
            if only_iso is not None and iso2 not in only_iso:
                continue
            if iso2 in skip_iso:
                continue
            if absolute_bar_scale_iso is not None and iso2 not in absolute_bar_scale_iso:
                continue
            bar_val = absolute_bar_series.get(iso2, np.nan)
            if pd.isna(bar_val) or abs(float(bar_val)) < float(absolute_bar_min_value):
                continue
            try:
                x, y = forced_label_xy.get(
                    iso2,
                    (geom.representative_point().x, geom.representative_point().y),
                )
                if in_view(x, y):
                    visible_bar_vals.append(abs(float(bar_val)))
            except Exception:
                pass
        if visible_bar_vals:
            absolute_bar_vmax = float(max(visible_bar_vals)) * float(absolute_bar_headroom)
        else:
            absolute_bar_vmax = 0.0

    # thresholds: below color threshold -> missing; labels only for colored values
    thr_color = float(min_plot_value)
    thr_label = max(float(label_eps), thr_color)
    pending_bars = []

    eu_members = set()
    eu_total = 0.0
    eu_union = None

    if group_eu and eu27:
        eu_members = {c for c in eu27 if c in s.index}
        if eu_members:
            eu_vals = s.loc[list(eu_members)].astype(float)
            eu_total = float(eu_vals.where(eu_vals.abs() >= thr_color, 0.0).sum())
        if dissolve_eu and eu_members:
            geoms = [rec.geometry for rec in shapereader.Reader(shp).records()
                     if _fix_iso2(rec) in eu_members]
            if geoms:
                eu_union = unary_union(geoms)

    for rec in shapereader.Reader(shp).records():
        iso2, geom = _fix_iso2(rec), rec.geometry
        if only_iso is not None and iso2 not in only_iso:
            continue
        if iso2 in skip_iso:
            continue
        if group_eu and dissolve_eu and iso2 in eu_members:
            continue

        val0 = s.get(iso2, np.nan)
        if (not pd.isna(val0)) and abs(float(val0)) < thr_color:
            val = np.nan
        else:
            val = val0

        if iso2 in fixed_color_by_iso:
            face = fixed_color_by_iso[iso2]
        else:
            face = (
                missing_color
                if pd.isna(val)
                else (cmap_obj(0.0) if scale == "log" and float(val) <= 0 else cmap_obj(norm(float(val))))
            )

        ax.add_geometries([geom], ccrs.PlateCarree(),
                          facecolor=face, edgecolor="white", linewidth=0.25, zorder=1)

        if group_eu and iso2 in eu_members:
            continue

        if (not pd.isna(val)) and (abs(float(val)) >= thr_label) and (iso2 not in label_exclude):
            try:
                x, y = forced_label_xy.get(
                    iso2,
                    (geom.representative_point().x, geom.representative_point().y),
                )
                if in_view(x, y):
                    lab = label_formatter(float(val))
                    if lab is None:
                        continue
                    lab = str(lab)
                    if not lab.strip():
                        continue
                    ax.text(
                        x, y, tex_safe(lab),
                        transform=ccrs.PlateCarree(),
                        ha="center", va="center",
                        fontsize=label_fontsize,
                        fontweight=label_fontweight,
                        color=force_text_color.get(iso2, get_text_color(face)),
                        zorder=3,
                        clip_on=True,
                        clip_path=ax.patch,
                    )
            except Exception:
                pass

        if absolute_bar_series is not None and absolute_bar_vmax > 0:
            bar_val = absolute_bar_series.get(iso2, np.nan)
            if (not pd.isna(bar_val)) and abs(float(bar_val)) >= float(absolute_bar_min_value):
                try:
                    x, y = forced_label_xy.get(
                        iso2,
                        (geom.representative_point().x, geom.representative_point().y),
                    )
                    if in_view(x, y):
                        h = float(absolute_bar_height) * min(abs(float(bar_val)) / absolute_bar_vmax, 1.0)
                        if h > 0:
                            pending_bars.append((x, y, h, float(bar_val)))
                except Exception:
                    pass

    if group_eu and dissolve_eu and eu_union is not None:
        if abs(float(eu_total)) < thr_color:
            eu_face = mcolors.to_rgba(missing_color)
            draw_eu_label = False
        else:
            if eu_fixed_color is not None:
                eu_face = eu_fixed_color
            else:
                eu_face = cmap_obj(0.0) if scale == "log" and eu_total <= 0 else cmap_obj(norm(eu_total))
            draw_eu_label = True

        ax.add_geometries([eu_union], ccrs.PlateCarree(),
                          facecolor=eu_face, edgecolor="black", linewidth=0.9, zorder=4)

        if draw_eu_label and abs(float(eu_total)) >= thr_label:
            x, y = eu_label_xy
            lab = label_formatter(float(eu_total))
            if lab is None:
                lab = ""
            lab = str(lab)
            if not lab.strip():
                lab = " "
            ax.text(
                x, y, tex_safe(lab),
                transform=ccrs.PlateCarree(),
                ha="center", va="center",
                fontsize=label_fontsize + 1,
                fontweight="bold",
                color=get_text_color(eu_face),
                zorder=5,
                clip_on=True,
                clip_path=ax.patch,
            )

    for x, y, h, bar_val in pending_bars:
        ax.plot(
            [x, x],
            [y, y + h],
            transform=ccrs.PlateCarree(),
            color=absolute_bar_edgecolor,
            linewidth=float(absolute_bar_linewidth) + 1.4,
            alpha=0.95,
            solid_capstyle="butt",
            zorder=8,
            clip_on=True,
        )
        ax.plot(
            [x, x],
            [y, y + h],
            transform=ccrs.PlateCarree(),
            color=absolute_bar_color,
            linewidth=float(absolute_bar_linewidth),
            alpha=float(absolute_bar_alpha),
            solid_capstyle="butt",
            zorder=9,
            clip_on=True,
        )
        if absolute_bar_label_formatter is not None:
            lab = absolute_bar_label_formatter(float(bar_val))
            if lab is not None and str(lab).strip():
                ax.text(
                    x,
                    y + h + float(absolute_bar_label_offset),
                    tex_safe(str(lab)),
                    transform=ccrs.PlateCarree(),
                    ha="center",
                    va="bottom",
                    fontsize=float(absolute_bar_label_fontsize),
                    color="black",
                    zorder=10,
                    clip_on=True,
                    clip_path=ax.patch,
                )

    ax.set_title(tex_bold(title), fontsize=title_fontsize, pad=8)

    # -------------------------
    # COLORBAR (clean ticks)
    # -------------------------
    pos = ax.get_position()
    cbar_gap = gap / fig_width
    cbar_w = cbar_width / fig_width
    cax = fig.add_axes([pos.x1 + cbar_gap, pos.y0, cbar_w, pos.height])

    # Linear ticks from 0 to vmax (default: 10% steps)
    steps = 10
    if vmax <= 0:
        ticks = np.array([0.0], dtype=float)
    else:
        step = vmax / steps
        ticks = np.arange(0.0, vmax + 0.5 * step, step, dtype=float)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    cb = plt.colorbar(sm, cax=cax, ticks=ticks)

    # Remove the "comb" (minor ticks)
    cb.ax.minorticks_off()
    cb.ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    cb.ax.tick_params(which="both", length=0)  # no tick marks at all

    # You already draw your own labels to the right; keep the axis labels empty
    cb.ax.set_yticklabels([])

    trans = cb.ax.get_yaxis_transform()
    fmt_cb = cbar_formatter or label_formatter or fmt_int
    for t in ticks:
        cb.ax.text(
            cbar_text_x, float(t),
            tex_safe(fmt_cb(float(t))),
            transform=trans,
            ha=cbar_text_ha,
            va="center",
            fontsize=8,
            clip_on=False,
        )

    if cbar_title:
        cb.ax.text(
            float(cbar_title_x),
            0.5,
            tex_safe(str(cbar_title)),
            transform=cb.ax.transAxes,
            rotation=float(cbar_title_rotation),
            ha="center",
            va="center",
            fontsize=8,
            fontweight=cbar_title_fontweight,
            clip_on=False,
        )

    return fig, ax


def add_absolute_country_bars(
    ax,
    s_abs: pd.Series,
    *,
    extent=None,
    forced_label_xy: Optional[Dict[str, Tuple[float, float]]] = None,
    skip_iso: Optional[set[str]] = None,
    scale_iso: Optional[set[str]] = None,
    vmax: float | None = None,
    min_value: float = 0.0,
    height: float = 3.5,
    linewidth: float = 7.0,
    headroom: float = 1.8,
    color: str = "#222222",
    edgecolor: str = "white",
    alpha: float = 0.90,
    labels: bool = True,
    label_formatter=fmt_int,
    label_fontsize: float = 6.0,
    label_color_by_iso: Optional[Dict[str, str]] = None,
    label_offset: float = 0.25,
    draw_in_axes_coords: bool = True,
):
    """Draw absolute-value bars independent of choropleth coloring logic."""
    if s_abs is None:
        return []

    s_abs = pd.to_numeric(s_abs.copy(), errors="coerce").dropna()
    s_abs.index = s_abs.index.astype(str).str.strip()
    if s_abs.empty:
        return []

    forced_label_xy = forced_label_xy or {}
    label_color_by_iso = label_color_by_iso or {}
    skip_iso = {str(c).strip() for c in (skip_iso or set())}
    scale_iso = {str(c).strip() for c in (scale_iso or set())} if scale_iso else None

    shp = shapereader.natural_earth("50m", "cultural", "admin_0_countries")

    points = {}
    for rec in shapereader.Reader(shp).records():
        iso2 = _fix_iso2(rec)
        if iso2 in skip_iso:
            continue
        if iso2 not in s_abs.index:
            continue
        try:
            x, y = forced_label_xy.get(
                iso2,
                (rec.geometry.representative_point().x, rec.geometry.representative_point().y),
            )
        except Exception:
            continue
        if extent is not None and not _in_extent(x, y, extent):
            continue
        points[iso2] = (float(x), float(y))

    if not points:
        return []

    if vmax is None:
        scale_candidates = [
            abs(float(s_abs.get(iso, 0.0)))
            for iso in points
            if scale_iso is None or iso in scale_iso
        ]
        scale_candidates = [v for v in scale_candidates if v >= float(min_value)]
        if not scale_candidates:
            scale_candidates = [abs(float(s_abs.get(iso, 0.0))) for iso in points]
        vmax = (max(scale_candidates) * float(headroom)) if scale_candidates else 0.0
    vmax = float(vmax or 0.0)
    if vmax <= 0.0:
        return []

    drawn = []
    for iso, (x, y) in points.items():
        val = float(s_abs.get(iso, 0.0))
        if abs(val) < float(min_value):
            continue
        h = float(height) * min(abs(val) / vmax, 1.0)
        if h <= 0:
            continue

        if draw_in_axes_coords:
            try:
                x_proj, y_proj = ax.projection.transform_point(x, y, ccrs.PlateCarree())
                x_disp, y_disp = ax.transData.transform((x_proj, y_proj))
                x_ax, y_ax = ax.transAxes.inverted().transform((x_disp, y_disp))
            except Exception:
                continue
            if not (np.isfinite(x_ax) and np.isfinite(y_ax)):
                continue
            if x_ax < -0.05 or x_ax > 1.05 or y_ax < -0.05 or y_ax > 1.05:
                continue

            if extent is not None:
                y_span = float(extent[3] - extent[2]) or 1.0
            else:
                y0, y1 = ax.get_ylim()
                y_span = abs(float(y1 - y0)) or 1.0
            h_ax = h / y_span
            label_offset_ax = float(label_offset) / y_span
            line_transform = ax.transAxes
            x_line = [x_ax, x_ax]
            y_line = [y_ax, y_ax + h_ax]
            label_x = x_ax
            label_y = y_ax + h_ax + label_offset_ax
            clip = False
        else:
            line_transform = ccrs.PlateCarree()
            x_line = [x, x]
            y_line = [y, y + h]
            label_x = x
            label_y = y + h + float(label_offset)
            clip = True

        ax.plot(
            x_line,
            y_line,
            transform=line_transform,
            color=edgecolor,
            linewidth=float(linewidth) + 1.4,
            alpha=0.98,
            solid_capstyle="butt",
            zorder=99,
            clip_on=clip,
        )
        ax.plot(
            x_line,
            y_line,
            transform=line_transform,
            color=color,
            linewidth=float(linewidth),
            alpha=float(alpha),
            solid_capstyle="butt",
            zorder=100,
            clip_on=clip,
        )

        if labels and label_formatter is not None:
            lab = label_formatter(val)
            if lab is not None and str(lab).strip():
                ax.text(
                    label_x,
                    label_y,
                    tex_safe(str(lab)),
                    transform=line_transform,
                    ha="center",
                    va="bottom",
                    fontsize=float(label_fontsize),
                    color=label_color_by_iso.get(iso, "black"),
                    zorder=101,
                    clip_on=False,
                )
        drawn.append(iso)

    return drawn




# ==============================================================
# 6) WRAPPER (same calculation logic; saves PDFs)
# ==============================================================

def plot_deployment_supplychain_maps(
    results_by_origin,
    deployment: str,
    *,
    tier: str,
    impact_type: str,
    eu27: set[str],
    filename_prefix: str,
    cmap=None,
    deployments_avg: list[str] | tuple[str, ...] | None = None,
    avg_label: str = "Average",
    title_prefix: str | None = None,
    title_override: str | None = None,
    title_include_dep: bool = False,
    title_units_style: str = "paren",  # "paren" or "dash"
    components_map: dict[str, list[str]] | None = None,
    component_label: str | None = None,
    lvl: int | None = 2,
    as_multiplier: bool = False,
    as_share: bool = False,
    share_scope: str = "global",  # "global" or "eu27"
    eu_only: bool = False,
    annualize: bool = False,
    lifetime_years: float = 25.0,
    connected_country: dict | None = None,
    weight_key: str = "Power (MW)",
    world_cmap=None,
    eu_cmap=None,
    world_eu_fixed_color: Optional[str] = None,
    exclude_eu_from_scale: bool = True,
    world_cbar_text_x: float | None = None,
    world_cbar_text_ha: str = "left",
    world_cbar_title_x: float | None = None,
    eu_cbar_text_x: float | None = None,
    eu_cbar_text_ha: str = "left",
    eu_cbar_title_x: float | None = None,
    auto_vmax_E: bool = False,
    auto_vmax_W: bool = False,
    auto_vmax_E_scope: str = "eu27",  # "eu27" or "all"
    auto_vmax_step_E: float | None = None,
    auto_vmax_step_W: float | None = None,
    world_label_exclude: Optional[set[str]] = None,
    eu_label_exclude: Optional[set[str]] = None,
    world_skip_iso: Optional[set[str]] = None,
    eu_skip_iso: Optional[set[str]] = None,
    world_forced_label_xy: Optional[dict] = None,
    eu_forced_label_xy: Optional[dict] = None,
    world_eu_label_xy: Optional[tuple] = None,
    eu_eu_label_xy: Optional[tuple] = None,
    return_series: bool = False,
    exclude_countries: set[str] | list[str] | None = None,
    excluded_countries: set[str] | list[str] | None = None,
    scale="linear",
    gamma=1.0,
    soft0_start=0.0,
    vmax_W=40.0,
    vmax_E=20.0,
    min_plot_pct: float = 0.0,
    min_plot_jobs: float = 0.0,
    label_eps: float | None = None,
    show_absolute_bars: bool = False,
    absolute_bar_vmax: float | None = None,
    absolute_bar_scale: str = "eu27",  # "eu27" or "visible"
    absolute_bar_min_value: float = 0.0,
    absolute_bar_height: float = 3.5,
    absolute_bar_width: float = 1.20,
    absolute_bar_linewidth: float = 8.0,
    absolute_bar_headroom: float = 1.7,
    absolute_bar_label_offset: float = 0.18,
    absolute_bar_color: str = "#8A0000",
    absolute_bar_edgecolor: str = "white",
    absolute_bar_alpha: float = 0.90,
    absolute_bar_labels: bool = True,
    absolute_bar_label_color_overrides: Optional[Dict[str, str]] = None,
    plot_world: bool = False,
    plot_eu: bool = True,
    world_group_eu: bool = True,
    world_row_annotation: bool = True,
    world_row_annotation_xy: tuple[float, float] = (-115.0, -35.0),
    world_row_annotation_label: str = "Rest-of-the-World",
    world_row_annotation_fontsize: float = 7.0,
    layout_eu: Optional[dict] = None,
    layout_world: Optional[dict] = None,
    title_fontsize: float | int = 12,
    save: bool = SAVE_FIGURES,
    close: bool = CLOSE_FIGURES,
    left_adjust=0.01,
):
    set_house_style()

    if as_multiplier and as_share:
        raise ValueError("as_multiplier and as_share cannot both be True.")

    cmap = cardo_blue if cmap is None else cmap

    if label_eps is None:
        label_eps = 0.10 if bool(plot_world) and not bool(plot_eu) else 0.05

    share_scope_key = str(share_scope or "global").strip().lower()
    if share_scope_key in {"eu", "eu27", "europe", "european"}:
        share_scope_key = "eu27"
    elif share_scope_key in {"global", "world", "all"}:
        share_scope_key = "global"
    else:
        raise ValueError("share_scope must be 'global' or 'eu27'.")

    exclude_set: set[str] = {"MT"}
    if exclude_countries:
        exclude_set |= {str(c).strip() for c in exclude_countries}
    if excluded_countries:
        exclude_set |= {str(c).strip() for c in excluded_countries}

    eu27_eff = {str(c).strip() for c in (eu27 or set())} - exclude_set

    # optional: aggregate components on the fly
    if components_map is not None:
        if isinstance(results_by_origin, pd.DataFrame):
            raise ValueError("components_map requires results_by_origin to be a dict (by component).")
        if lvl is not None and lvl in results_by_origin:
            base = results_by_origin[lvl]
        else:
            base = results_by_origin
        agg_map = _aggregate_components_from_base(base, components_map)
        if component_label is None:
            if len(components_map) == 1:
                component_label = next(iter(components_map.keys()))
            else:
                raise ValueError("component_label is required when components_map has multiple entries.")
        if component_label not in agg_map:
            raise KeyError(f"component_label '{component_label}' not found in components_map.")
        results_by_origin = agg_map[component_label]

    long_df = melt_results_by_origin(results_by_origin)
    absolute_bar_plot = None

    def _average_country_values(vals: pd.DataFrame, dep_set: set[str]) -> pd.Series:
        by_dep = vals.groupby(["Deployment", "Country"], as_index=False)["value"].sum()
        if connected_country is not None:
            w_map = {str(k): float(v.get(weight_key, 0.0)) for k, v in connected_country.items()}
            w_map = {d: w for d, w in w_map.items() if d in dep_set and w > 0}
            if not w_map:
                w_map = {d: 1.0 for d in dep_set}
            by_dep["_w"] = by_dep["Deployment"].astype(str).map(w_map).fillna(0.0)
            by_dep["_wv"] = by_dep["value"] * by_dep["_w"]
            num = by_dep.groupby("Country")["_wv"].sum()
            den = by_dep.groupby("Country")["_w"].sum().replace(0.0, np.nan)
            return (num / den).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return by_dep.groupby("Country")["value"].mean()

    if deployments_avg:
        dep_set = {str(d).strip() for d in deployments_avg}
        sel = long_df[
            (long_df.Deployment.astype(str).isin(dep_set))
            & (long_df.tier == tier)
            & (long_df.type == impact_type)
        ]
        if sel.empty:
            raise ValueError(
                f"No rows: Deployments={sorted(dep_set)}, tier={tier}, type={impact_type}"
            )
        absolute_base = _average_country_values(sel, dep_set)
        if as_share:
            vals = sel.groupby(["Deployment", "Country"], as_index=False)["value"].sum()
            if share_scope_key == "eu27" and eu27_eff:
                # Build full deployment x country table to keep missing countries as zeros.
                pivot = vals.pivot(index="Deployment", columns="Country", values="value").fillna(0.0)
                for c in eu27_eff:
                    if c not in pivot.columns:
                        pivot[c] = 0.0
                eu_cols = [c for c in pivot.columns if c in eu27_eff]
                denom = pivot[eu_cols].sum(axis=1).replace(0.0, np.nan) if eu_cols else pivot.sum(axis=1)
                shares = pivot.div(denom, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

                if connected_country is not None:
                    w_map = {str(k): float(v.get(weight_key, 0.0)) for k, v in connected_country.items()}
                    w_map = {d: w for d, w in w_map.items() if d in dep_set and w > 0}
                    if not w_map:
                        w_map = {d: 1.0 for d in dep_set}
                    weights = pd.Series(w_map, dtype=float).reindex(shares.index).fillna(0.0)
                    if float(weights.sum()) <= 0.0:
                        weights = pd.Series(1.0, index=shares.index, dtype=float)
                    s = (shares.mul(weights, axis=0).sum(axis=0) / float(weights.sum())).fillna(0.0)
                else:
                    s = shares.mean(axis=0).fillna(0.0)
            else:
                totals = vals.groupby("Deployment", as_index=False)["value"].sum().rename(columns={"value": "_tot"})
                vals = vals.merge(totals, on="Deployment", how="left")
                denom = vals["_tot"].replace(0.0, np.nan)
                vals["_share"] = (vals["value"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

                if connected_country is not None:
                    w_map = {str(k): float(v.get(weight_key, 0.0)) for k, v in connected_country.items()}
                    w_map = {d: w for d, w in w_map.items() if d in dep_set and w > 0}
                    if not w_map:
                        w_map = {d: 1.0 for d in dep_set}
                    vals["_w"] = vals["Deployment"].map(w_map).fillna(0.0)
                    vals["_ws"] = vals["_share"] * vals["_w"]
                    num = vals.groupby("Country")["_ws"].sum()
                    den = vals.groupby("Country")["_w"].sum().replace(0.0, np.nan)
                    s = (num / den).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                else:
                    s = vals.groupby("Country")["_share"].mean()
        elif as_multiplier:
            sel_shock = long_df[
                (long_df.Deployment.astype(str).isin(dep_set))
                & (long_df.tier == "Direct")
                & (long_df.type == "Output")
            ]
            if sel_shock.empty:
                raise ValueError("No rows for shock (Output_Direct).")
            vals = sel.groupby(["Deployment", "Country"], as_index=False)["value"].sum()
            shocks = sel_shock.groupby("Deployment", as_index=False)["value"].sum().rename(columns={"value": "_shock"})
            vals = vals.merge(shocks, on="Deployment", how="left")
            denom = vals["_shock"].replace(0.0, np.nan)
            if str(impact_type).upper() == "JOBS":
                denom = denom / 1e6
            vals["_mult"] = vals["value"] / denom
            vals["_mult"] = vals["_mult"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if connected_country is not None:
                w_map = {str(k): float(v.get(weight_key, 0.0)) for k, v in connected_country.items()}
                # keep only deployments in dep_set
                w_map = {d: w for d, w in w_map.items() if d in dep_set and w > 0}
                if not w_map:
                    # fallback to equal weights
                    w_map = {d: 1.0 for d in dep_set}
                vals["_w"] = vals["Deployment"].map(w_map).fillna(0.0)
                vals["_wm"] = vals["_mult"] * vals["_w"]
                num = vals.groupby("Country")["_wm"].sum()
                den = vals.groupby("Country")["_w"].sum().replace(0.0, np.nan)
                s = (num / den).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            else:
                s = vals.groupby("Country")["_mult"].mean()
        else:
            # sum per deployment, then average across deployments
            s = absolute_base
        dep_label = avg_label
    else:
        sel = long_df[
            (long_df.Deployment == deployment)
            & (long_df.tier == tier)
            & (long_df.type == impact_type)
        ]
        if sel.empty:
            raise ValueError(f"No rows: Deployment={deployment}, tier={tier}, type={impact_type}")
        absolute_base = sel.groupby("Country")["value"].sum()
        if as_share:
            vals = sel.groupby("Country")["value"].sum()
            if share_scope_key == "eu27" and eu27_eff:
                total = float(vals[vals.index.isin(eu27_eff)].sum()) or 0.0
            else:
                total = float(vals.sum()) or 0.0
            if total <= 0.0:
                s = vals * 0.0
            else:
                s = (vals / total).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        elif as_multiplier:
            sel_shock = long_df[
                (long_df.Deployment == deployment)
                & (long_df.tier == "Direct")
                & (long_df.type == "Output")
            ]
            if sel_shock.empty:
                raise ValueError("No rows for shock (Output_Direct).")
            shock_total = float(sel_shock["value"].sum())
            denom = (shock_total / 1e6) if str(impact_type).upper() == "JOBS" else shock_total
            denom = denom if denom != 0.0 else np.nan
            s = sel.groupby("Country")["value"].sum() / denom
            s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            s = sel.groupby("Country")["value"].sum()
        dep_label = deployment

    s_plot = s.copy()
    value_scale = 1.0 if bool(annualize) else float(lifetime_years)
    if show_absolute_bars:
        absolute_bar_plot = absolute_base.copy() * value_scale
        if str(impact_type).upper() != "JOBS":
            absolute_bar_plot = absolute_bar_plot / 1_000_000.0
        if exclude_set:
            absolute_bar_plot = absolute_bar_plot.drop(
                index=[c for c in absolute_bar_plot.index if str(c).strip() in exclude_set],
                errors="ignore",
            )
    if exclude_set:
        drop = exclude_set
        s_plot = s_plot.drop(index=[c for c in s_plot.index if str(c).strip() in drop], errors="ignore")
    if eu_only and eu27_eff and not show_absolute_bars:
        s_plot = s_plot[s_plot.index.isin(eu27_eff)]
    if as_share:
        # share: keep as percentage (0-100) for labels
        s_plot = s_plot * 100.0
        if show_absolute_bars:
            # Color only EU27 by share; non-EU countries remain neutral but can show bars.
            s_plot = s_plot.where(s_plot.index.isin(eu27_eff), np.nan)
            label_formatter = lambda _v: ""
        else:
            label_formatter = fmt_pct
        min_plot_value = float(min_plot_pct)
        title_suffix = ""
    elif as_multiplier:
        # multipliers: no monetary scaling; adapt decimals to avoid "0.0"
        def _fmt_mult(v):
            v = float(v)
            for d in (1, 2, 3):
                s = f"{v:.{d}f}"
                try:
                    if float(s) != 0.0:
                        return s
                except Exception:
                    return s
            return ""

        label_formatter = _fmt_mult
        min_plot_value = float(min_plot_jobs if str(impact_type).upper() == "JOBS" else min_plot_pct)
        title_suffix = ""
    else:
        if impact_type == "Jobs":
            s_plot = s_plot * value_scale
            label_formatter = fmt_int
            min_plot_value = float(min_plot_jobs)
            title_suffix = " (jobs per year)" if bool(annualize) else f" (job-years, cumulative {lifetime_years:g}-year)"
        else:
            # Convert monetary impacts to millions
            s_plot = s_plot * value_scale / 1_000_000.0
            label_formatter = fmt_int
            min_plot_value = float(min_plot_pct)
            title_suffix = " (million EUR per year)" if bool(annualize) else f" (million EUR, cumulative {lifetime_years:g}-year)"

    # sanitize for filesystem (avoid path separators, spaces, etc.)
    def _safe_fname(s: str) -> str:
        return (
            str(s)
            .replace("/", "_")
            .replace("\\", "_")
            .replace("&", "and")
            .replace(" ", "_")
        )

    safe_label = _safe_fname(dep_label)
    safe_prefix = _safe_fname(filename_prefix)
    base = f"{safe_prefix}_{impact_type}_{safe_label}"

    layW = {**DEFAULT_LAYOUT_WORLD, **(layout_world or {})}
    layE = {**DEFAULT_LAYOUT_EU, **(layout_eu or {})}
    if show_absolute_bars and as_share and layout_eu is None:
        layE["right_pad"] = max(float(layE.get("right_pad", 0.0)), 0.36)
    if show_absolute_bars and as_share and layout_world is None:
        layW["right_pad"] = max(float(layW.get("right_pad", 0.0)), 0.36)

    # ---- Forced label positions: EU vs WORLD (TR differs) ----
    forced_xy_eu = {
        "GB": (-2.5, 54.8),
        "FR": (2.0, 46.5),
        "NO": (10.0, 61.5),
        "TR": (28.5, 39.0),   # good for EU extent
    }
    forced_xy_world = {
        **forced_xy_eu,
        "TR": (35.5, 39.0),   # push right for Robinson/global view
        # World defaults as per your preferred layout
        "GB": (-5.5, 61.5),
        "NO": (8.5, 68.0),
        "KR": (125.0, 31.0),
        "JP": (140.5, 33.0),
    }
    if eu_forced_label_xy:
        forced_xy_eu.update(eu_forced_label_xy)
    if world_forced_label_xy:
        forced_xy_world.update(world_forced_label_xy)
    if eu_label_exclude is None:
        eu_label_exclude = {"RS"}
    if world_label_exclude is None:
        world_label_exclude = {"RS"}
    if world_skip_iso is None:
        world_skip_iso = set()
    if eu_skip_iso is None:
        eu_skip_iso = set()
    if world_eu_label_xy is None:
        world_eu_label_xy = (10.0, 48.0)
    if eu_eu_label_xy is None:
        eu_eu_label_xy = (10.0, 48.0)

    if as_share:
        scope_label = "EU27" if share_scope_key == "eu27" else "global"
        if show_absolute_bars:
            title_type = "Employment" if str(impact_type).upper() == "JOBS" else impact_type
            if str(impact_type).upper() == "JOBS":
                unit_text = "jobs per year" if bool(annualize) else f"job-years, cumulative {lifetime_years:g}-year"
            else:
                unit_text = "million EUR per year" if bool(annualize) else f"million EUR, cumulative {lifetime_years:g}-year"
            title_tail = f"{title_type} ({unit_text})"
        else:
            share_type = "Employment" if str(impact_type).upper() == "JOBS" else impact_type
            title_tail = f"{share_type} capture (% of {scope_label} total)"
    elif as_multiplier:
        if str(impact_type).upper() == "JOBS":
            title_tail = "Employment multiplier [job-years/mEUR]"
        else:
            title_tail = "GVA multiplier [EUR/EUR]" if str(impact_type).upper() == "GVA" else f"{impact_type} multiplier"
    else:
        if impact_type == "Income":
            title_type = "Labor income"
        elif impact_type == "Jobs":
            title_type = "Employment"
        else:
            title_type = impact_type

        if impact_type == "Jobs":
            unit_text = "jobs per year" if bool(annualize) else f"job-years, cumulative {lifetime_years:g}-year"
        else:
            unit_text = "million EUR per year" if bool(annualize) else f"million EUR, cumulative {lifetime_years:g}-year"

        if title_units_style == "dash":
            title_tail = f"{title_type} - {unit_text}"
        else:
            title_tail = f"{title_type} ({unit_text})"

    if title_override is not None:
        full_title = str(title_override)
    else:
        if title_prefix is not None:
            base_title = str(title_prefix)
            if title_include_dep:
                base_title = f"{base_title} - {dep_label}"
        else:
            base_title = f"Deployment scenario: {dep_label}"
        full_title = f"{base_title} - {title_tail}"

    # ---- auto vmax (after any unit conversion) ----
    if auto_vmax_E:
        if auto_vmax_E_scope == "all":
            data_e = s_plot
        else:
            data_e = s_plot[s_plot.index.isin(eu27)] if eu27 else s_plot
        step_e = auto_vmax_step_E if auto_vmax_step_E is not None else (0.1 if as_multiplier else 10.0)
        vmax_E = round_up_nice(float(data_e.max()) if len(data_e) else 0.0, min_step=step_e)

    if auto_vmax_W:
        data_w = s_plot
        if exclude_eu_from_scale and eu27:
            data_w = data_w[~data_w.index.isin(eu27)]
        step_w = auto_vmax_step_W if auto_vmax_step_W is not None else (0.1 if as_multiplier else 10.0)
        vmax_W = round_up_nice(float(data_w.max()) if len(data_w) else 0.0, min_step=step_w)

    if return_series == "absolute_bars":
        return absolute_bar_plot
    if return_series:
        return s_plot

    absolute_bar_label_formatter = None
    if absolute_bar_labels:
        absolute_bar_label_formatter = fmt_int
    absolute_bar_scale_iso = None
    if show_absolute_bars and str(absolute_bar_scale).strip().lower() in {"eu", "eu27", "europe"}:
        absolute_bar_scale_iso = eu27_eff
    cbar_title = None
    if show_absolute_bars and as_share:
        scope_label = "EU27" if share_scope_key == "eu27" else "global"
        cbar_title = f"% of {scope_label}"

    def _bar_label_colors(vmax_for_map):
        if not (show_absolute_bars and as_share):
            return {}
        try:
            norm_for_map = make_norm(float(vmax_for_map), scale, gamma)
            cmap_for_map = make_soft0_cmap(cmap_eu if "cmap_eu" in locals() else cmap, zero=None, start=soft0_start)
        except Exception:
            return {}
        out = {}
        for iso, val in s_plot.dropna().items():
            try:
                face = cmap_for_map(norm_for_map(float(val)))
                out[str(iso)] = get_text_color(face)
            except Exception:
                pass
        overrides = {"DK": "black", "NL": "black", "LU": "white"}
        if absolute_bar_label_color_overrides:
            overrides.update({str(k).strip(): str(v) for k, v in absolute_bar_label_color_overrides.items()})
        out.update(overrides)
        return out

    if plot_world:
        group_eu = bool(world_group_eu)
        dissolve_eu = bool(world_group_eu)
        cmap_world = world_cmap or cardo_red
        world_eu_color = world_eu_fixed_color
        if world_eu_color is None and exclude_eu_from_scale:
            world_eu_color = cmap(0.85)
        if world_skip_iso is None:
            world_skip_iso = {"AQ"}
        if world_label_exclude is None:
            world_label_exclude = {"RS"}
        cbar_text_x = 1.10 if world_cbar_text_x is None else float(world_cbar_text_x)
        only_iso = eu27_eff if (eu_only and eu27_eff and not show_absolute_bars) else None
        fig, ax = plot_choropleth_with_labels(
            s_plot,
            title=full_title,
            projection=ccrs.Robinson(),
            global_view=True,
            cmap=cmap_world,
            missing_color="whitesmoke" if eu_only else "lightgrey",
            min_plot_value=min_plot_value,
            label_eps=label_eps,
            label_fontsize=6,
            force_text_color={"GB": "black"},
            forced_label_xy=forced_xy_world,
            scale=scale,
            gamma=gamma,
            soft0_start=soft0_start,
            eu27=eu27,
            group_eu=group_eu,
            dissolve_eu=dissolve_eu,
            only_iso=only_iso,
            eu_fixed_color=world_eu_color,
            exclude_from_scale=eu27 if exclude_eu_from_scale else None,
            vmax=vmax_W,
            left_adjust=left_adjust,
            title_fontsize=title_fontsize,
            label_formatter=label_formatter,
            cbar_formatter=fmt_pct if (show_absolute_bars and as_share) else None,
            cbar_title=cbar_title,
            cbar_title_x=3.00 if world_cbar_title_x is None else float(world_cbar_title_x),
            cbar_text_x=cbar_text_x,
            cbar_text_ha=world_cbar_text_ha,
            label_exclude=world_label_exclude,
            skip_iso=world_skip_iso,
            eu_label_xy=world_eu_label_xy,
            **layW,
        )
        if show_absolute_bars and absolute_bar_plot is not None:
            add_absolute_country_bars(
                ax,
                absolute_bar_plot,
                extent=None,
                forced_label_xy=forced_xy_world,
                skip_iso=world_skip_iso,
                scale_iso=absolute_bar_scale_iso,
                vmax=absolute_bar_vmax,
                min_value=absolute_bar_min_value,
                height=absolute_bar_height,
                linewidth=absolute_bar_linewidth,
                headroom=absolute_bar_headroom,
                color=absolute_bar_color,
                edgecolor=absolute_bar_edgecolor,
                alpha=absolute_bar_alpha,
                labels=absolute_bar_labels,
                label_formatter=absolute_bar_label_formatter,
                label_color_by_iso=_bar_label_colors(vmax_W),
                label_offset=absolute_bar_label_offset,
            )
        if world_row_annotation and as_share and share_scope_key == "global":
            row_share = _rest_of_world_share_for_map(s_plot)
            if row_share is not None and abs(float(row_share)) >= float(label_eps):
                ax.text(
                    world_row_annotation_xy[0],
                    world_row_annotation_xy[1],
                    tex_safe(f"{world_row_annotation_label}: {fmt_pct(float(row_share))}"),
                    transform=ccrs.PlateCarree(),
                    ha="center",
                    va="center",
                    fontsize=float(world_row_annotation_fontsize),
                    color="black",
                    zorder=8,
                    clip_on=True,
                    clip_path=ax.patch,
                )
        save_map_pdf_no_tight(fig, f"{base}_WORLD.pdf", save=save, close=close)

    if not plot_eu:
        return None

    cmap_eu = eu_cmap or cmap
    cbar_text_x = 1.35 if eu_cbar_text_x is None else float(eu_cbar_text_x)
    only_iso = eu27_eff if (eu_only and eu27_eff and not show_absolute_bars) else None
    eu_extent = (-12, 35, 35, 66)
    fig, ax = plot_choropleth_with_labels(
        s_plot,
        title=full_title,
        projection=ccrs.PlateCarree(),
        extent=eu_extent,
        cmap=cmap_eu,
        missing_color="whitesmoke" if eu_only else "lightgrey",
        min_plot_value=min_plot_value,
        label_eps=label_eps,
        forced_label_xy=forced_xy_eu,
        scale=scale,
        gamma=gamma,
        soft0_start=soft0_start,
        eu27=eu27,
        group_eu=False,
        dissolve_eu=False,
        only_iso=only_iso,
        vmax=vmax_E,
        left_adjust=left_adjust,
        title_fontsize=title_fontsize,
        label_formatter=label_formatter,
        cbar_formatter=fmt_pct if (show_absolute_bars and as_share) else None,
        cbar_title=cbar_title,
        cbar_title_x=4.35 if eu_cbar_title_x is None else float(eu_cbar_title_x),
        cbar_text_x=cbar_text_x,
        cbar_text_ha=eu_cbar_text_ha,
        label_exclude=eu_label_exclude,
        skip_iso=eu_skip_iso,
        eu_label_xy=eu_eu_label_xy,
        **layE,
    )
    if show_absolute_bars and absolute_bar_plot is not None:
        add_absolute_country_bars(
            ax,
            absolute_bar_plot,
            extent=eu_extent,
            forced_label_xy=forced_xy_eu,
            skip_iso=eu_skip_iso,
            scale_iso=absolute_bar_scale_iso,
            vmax=absolute_bar_vmax,
            min_value=absolute_bar_min_value,
            height=absolute_bar_height,
            linewidth=absolute_bar_linewidth,
            headroom=absolute_bar_headroom,
            color=absolute_bar_color,
            edgecolor=absolute_bar_edgecolor,
            alpha=absolute_bar_alpha,
            labels=absolute_bar_labels,
            label_formatter=absolute_bar_label_formatter,
            label_color_by_iso=_bar_label_colors(vmax_E),
            label_offset=absolute_bar_label_offset,
        )
    save_map_pdf_no_tight(fig, f"{base}_EU.pdf", save=save, close=close)

    return None


def plot_paper_eu27_component_share_maps(
    *,
    results_by_origin,
    components_map: dict[str, list[str]],
    core_deployment,
    eu27: set[str],
    connected_country: dict,
    indicators: tuple[str, ...] = ("GVA", "Jobs"),
    lifetime_years: float = 25.0,
    save: bool = SAVE_FIGURES,
    close: bool = CLOSE_FIGURES,
):
    """Save the paper D.1 EU27 component value-capture maps."""
    outputs = []
    for ind in indicators:
        for comp_label in components_map:
            outputs.append(
                plot_deployment_supplychain_maps(
                    results_by_origin=results_by_origin,
                    components_map=components_map,
                    component_label=comp_label,
                    deployment=core_deployment[0],
                    deployments_avg=core_deployment,
                    avg_label="Average",
                    tier="Total",
                    impact_type=ind,
                    eu27=eu27,
                    filename_prefix=f"D.1.map_effects_{comp_label}",
                    title_prefix=comp_label,
                    as_share=True,
                    share_scope="eu27",
                    eu_only=True,
                    lifetime_years=lifetime_years,
                    show_absolute_bars=True,
                    connected_country=connected_country,
                    save=save,
                    close=close,
                )
            )
    return outputs


def plot_paper_world_component_share_maps(
    *,
    results_by_origin,
    components_map: dict[str, list[str]],
    core_deployment,
    eu27: set[str],
    connected_country: dict,
    indicators: tuple[str, ...] = ("GVA", "Jobs"),
    save: bool = SAVE_FIGURES,
    close: bool = CLOSE_FIGURES,
):
    """Save the paper D.2 world component value-capture maps."""
    outputs = []
    for ind in indicators:
        for comp_label in components_map:
            outputs.append(
                plot_deployment_supplychain_maps(
                    results_by_origin=results_by_origin,
                    components_map=components_map,
                    component_label=comp_label,
                    deployment=core_deployment[0],
                    deployments_avg=core_deployment,
                    avg_label="Average",
                    tier="Total",
                    impact_type=ind,
                    eu27=eu27,
                    filename_prefix=f"D.2.map_effects_{comp_label}",
                    title_prefix=comp_label,
                    as_share=True,
                    connected_country=connected_country,
                    plot_world=True,
                    plot_eu=False,
                    save=save,
                    close=close,
                )
            )
    return outputs




