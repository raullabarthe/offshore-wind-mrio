"""Supply-chain figures and plot-specific aggregation helpers."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap
from matplotlib.ticker import PercentFormatter

from modules.config import SAVE_FIGURES, CLOSE_FIGURES, project_relative_path
from modules.country import canonical_name, EU27, CODE_TO_NAME, COLOR_MAP as COUNTRY_COLOR_MAP
from modules.cost_model.charts import PALETTE_CARDO
from modules.latex_style import (
    apply_house_style,
    save_and_close,
    set_house_style,
    tex_safe,
    tex_bold,
    adjust_lightness,
    get_text_color,
)
from modules.impacts.impacts import _short_subcomponent_label

FALLBACK_GREY = "#BDBDBD"
EU_LABEL = "EU27"
UK_LABEL = "United Kingdom"
OTHER_EU27_LABEL = "Other EU27"
OTHER_ORIGINS_LABEL = "Other origins"
ROW_LABEL = "RoW residual"


def _save_and_close_exact_aware(
    *,
    fig,
    filename,
    save,
    close,
    exact_size=False,
):
    if not exact_size:
        return save_and_close(fig=fig, filename=filename, save=save, close=close)

    from pathlib import Path
    from modules.config import IMG_ROOT, SAVE_FIGURES

    fig = fig or plt.gcf()
    save = SAVE_FIGURES if save is None else save
    close = CLOSE_FIGURES if close is None else close
    outpath = None

    if save:
        outpath = Path(IMG_ROOT) / filename
        try:
            fig.set_layout_engine("none")
        except Exception:
            pass
        fig.savefig(outpath)
        print(f"Saved to: {project_relative_path(outpath)}")

    if close:
        plt.close(fig)

    return outpath


# Cost category order for tables and charts
CAT_ORDER = [
    "Devex",
    "Turbine",
    "BoP",
    "Installation",
    "O&M",
    "ABEX",
    "Other",
    "To review",
]


def freq_costtype_to_latex(
    df: pd.DataFrame,
    *,
    caption: str = "Supplier contracts by component, subcomponent, and deployment country.",
    label: str = "tab:supplier_contracts_country",
    tabcolsep: int = 2,
    arraystretch: float = 1.0,
    font_size: str = r"\scriptsize",
) -> str:
    """Export the pre-aggregated country contract table in the paper format."""
    required = {"component", "subcomponent", "GB", "DE", "NL", "DK", "BE", "Grand Total"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"supplier contract summary is missing columns: {missing}")

    wide = df.copy()
    category = wide["component"].astype(str)
    total_mask = category.eq("Grand Total")
    body = wide.loc[~total_mask].copy()
    total_rows = wide.loc[total_mask]
    if total_rows.empty:
        raise ValueError("supplier contract summary has no Grand Total row")
    total_row = total_rows.iloc[0]

    country_cols = ["GB", "DE", "NL", "DK", "BE"]
    numeric_cols = country_cols + ["Grand Total"]
    for column in numeric_cols:
        body[column] = pd.to_numeric(body[column], errors="raise").astype(int)

    def escape(value) -> str:
        return str(value).replace("&", r"\&")

    colfmt = r"@{\extracolsep{\fill}}ll" + "r" * len(country_cols + ["Total"])
    header = (
        r"\multicolumn{1}{c}{\textbf{Category}} & "
        r"\multicolumn{1}{c}{\textbf{Component}} & "
        + " & ".join(
            r"\multicolumn{1}{c}{\textbf{" + c + r"}}"
            for c in country_cols + ["Total"]
        )
        + " \\\\"
    )

    lines = [r"\begingroup", font_size]
    lines.extend([
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}pt}}",
        rf"\renewcommand{{\arraystretch}}{{{arraystretch}}}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\begin{longtable}{" + colfmt + r"}",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ])

    body["component_print"] = body["component"].where(
        body["component"].ne(body["component"].shift()), ""
    )
    previous = None
    for _, row in body.iterrows():
        current = row["component"]
        if previous is not None and current != previous:
            lines.append(r"\midrule")
        previous = current
        values = [
            escape(row["component_print"]),
            escape(row["subcomponent"]),
            *[int(row[column]) for column in country_cols],
            int(row["Grand Total"]),
        ]
        lines.append(
            f"{values[0]} & {values[1]} & "
            + " & ".join(r"\num{" + str(value) + r"}" for value in values[2:])
            + r" \\"
        )

    lines.extend([
        r"\midrule",
        r" & \textbf{Total} & "
        + " & ".join(
            r"{\bfseries\num{" + str(int(total_row[column])) + r"}}"
            for column in country_cols + ["Grand Total"]
        )
        + r" \\",
        r"\end{longtable}",
        r"\endgroup",
        "",
        r"\addtocounter{table}{-1}",
        r"\refstepcounter{table}\label{" + label + r"}",
        r"Table \thetable: " + caption,
    ])
    return "\n".join(lines)


def origin_family(origin):
    s = str(origin).strip().lower()
    if "import" in s:         # "Import", "Imports", "Imported", etc.
        return "imports"
    if s.startswith("assembly"):
        return "assembly"
    if s.startswith("bom"):
        return "bom"
    if s.startswith("indirect"):
        return "indirect"
    return "other"

def build_origin_color_map(origins):
    """
    Assign colors with semantic correlation:
    - Imports: same base hue, multiple shades (gradient)
    - Assembly/BoM/Indirect: each family its own base hue, shades within family
    """
    BASE = {
        "imports":  "#972E1E",  # sober blue (Tableau-like)
        "assembly": "#4C53A8",  # muted teal
        "bom":      "#2D6725",  # muted green
        "indirect": "#D2CD94",  # cool gray
        "other":    "#8B8B8B",
    }

    # collect members per family (preserve input order)
    fam_members = {}
    for o in origins:
        fam = origin_family(o)
        fam_members.setdefault(fam, []).append(o)

    origin_to_color = {}
    for fam, members in fam_members.items():
        base = BASE.get(fam, BASE["other"])

        n = len(members)
        if n == 1:
            factors = [1.0]
        else:
            # smooth shades: darker -> lighter (keeps “same family” feel)
            factors = np.linspace(0.80, 1.20, n)

        for o, f in zip(members, factors):
            origin_to_color[o] = adjust_lightness(base, float(f))

    return origin_to_color


def plot_alloc_share_stacked(
    df_alloc,
    *,
    deployment_country="BE",
    filename="alloc_share_stacked.pdf",
    label_threshold=0.05,
    legend_font=12,
    save=SAVE_FIGURES, close=CLOSE_FIGURES,
):
    set_house_style()

    dfh = df_alloc[df_alloc["deployment_country"] == deployment_country].copy()
    dfh["sub_key"] = (
        dfh["component_group"].astype(str).str.strip()
        + " | "
        + dfh["sub_component"].astype(str).str.strip()
    )

    vs_df = (
        dfh.groupby(["origin_type", "sub_key"], as_index=False)["value_alloc"]
        .sum()
    )

    M = (
        vs_df.pivot(index="origin_type", columns="sub_key", values="value_alloc")
        .fillna(0.0)
    )

    col_sum = M.sum(axis=0).replace(0, np.nan)
    M = M.div(col_sum, axis=1).fillna(0.0)

    rename_origin = {
        "Assembly - Known facility": "Assembly - OEM facility",
        "BoM - Unknown - Inhouse": "BoM - Unknown - OEM Inhouse",
        "Indirect - OEM HQ": "Indirect - OEM",
        "Indirect - Tower": "Indirect - OEM",
    }

    M = M.rename(index=rename_origin)
    M = M.groupby(level=0).sum()

    desired_order = [
        "Indirect - OEM",
        "Indirect - Imports",
        "Assembly - OEM facility",
        "Assembly - Imports",
        "BoM - OEM known supplier",
        "BoM - Unknown - OEM Inhouse",
        "BoM - Unknown - Outsourced",
        "BoM - Imports",
    ]

    ordered_index = [k for k in desired_order if k in M.index] + [k for k in M.index if k not in desired_order]
    M = M.loc[ordered_index]
    origin_to_color = build_origin_color_map(M.index)
    M = M[sorted(M.columns)]

    fig_w = max(12, 0.30 * M.shape[1])
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    x = np.arange(M.shape[1])
    bottom = np.zeros(M.shape[1])

    for origin in M.index:
        values = M.loc[origin].values
        color = origin_to_color[origin]

        bars = ax.bar(
            x, values, bottom=bottom,
            label=tex_safe(origin),
            width=0.9,
            color=color,
        )

        # Share labels
        for xi, v, b in zip(x, values, bottom):
            if v >= label_threshold:
                ax.text(
                    xi,
                    b + v/2,
                    tex_bold(f"{v*100:.0f}%"),
                    ha="center", va="center",
                    fontsize=7,
                    color=get_text_color(color),  # <-- per-segment contrast
                )

        bottom += values

    # Always bold labels (LaTeX-bold)
    ax.set_xlabel(tex_bold("Component group | Sub-component"))
    ax.set_ylabel(tex_bold("Share"))

    # Always comment out title
    # ax.set_title(tex_bold("Sub-component cost share by allocation rule"))

    ax.set_xticks(x)
    ax.set_xticklabels([tex_safe(c) for c in M.columns], rotation=90, ha="center", fontsize=7)
    ax.set_ylim(0, 1)

    # Legend title bold (LaTeX-bold)
    ax.legend(
        title=tex_bold("Origin type"),
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        fontsize=legend_font,
        title_fontsize=legend_font,
        frameon=False,
    )

    apply_house_style(ax, grid=True)

    outpath = save_and_close(fig=fig, filename=filename, save=save, close=close)
    return outpath


def build_weighted_avg_flows_hier(
    flows_hier: dict,
    connected_country: dict,
    *,
    avg_dep_order=("GB","BE","NL","DK","DE"),
    weight_key: str = "Power (MW)",
    label: str = "AVG",
) -> dict:
    """
    Build a synthetic flows_hier for a weighted-average deployment.
    Values are averaged using connected capacity weights.
    """
    avg_dep_order = tuple(str(d).strip() for d in (avg_dep_order or ()))
    weights = {}
    for d in avg_dep_order:
        if d not in flows_hier:
            continue
        w = float(connected_country.get(d, {}).get(weight_key, 0.0))
        if w > 0:
            weights[d] = w

    if not weights:
        raise ValueError("No deployments with positive weights found in avg_dep_order.")

    w_sum = sum(weights.values()) or 1.0
    out = {label: {}}

    for d, w in weights.items():
        data = flows_hier.get(d, {})
        for parent, child_dict in data.items():
            for child, origin_dict in child_dict.items():
                for origin, v in origin_dict.items():
                    parent_dict = out[label].setdefault(parent, {})
                    child_map = parent_dict.setdefault(child, {})
                    key = str(origin)
                    child_map[key] = child_map.get(key, 0.0) + (w / w_sum) * float(v)

    return out
def plot_macro_vertical(
    flows_hier, *,
    dep_order=("GB","BE","NL","DK","DE"),
    comp_order=("Devex","Turbine","BoP","Installation","O&M"),
    split_iso=("DK", "NL", "DE", "BE", "ES", "FR", "NO"),
    split_names=None,
    row_style=("white","black",0.4),
    figsize=(12.0617, 3.8818),
    dpi=160,
    filename=None,
    save=False,
    close=True,
    exact_size=True,
):
    """
    Plot macro-level vertical sourcing breakdown across deployment countries.

    Parameters
    ----------
    flows_hier : dict
        Hierarchical flow data by country and component
    dep_order : tuple, default ("GB","BE","NL","DK","DE")
        Order of deployment countries to display
    comp_order : tuple, default ("Devex","Turbine","BoP","Installation","O&M")
        Order of components to display
    split_iso : tuple, default ()
        ISO codes of countries to split out from EU27
    split_names : dict, optional
        Mapping of ISO codes to display names
    row_style : tuple, default ("white","black",0.4)
        Style parameters for RoW display
    figsize : tuple, default (12.2, 4.0)
        Figure size in inches
    dpi : int, default 160
        Figure resolution
    filename : str, optional
        Output filename for saving (e.g., '4.6.1_EU_macro_sourcing.pdf')
    save : bool, default False
        Whether to save the figure
    close : bool, default True
        Whether to close the figure after saving

    Returns
    -------
    str or tuple
        If filename provided: output path (str) or None
        If filename is None: (fig, ax) tuple
    """
    set_house_style()
    split_names = split_names or {}
    split_iso = tuple(
        code
        for code in (str(c).strip().upper() for c in (split_iso or ()))
        if code not in {"ROW", "X01"}
    )
    EU_DISPLAY = OTHER_EU27_LABEL if any(code in EU27 for code in split_iso) else EU_LABEL
    split_labels = [
        split_names.get(c) or CODE_TO_NAME.get(c) or canonical_name(c) or c
        for c in split_iso
    ]

    # Keep the United Kingdom and RoW at the end of the legend/order.
    norway_label = split_names.get("NO") or CODE_TO_NAME.get("NO", "Norway")
    origin_list = (
        [EU_DISPLAY]
        + [l for l in split_labels if l != norway_label]
        + [UK_LABEL]
        + ([norway_label] if norway_label in split_labels else [])
        + [ROW_LABEL, OTHER_ORIGINS_LABEL]
    )

    STYLE_MAP = {
        EU_LABEL: dict(fc=COUNTRY_COLOR_MAP.get("EU27") or COUNTRY_COLOR_MAP.get(EU_LABEL) or "#4C78A8", ec="#1F3A5F", ls="solid", lw=0.6),
        OTHER_EU27_LABEL: dict(
            fc=adjust_lightness(COUNTRY_COLOR_MAP.get("EU27") or COUNTRY_COLOR_MAP.get(EU_LABEL) or "#4C78A8", 0.45),
            ec="#6B7FA3", ls="solid", lw=0.6,
        ),
        OTHER_ORIGINS_LABEL: dict(fc="black", ec="black", ls="solid", lw=0.6),
        UK_LABEL: dict(fc=COUNTRY_COLOR_MAP.get("GB") or "#5C1F1F", ec="#5C1F1F", ls="solid", lw=0.6),
        "Norway": dict(fc=COUNTRY_COLOR_MAP.get("NO") or "#2F6F6D", ec=COUNTRY_COLOR_MAP.get("NO") or "#2F6F6D", ls="solid", lw=0.6),
        ROW_LABEL: dict(fc="white", ec="black", ls=(0, (3, 2)), lw=0.6),
    }

    def _bucket(o):
        o = str(o).strip().upper()
        if o in {"ROW", "X01"}:
            return ROW_LABEL
        if o in {"GB", "UK"}:
            return UK_LABEL
        if o in split_iso:
            return split_names.get(o) or CODE_TO_NAME.get(o) or canonical_name(o) or o
        if o in EU27:
            return EU_DISPLAY
        return OTHER_ORIGINS_LABEL

    def _color(label):
        if label == ROW_LABEL:
            return None
        # Map display labels such as "Denmark" back to ISO codes.
        inv = {
            (split_names.get(k) or CODE_TO_NAME.get(k) or canonical_name(k) or k): k
            for k in split_iso
        }
        iso = inv.get(label, label)
        return (COUNTRY_COLOR_MAP.get(label) or COUNTRY_COLOR_MAP.get(iso) or COUNTRY_COLOR_MAP.get(str(iso)) or "#BDBDBD")

    comps0 = []
    for c in comp_order:
        if any((d in flows_hier and c in flows_hier[d]) for d in dep_order):
            comps0.append(c)
    extra = []
    for d in dep_order:
        if d in flows_hier:
            for c in flows_hier[d].keys():
                if c not in comps0 and c not in extra:
                    extra.append(c)
    comps = comps0 + extra

    macro_vals = {c: {d: defaultdict(float) for d in dep_order} for c in comps}
    macro_tot = {d: defaultdict(float) for d in dep_order}

    for d in dep_order:
        if d not in flows_hier:
            continue
        data = flows_hier[d]
        for c in comps:
            if c not in data:
                continue
            for sub, odict in data[c].items():
                for origin, v in odict.items():
                    macro_vals[c][d][_bucket(origin)] += float(v)

    for d in dep_order:
        for c in comps:
            for b in origin_list:
                macro_tot[d][b] += macro_vals[c][d].get(b, 0.0)

    origin_list = [
        label for label in origin_list
        if max(
            (abs(float(macro_tot[d].get(label, 0.0))) for d in dep_order),
            default=0.0,
        ) > 1e-12
    ]
    if not origin_list:
        raise ValueError("No nonzero sourcing-origin contributions to plot.")

    def _shares(bd):
        s = sum(bd.get(b, 0.0) for b in origin_list) or 1.0
        return [bd.get(b, 0.0) / s for b in origin_list]

    blocks = [("Total", dep_order)] + [(c, dep_order) for c in comps]
    x, xt, centers, seps, k = [], [], [], [], 0
    for gi, (g, ds) in enumerate(blocks):
        start = k
        for d in ds:
            x.append(k); xt.append(d); k += 1
        end = k - 1
        centers.append((g, (start + end) / 2))
        if gi < len(blocks) - 1:
            seps.append(k)
        k += 1

    vals = []
    for d in dep_order:
        vals.append(_shares(macro_tot[d]))
    for c in comps:
        for d in dep_order:
            vals.append(_shares(macro_vals[c][d]))
    vals = np.array(vals)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    bottoms = np.zeros(vals.shape[0])
    xs = x[:vals.shape[0]]

    for j, lab in enumerate(origin_list):
        seg = vals[:, j]

        if lab in STYLE_MAP:
            st = STYLE_MAP[lab]
            fc = st["fc"]
            ec = st["ec"]
            lw = st["lw"]
            ls = st["ls"]
        else:
            fc = _color(lab)
            ec = fc
            lw = 0.6
            ls = "solid"

        ax.bar(
            xs,
            seg,
            bottom=bottoms,
            width=0.85,
            color=fc,
            edgecolor=ec,
            linewidth=lw,
            linestyle=ls,
        )

        bottoms += seg

    for i in range(vals.shape[0]):
        cum = 0.0
        for j, lab in enumerate(origin_list):
            w = float(vals[i, j])
            if w < 0.04:
                cum += w
                continue
            if lab == OTHER_ORIGINS_LABEL:
                tc = "white"
            elif lab == ROW_LABEL:
                tc = "black"
            elif lab in STYLE_MAP:
                tc = get_text_color(STYLE_MAP[lab]["fc"])
            else:
                tc = get_text_color(_color(lab))
            ax.text(xs[i], cum + w/2, tex_safe(f"{w*100:.0f}%"),
                    ha="center", va="center", fontsize=7, color=tc)
            cum += w

    for s in seps:
        ax.axvline(s, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

    ax.set_ylim(0, 1)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels([tex_safe(t) for t in ["0%","25%","50%","75%","100%"]], fontsize=10)
    ax.set_ylabel(tex_bold("Share of initial sourcing by origin"))
    ax.set_xticks(xs)
    ax.set_xticklabels([tex_safe("GB" if t in {"UK","GB"} else t) for t in xt[:vals.shape[0]]], fontsize=10)
    ax.tick_params(axis="x", length=0)

    for g, c in centers:
        ax.text(c, -0.08, tex_safe(g), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=12, fontweight="bold")

    # --- ESTA ERA LA PARTE ROTA: AHORA BIEN HECHA ---
    patches = []
    for lab in origin_list:
        if lab in STYLE_MAP:
            st = STYLE_MAP[lab]
            fc = st["fc"]
            ec = st["ec"]
            lw = st["lw"]
            ls = st["ls"]
        else:
            fc = _color(lab)
            ec = fc                  # <<< borde = mismo color
            lw = 0.6
            ls = "solid"

        patches.append(
            mpatches.Patch(
                facecolor=fc,
                edgecolor=ec,
                linewidth=lw,
                linestyle=ls,
                label=tex_safe(
                    EU_DISPLAY if lab in {EU_LABEL, OTHER_EU27_LABEL} else ("GB" if lab in {"UK","GB"} else lab)
                ),
            )
        )
    ax.legend(
        handles=patches,
        title=tex_bold("Origin"),
        bbox_to_anchor=(0.98, 0.5),
        loc="center left",
        fontsize=7,
        title_fontsize=7,
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="#CCCCCC",
    )

    apply_house_style(ax, grid=True)
    ax.grid(axis="y", alpha=0.2, linestyle=":", linewidth=0.5)
    ax.set_title(None)
    fig.tight_layout()

    # Save and close if filename provided
    if filename is not None:
        result = _save_and_close_exact_aware(
            fig=fig,
            filename=filename,
            save=save,
            close=close,
            exact_size=exact_size,
        )
        return result
    else:
        if close:
            plt.close(fig)
        return fig, ax



def plot_eu27_weighted_sourcing(
    flows_eu27_avg: dict,
    *,
    split_iso=(),
    split_names=None,
    figsize=(12.0617, 3.8818),
    dpi=160,
    filename="3.1_EU_weighted_sourcing.pdf",
    save=SAVE_FIGURES,
    close=CLOSE_FIGURES,
    label_min_share: float = 0.04,
):
    """Plot sourcing-country shares using the capture-chart layout."""
    set_house_style()

    data = flows_eu27_avg["EU27_AVG"]
    split_names = split_names or {}
    split_iso = tuple(str(country).strip().upper() for country in (split_iso or ()))

    if not 0 <= float(label_min_share) <= 1:
        raise ValueError("label_min_share must be between 0 and 1.")
    label_min_share = float(label_min_share)

    split_labels = [
        split_names.get(country) or CODE_TO_NAME.get(country) or canonical_name(country) or country
        for country in split_iso
    ]

    has_eu_split = any(country in EU27 for country in split_iso)
    eu_display = OTHER_EU27_LABEL if has_eu_split else EU_LABEL

    row_label = "RoW"
    origin_order = [EU_LABEL] + list(split_labels) + [row_label]

    def _bucket(origin):
        code = str(origin).strip().upper()
        if code in split_iso:
            return split_labels[split_iso.index(code)]
        if code in EU27:
            return EU_LABEL
        return row_label

    records = []
    for component, subcomponents in data.items():
        for subcomponent, origin_values in subcomponents.items():
            values = defaultdict(float)
            for origin, value in origin_values.items():
                values[_bucket(origin)] += float(value)

            total = sum(values.values())
            if not np.isfinite(total) or total <= 0:
                raise ValueError(
                    f"Non-positive sourcing total for {component} / {subcomponent}: {total}"
                )

            shares = {label: values[label] / total for label in origin_order}
            shares[origin_order[-1]] += 1.0 - sum(shares.values())

            records.append({
                "Component": str(component),
                "Subcomponent": str(subcomponent),
                **shares,
            })

    if not records:
        raise ValueError("flows_eu27_avg['EU27_AVG'] contains no sourcing flows to plot.")

    df_abs = pd.DataFrame(records)
    df_abs["Component"] = df_abs["Component"].astype(str)
    df_abs["Subcomponent"] = df_abs["Subcomponent"].astype(str)

    pref = ["Devex", "Turbine", "BoP", "Installation", "O&M"]
    comp_order = [c for c in pref if c in df_abs["Component"].unique().tolist()]
    comp_order += [c for c in df_abs["Component"].unique().tolist() if c not in comp_order]

    x, xt, centers, seps, rows, k = [], [], [], [], [], 0
    for gi, component in enumerate(comp_order):
        subdf = df_abs[df_abs["Component"] == component].sort_values(["Subcomponent"]).copy()
        if subdf.empty:
            continue

        start = k
        for _, row in subdf.iterrows():
            x.append(k)
            xt.append(row["Subcomponent"])
            rows.append(row)
            k += 1
        end = k - 1

        centers.append((component, (start + end) / 2))
        if gi < len(comp_order) - 1:
            seps.append(k)
        k += 1

    plot_df = pd.DataFrame(rows).reset_index(drop=True)
    # Keep the legacy 3.1 layout, but do not retain a zero-sized "Other EU27"
    # legend entry once every contributing EU27 origin is shown separately.
    if has_eu_split and float(np.nanmax(np.abs(plot_df[EU_LABEL].to_numpy(dtype=float)))) <= 1e-12:
        origin_order = [label for label in origin_order if label != EU_LABEL]
    vals = plot_df[list(origin_order)].to_numpy(dtype=float)
    xs = np.asarray(x[:vals.shape[0]], dtype=float)

    style_map = {
        EU_LABEL: dict(
            fc=COUNTRY_COLOR_MAP.get("EU27") or COUNTRY_COLOR_MAP.get(EU_LABEL) or "#4C78A8",
            ec="#1F3A5F",
            ls="solid",
            lw=0.6,
        ),
        row_label: dict(
            fc="white",
            ec="black",
            ls=(0, (3, 2)),
            lw=0.6,
        ),
    }

    def _color(label):
        if label == row_label:
            return None

        inverse = {
            (split_names.get(country) or CODE_TO_NAME.get(country) or canonical_name(country) or country): country
            for country in split_iso
        }
        iso = inverse.get(label, label)

        return (
            COUNTRY_COLOR_MAP.get(label)
            or COUNTRY_COLOR_MAP.get(iso)
            or COUNTRY_COLOR_MAP.get(str(iso))
            or FALLBACK_GREY
        )

    def _style(label):
        if label in style_map:
            return style_map[label]
        color = _color(label)
        return dict(fc=color, ec=color, ls="solid", lw=0.6)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    bottoms = np.zeros(vals.shape[0])
    for j, label in enumerate(origin_order):
        segment = vals[:, j]
        style = _style(label)
        ax.bar(
            xs,
            segment,
            bottom=bottoms,
            width=0.85,
            color=style["fc"],
            edgecolor=style["ec"],
            linewidth=style["lw"],
            linestyle=style["ls"],
        )
        bottoms += segment

    for i in range(vals.shape[0]):
        cum = 0.0
        for j, label in enumerate(origin_order):
            share = float(vals[i, j])
            if abs(share) < 1e-12:
                continue
            if share < label_min_share:
                cum += share
                continue

            style = _style(label)
            ax.text(
                xs[i],
                cum + share / 2,
                tex_safe(f"{share * 100:.0f}%"),
                ha="center",
                va="center",
                fontsize=9,
            color="black" if label == row_label else get_text_color(style["fc"]),
            )
            cum += share

    for separator in seps:
        ax.axvline(separator, color="black", linewidth=1.0, linestyle="--", alpha=0.6)

    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(
        [tex_safe(label) for label in ("0%", "25%", "50%", "75%", "100%")],
        fontsize=10,
    )
    ax.set_ylabel(tex_bold("Share of initial sourcing by origin"), fontsize=10)

    ax.set_xticks(xs)
    xt_disp = [_short_subcomponent_label(label) for label in xt[:vals.shape[0]]]
    ax.set_xticklabels(
        [tex_safe(label) for label in xt_disp],
        fontsize=10,
        rotation=45,
        ha="right",
    )
    ax.tick_params(axis="x", length=0)

    for component, center in centers:
        ax.text(
            center,
            -0.29,
            tex_safe(str(component)),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
        )

    legend_names = {
        EU_LABEL: eu_display,
        row_label: row_label,
    }

    patches = [
        mpatches.Patch(
            facecolor=_style(label)["fc"],
            edgecolor=_style(label)["ec"],
            linewidth=_style(label)["lw"],
            linestyle=_style(label)["ls"],
            label=tex_safe(legend_names.get(label, label)),
        )
        for label in origin_order
    ]

    ax.legend(
        handles=patches,
        title=tex_bold("Supplying country"),
        bbox_to_anchor=(0.995, 0.5),
        loc="center left",
        fontsize=9,
        title_fontsize=9,
        frameon=True,
        framealpha=0.9,
        facecolor="#F2F2F2",
        edgecolor="#BDBDBD",
        borderpad=0.35,
        labelspacing=0.35,
        handletextpad=0.5,
        borderaxespad=0.25,
    )

    apply_house_style(ax, grid=True)
    ax.grid(axis="y", alpha=0.2, linestyle=":", linewidth=0.5)
    ax.set_title(None)

    # Ajuste manual para que ambos gráficos queden iguales,
    # sin espacio excesivo a la derecha ni abajo.
    fig.subplots_adjust(left=0.075, right=0.865, bottom=0.27, top=0.96)

    if filename is None:
        return fig, ax

    return save_and_close(fig=fig, filename=filename, save=save, close=close)


def _short_component_label(s: str) -> str:
    s = str(s)
    mapping = {
        "Development & consenting services": "Dev. & consenting",
        "Engineering & consultancy": "Eng. & consult.",
        "Project management": "Proj. mgmt.",
    }
    return mapping.get(s, s)




def _get_gb_inputs():
    import importlib
    import modules.supply_chain.inputs as sc_inputs
    if not hasattr(sc_inputs, "GB_local") or not hasattr(sc_inputs, "GB_CATEGORY_MAP"):
        sc_inputs = importlib.reload(sc_inputs)
    if not hasattr(sc_inputs, "GB_local") or not hasattr(sc_inputs, "GB_CATEGORY_MAP"):
        raise ImportError("GB_local / GB_CATEGORY_MAP not found in modules.supply_chain.inputs")
    return sc_inputs.GB_local, sc_inputs.GB_CATEGORY_MAP


def _norm_gb_label(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("&", " and ")
    s = s.replace("-", " ")
    s = " ".join(s.split())
    if s == "oss":
        s = "offshore substation"
    if s.endswith("cables"):
        s = s[:-1]
    return s


def _map_gb_category(cost_cat: str, cost_type: str | None) -> str | None:
    GB_local, GB_CATEGORY_MAP = _get_gb_inputs()
    cc = _norm_gb_label(cost_cat)
    ct = _norm_gb_label(cost_type) if cost_type is not None else None

    # exact matches (CostType provided)
    for (cc_raw, ct_raw), target in GB_CATEGORY_MAP.items():
        if ct_raw is None:
            continue
        if _norm_gb_label(cc_raw) == cc and _norm_gb_label(ct_raw) == ct:
            return target

    # fallback (CostType ignored)
    for (cc_raw, ct_raw), target in GB_CATEGORY_MAP.items():
        if ct_raw is not None:
            continue
        if _norm_gb_label(cc_raw) == cc:
            return target

    # fallback: for aggregated benchmark-only categories, map all subcomponents
    if cc in {"devex", "o and m", "turbine", "abex"}:
        if cc == "o and m":
            return "O&M"
        return cost_cat

    return None


def build_gb_local_benchmark_table(
    flows_hier: dict,
    *,
    deploy: str = "GB",
    include_total: bool = False,
    general_parents: tuple[str, ...] = ("Turbine",),
    include_abex: bool = False,
) -> pd.DataFrame:
    """
    Build model vs benchmark local content shares for GB (UK),
    using the model's hierarchical structure on the x-axis.
    """
    if deploy not in flows_hier:
        raise KeyError(f"Deployment '{deploy}' not found in flows_hier.")

    GB_local, _ = _get_gb_inputs()
    bench_data = GB_local["data"]
    bench_norm = {_norm_gb_label(k): v for k, v in bench_data.items()}

    rows = []

    def _local_share(origin_dict: dict) -> float:
        total_val = sum(origin_dict.values())
        if total_val == 0:
            return np.nan
        local_val = sum(v for o, v in origin_dict.items() if str(o) == deploy)
        return local_val / total_val * 100.0

    data = flows_hier[deploy]
    for parent, child_dict in data.items():
        if parent == "ABEX" and not include_abex:
            continue

        # Aggregate general parents (Devex, Turbine, O&M)
        if parent in general_parents:
            total_val = 0.0
            local_val = 0.0
            for origin_dict in child_dict.values():
                total_val += sum(origin_dict.values())
                local_val += sum(v for o, v in origin_dict.items() if str(o) == deploy)
            model_share = (local_val / total_val * 100.0) if total_val else np.nan

            mapped = _map_gb_category(parent, parent)
            bench_share = bench_norm.get(_norm_gb_label(mapped), np.nan) if mapped else np.nan

            rows.append({
                "Parent": parent,
                "Child": parent,
                "Label": parent,
                "Model": model_share,
                "Benchmark": bench_share,
                "BenchmarkCategory": mapped,
            })
            continue

        # For BoP & Installation: keep model subcategories
        for child, origin_dict in child_dict.items():
            mapped = _map_gb_category(parent, child)
            model_share = _local_share(origin_dict)
            bench_share = bench_norm.get(_norm_gb_label(mapped), np.nan) if mapped else np.nan
            rows.append({
                "Parent": parent,
                "Child": child,
                "Label": f"{parent} \u2013 {child}",
                "Model": model_share,
                "Benchmark": bench_share,
                "BenchmarkCategory": mapped,
            })

    # Optional total (across included rows)
    if include_total and "Total" in bench_data:
        model_vals = [r["Model"] for r in rows if np.isfinite(r["Model"])]
        model_total = np.nan if len(model_vals) == 0 else np.average(model_vals)
        rows.append({
            "Parent": "Total",
            "Child": "Total",
            "Label": "Total",
            "Model": model_total,
            "Benchmark": bench_norm.get(_norm_gb_label("Total"), np.nan),
            "BenchmarkCategory": "Total",
        })

    return pd.DataFrame(rows)


def plot_gb_local_benchmark(
    flows_hier: dict,
    *,
    deploy: str = "GB",
    include_total: bool = False,
    general_parents: tuple[str, ...] = ("Turbine",),
    include_abex: bool = False,
    parent_order: tuple[str, ...] | None = None,
    exclude_parents: tuple[str, ...] = ("Turbine",),
    steen_matrices: dict | None = None,
    steen_label: str = "Steen",
    benchmark_label: str = "LC benchmark BVG (2021)",
    contract_label: str = "Contract-share (2000-2025)",
    asset_label: str = "Asset-share (2000-2019)",
    include_model: bool = True,
    extra_series: list[tuple[str, dict, str]] | None = None,
    show_k: bool = True,
    k_label: str = r"$k_c$",
    k_color: str | None = None,
    k_marker: str = "x",
    k_size: float = 12.0,
    group_gap: float = 1.0,
    label_min: float = 2.0,
    xtick_font: int = 7,
    value_font: int = 6,
    legend_shift: float = 2.3,
    legend_font: int = 8,
    bar_width: float = 0.30,
    bar_span: float | None = 0.20,
    figsize: tuple = (7.5, 4.2),
    filename: str | None = "B.1.3.pdf",
    save=SAVE_FIGURES,
    close=CLOSE_FIGURES,
):
    """
    Compare model GB local content vs GB benchmark (BVG).
    """
    set_house_style()
    df = build_gb_local_benchmark_table(
        flows_hier,
        deploy=deploy,
        include_total=include_total,
        general_parents=general_parents,
        include_abex=include_abex,
    )
    if exclude_parents:
        df = df.loc[~df["Parent"].isin(exclude_parents)].copy()

    def _matrix_local_share(mats: dict | None, parent: str, child: str) -> float:
        if mats is None:
            return np.nan
        if parent not in mats:
            return np.nan

        mat = mats[parent].get(child)
        if mat is None:
            # fallback: use _TOTAL if present
            mat = mats[parent].get("_TOTAL")
        if mat is None and child == parent:
            # fallback: average local share across all components of parent
            vals = []
            for m in mats[parent].values():
                if not isinstance(m, pd.DataFrame):
                    continue
                dep_col = deploy
                if dep_col not in m.columns:
                    if dep_col == "GB" and "UK" in m.columns:
                        dep_col = "UK"
                    elif dep_col == "UK" and "GB" in m.columns:
                        dep_col = "GB"
                    else:
                        continue
                for o in [deploy, "GB", "UK"]:
                    if o in m.index:
                        vals.append(float(m.loc[o, dep_col]) * 100.0)
                        break
            return float(np.nanmean(vals)) if vals else np.nan

        if mat is None:
            return np.nan

        # pick deploy column
        dep_col = deploy
        if dep_col not in mat.columns:
            if dep_col == "GB" and "UK" in mat.columns:
                dep_col = "UK"
            elif dep_col == "UK" and "GB" in mat.columns:
                dep_col = "GB"
            else:
                return np.nan
        # pick origin row
        for o in [deploy, "GB", "UK"]:
            if o in mat.index:
                return float(mat.loc[o, dep_col]) * 100.0
        return np.nan

    df["Steen"] = [
        _matrix_local_share(steen_matrices, p, c) for p, c in zip(df["Parent"], df["Child"])
    ]

    # ---- order rows by parent groups (macro categories)
    if parent_order is None:
        parent_order = tuple(CAT_ORDER)
    if exclude_parents:
        parent_order = tuple(p for p in parent_order if p not in exclude_parents)
    parent_order = [p for p in parent_order if p in df["Parent"].unique()]
    remaining = [p for p in df["Parent"].unique() if p not in parent_order]
    parent_order = parent_order + remaining

    # ---- x positions with group gaps + centers for macro labels
    x = []
    centers = []
    seps = []
    pos = 0.0
    parents_in_plot = [p for p in parent_order if (df["Parent"] == p).any()]
    row_idx = []
    for p in parents_in_plot:
        sub = df[df["Parent"] == p].copy()
        if sub.empty:
            continue
        sub["_child_sort"] = sub["Child"].astype(str).str.casefold()
        sub = sub.sort_values("_child_sort")
        row_idx.extend(sub.index.tolist())
    df = df.loc[row_idx].reset_index(drop=True)
    for i, p in enumerate(parents_in_plot):
        rows_p = df.index[df["Parent"] == p].tolist()
        if not rows_p:
            continue
        start = pos
        for _ in rows_p:
            x.append(pos)
            pos += 1.0
        end = pos - 1.0
        centers.append((p, (start + end) / 2.0))
        if i < len(parents_in_plot) - 1:
            seps.append(pos - 0.5 + group_gap / 2.0)
            pos += group_gap
    x = np.array(x, dtype=float)

    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    width = bar_width
    if bar_span is None:
        bar_span = width * 0.8

    series = []
    if include_model:
        series.append((contract_label, df["Model"].values, PALETTE_CARDO["secondary"]))
    series.append((benchmark_label, df["Benchmark"].values, PALETTE_CARDO["light"]))
    if steen_matrices is not None and np.isfinite(df["Steen"].values).any():
        series.append((asset_label, df["Steen"].values, PALETTE_CARDO["highlight"]))

    if extra_series:
        for label, mats, color in extra_series:
            vals = [
                _matrix_local_share(mats, p, c) for p, c in zip(df["Parent"], df["Child"])
            ]
            series.append((label, np.array(vals, dtype=float), color))

    offsets = np.linspace(-bar_span, bar_span, len(series))
    bar_groups = []
    label_offsets = {}
    for (label, vals, color), off in zip(series, offsets):
        vals = np.array(vals, dtype=float)
        mask = np.isfinite(vals)
        bars = ax.bar(
            x[mask] + off,
            vals[mask],
            width,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=tex_safe(label),
        )
        bar_groups.append((label, bars, color, mask))
        label_offsets[label] = off

    # close benchmark vs contract -> single label
    close_mask = np.zeros(len(df), dtype=bool)
    if include_model:
        bench_vals = df["Benchmark"].values.astype(float)
        model_vals = df["Model"].values.astype(float)
        close_mask = np.isfinite(bench_vals) & np.isfinite(model_vals) & (np.abs(bench_vals - model_vals) <= 1.0)

    def _label_bars(label, bars, mask):
        idxs = np.where(mask)[0]
        for b, idx in zip(bars, idxs):
            h = b.get_height()
            if not np.isfinite(h) or h < label_min:
                continue
            if label in (benchmark_label, contract_label) and close_mask[idx]:
                continue
            y = min(h + 1.0, 102.0)
            ax.text(
                b.get_x() + b.get_width() / 2,
                y,
                tex_bold(f"{h:.0f}%"),
                ha="center",
                va="bottom",
                fontsize=value_font,
                color="#222222",
                clip_on=True,
            )

    for label, bars, color, mask in bar_groups:
        _label_bars(label, bars, mask)

    # merged labels for close benchmark/contract
    if include_model and np.any(close_mask):
        off_b = label_offsets.get(benchmark_label, 0.0)
        off_m = label_offsets.get(contract_label, 0.0)
        for i, is_close in enumerate(close_mask):
            if not is_close:
                continue
            v_b = df["Benchmark"].iloc[i]
            v_m = df["Model"].iloc[i]
            v = float(np.nanmean([v_b, v_m]))
            if not np.isfinite(v) or v < label_min:
                continue
            y = min(max(v_b, v_m) + 1.0, 102.0)
            x_mid = x[i] + (off_b + off_m) / 2.0
            ax.text(
                x_mid,
                y,
                tex_bold(f"{v:.0f}%"),
                ha="center",
                va="bottom",
                fontsize=value_font,
                color="#222222",
                clip_on=True,
            )

    # ---- optional k_c overlay on secondary axis
    if show_k and include_model:
        if "$" in k_label or "\\" in k_label:
            k_label_render = k_label
        else:
            k_label_render = f"${k_label}$"
        bench_vals = df["Benchmark"].values.astype(float)
        model_vals = df["Model"].values.astype(float)
        k_vals = np.full_like(model_vals, np.nan, dtype=float)
        mask_k = np.isfinite(bench_vals) & np.isfinite(model_vals) & (model_vals > 0)
        k_vals[mask_k] = bench_vals[mask_k] / model_vals[mask_k]

        ax2 = ax.twinx()
        k_col = k_color or PALETTE_CARDO["highlight"]
        ax2.scatter(
            x[mask_k],
            k_vals[mask_k],
            marker=k_marker,
            s=k_size,
            color=k_col,
            linewidths=1.4,
            zorder=6,
        )
        if np.isfinite(k_vals[mask_k]).any():
            k_max = float(np.nanmax(k_vals[mask_k]))
            ax2.set_ylim(0, max(1.05, k_max * 1.10))
        # Bold label in math mode without \textbf underscore issues
        if k_label_render.startswith("$") and k_label_render.endswith("$"):
            inner = k_label_render[1:-1]
            if "\\mathbf" not in inner:
                k_label_bold = f"$\\mathbf{{{inner}}}$"
            else:
                k_label_bold = k_label_render
        else:
            k_label_bold = k_label_render
        ax2.set_ylabel(k_label_bold)
        # Match primary axis tick size and bold ticks
        base_tick = ax.yaxis.get_ticklabels()
        tick_size = base_tick[0].get_fontsize() if base_tick else 8
        ax2.tick_params(axis="y", labelsize=tick_size)
        for t in ax2.get_yticklabels():
            t.set_fontweight("bold")

    def _wrap(s: str) -> str:
        return "\n".join(textwrap.wrap(str(s), width=18))

    # Micro labels (child components). If child == parent, leave blank to avoid repetition.
    micro_labels = []
    for p, c in zip(df["Parent"], df["Child"]):
        if str(p) == str(c):
            micro_labels.append("")
        else:
            micro_labels.append(_short_component_label(c))

    ax.set_xticks(x)
    ax.set_xticklabels(
        [tex_safe(_wrap(c)) for c in micro_labels],
        fontsize=xtick_font,
        rotation=35,
        ha="right",
    )
    ax.tick_params(axis="x", length=0)

    # Macro labels (parent categories) below axis
    for p, cx in centers:
        ax.text(
            cx,
            -0.26,
            tex_bold(str(p)),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11,
        )

    # Vertical separators between groups
    for s in seps:
        ax.axvline(s, color="#7A7A7A", linestyle="--", linewidth=1.2, alpha=0.9, zorder=0)

    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_ylabel(tex_bold("Local content share"))

    # Legend: upper-left, left edge aligned with the first bar's left edge
    left_edge = x.min() + offsets.min() - (width / 2) + 1.70
    handles, labels = ax.get_legend_handles_labels()
    if show_k and include_model:
        from matplotlib.lines import Line2D
        k_col = k_color or PALETTE_CARDO["highlight"]
        handles.append(Line2D([0], [0], marker=k_marker, color=k_col, lw=0, markersize=7))
        labels.append(k_label_render)

    legend = ax.legend(
        handles=handles,
        labels=labels,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(left_edge + legend_shift, ax.get_ylim()[1]),
        bbox_transform=ax.transData,
        fontsize=legend_font,
        prop={"weight": "normal"},
    )
    # Force legend text size/weight (avoid LaTeX bolding artifacts)
    for t in legend.get_texts():
        t.set_fontsize(legend_font)
        t.set_fontweight("normal")

    apply_house_style(ax, grid=True)
    fig.subplots_adjust(bottom=0.26)

    if filename is None:
        return fig, ax, df
    return save_and_close(fig=fig, filename=filename, save=save, close=close)
