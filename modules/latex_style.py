"""
Paper plotting and LaTeX helpers.

Centralizes all formatting, figure saving, and table export utilities
used across the notebook.
"""

from __future__ import annotations

import os
import platform
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from .config import (
    CLOSE_FIGURES,
    CORE_DEPLOYMENT,
    IMG_ROOT,
    SAVE_FIGURES,
    project_relative_path,
)


_DEFAULT_WINDOWS_MIKTEX_BIN = Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64"
_SITECUSTOMIZE_MARKER_BEGIN = "# --- latex_style MiKTeX PATH bootstrap BEGIN ---"
_SITECUSTOMIZE_MARKER_END = "# --- latex_style MiKTeX PATH bootstrap END ---"


def _prepend_path(path: Path) -> None:
    path_s = str(path)
    paths = os.environ.get("PATH", "").split(os.pathsep)
    if not any(os.path.normcase(p) == os.path.normcase(path_s) for p in paths):
        os.environ["PATH"] = path_s + os.pathsep + os.environ.get("PATH", "")


def _run_setup_command(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print(">", " ".join(cmd))
    return subprocess.run(cmd, check=check)


def _conda_executable() -> str | None:
    conda = shutil.which("conda")
    if conda:
        return conda

    candidates = [
        Path(sys.prefix) / "Scripts" / "conda.exe",
        Path(sys.prefix).parent.parent / "Scripts" / "conda.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _conda_environment_name() -> str:
    return os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for part in version.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits == "":
            break
        parts.append(int(digits))
    return tuple(parts)


def _persist_miktex_path_for_environment(miktex_bin: Path) -> Path:
    """Make MiKTeX visible whenever this Python environment starts."""
    sitecustomize = Path(sys.prefix) / "Lib" / "site-packages" / "sitecustomize.py"
    sitecustomize.parent.mkdir(parents=True, exist_ok=True)

    block = f"""{_SITECUSTOMIZE_MARKER_BEGIN}
import os

_THESIS_MIKTEX_BIN = r"{miktex_bin}"
if os.path.isdir(_THESIS_MIKTEX_BIN):
    _latex_paths = os.environ.get("PATH", "").split(os.pathsep)
    if not any(os.path.normcase(p) == os.path.normcase(_THESIS_MIKTEX_BIN) for p in _latex_paths):
        os.environ["PATH"] = _THESIS_MIKTEX_BIN + os.pathsep + os.environ.get("PATH", "")
{_SITECUSTOMIZE_MARKER_END}
"""

    existing = sitecustomize.read_text(encoding="utf-8") if sitecustomize.exists() else ""
    if _SITECUSTOMIZE_MARKER_BEGIN in existing and _SITECUSTOMIZE_MARKER_END in existing:
        before, rest = existing.split(_SITECUSTOMIZE_MARKER_BEGIN, 1)
        _, after = rest.split(_SITECUSTOMIZE_MARKER_END, 1)
        content = before.rstrip() + "\n\n" + block + after
    else:
        content = existing.rstrip() + "\n\n" + block if existing.strip() else block

    sitecustomize.write_text(content, encoding="utf-8")
    return sitecustomize


def ensure_lualatex_pgf_setup(
    *,
    install_miktex: bool = True,
    require_packages: bool = True,
    persist_env_path: bool = True,
    miktex_bin: str | Path | None = None,
) -> Path:
    """
    Ensure the current Python environment can use the paper PGF/LuaLaTeX setup.

    This does not change the plotting style. It only makes ``lualatex`` and the
    packages used by ``set_house_style()`` available to Matplotlib's PGF backend.
    """
    miktex_bin = Path(miktex_bin) if miktex_bin is not None else _DEFAULT_WINDOWS_MIKTEX_BIN

    if miktex_bin.exists():
        _prepend_path(miktex_bin)

    lualatex = shutil.which("lualatex")
    if lualatex is None:
        if platform.system() != "Windows":
            raise RuntimeError(
                "lualatex is not on PATH. Install TeX Live/MiKTeX and make lualatex available."
            )
        if not install_miktex:
            raise RuntimeError("lualatex is not on PATH and install_miktex=False.")
        if shutil.which("winget") is None:
            raise RuntimeError("lualatex is not on PATH and winget is not available.")

        _run_setup_command([
            "winget",
            "install",
            "--id",
            "MiKTeX.MiKTeX",
            "--exact",
            "--scope",
            "user",
            "--silent",
            "--accept-package-agreements",
            "--accept-source-agreements",
        ])

        if miktex_bin.exists():
            _prepend_path(miktex_bin)

    lualatex = shutil.which("lualatex")
    if lualatex is None:
        raise RuntimeError(
            "MiKTeX may be installed, but lualatex is still not on PATH. "
            "Restart the kernel/VS Code and run this setup again."
        )

    print(f"lualatex: {lualatex}")

    miktex = miktex_bin / "miktex.exe"
    if require_packages and miktex.exists():
        _run_setup_command([str(miktex), "packages", "update-package-database"], check=False)
        _run_setup_command([
            str(miktex),
            "packages",
            "require",
            "fontspec",
            "unicode-math",
            "lm",
            "lm-math",
            "lualatex-math",
        ], check=False)
        _run_setup_command([str(miktex), "packages", "check-update"], check=False)

    if persist_env_path and miktex_bin.exists():
        sitecustomize = _persist_miktex_path_for_environment(miktex_bin)
        print(f"Persisted MiKTeX PATH for this Python environment: {sitecustomize}")

    return Path(lualatex)


def ensure_cartopy_setup(*, install: bool = True, channel: str = "conda-forge") -> None:
    """
    Ensure Cartopy is importable in the current Python environment.

    Cartopy is installed with conda because its native GIS dependencies are much
    more reliable from conda-forge on Windows.
    """
    if importlib.util.find_spec("cartopy") is not None:
        print("cartopy: available")
        return

    if not install:
        raise RuntimeError("cartopy is not installed and install=False.")

    conda = _conda_executable()
    if conda is None:
        raise RuntimeError(
            "cartopy is not installed and conda was not found. "
            "Install cartopy in this environment, preferably from conda-forge."
        )

    env_name = _conda_environment_name()
    _run_setup_command([
        conda,
        "install",
        "-n",
        env_name,
        "-c",
        channel,
        "cartopy",
        "-y",
    ])

    importlib.invalidate_caches()
    if importlib.util.find_spec("cartopy") is None:
        raise RuntimeError(
            "cartopy was installed, but this running kernel still cannot import it. "
            "Restart the kernel and run the setup cell again."
        )

    print("cartopy: installed")


def ensure_openpyxl_setup(
    *,
    min_version: str = "3.1.5",
    install: bool = True,
    channel: str = "conda-forge",
) -> None:
    """Ensure pandas can read .xlsx files in the current Python environment."""
    try:
        import openpyxl
        version = openpyxl.__version__
    except ImportError:
        version = None

    if version is not None and _version_tuple(version) >= _version_tuple(min_version):
        print(f"openpyxl: {version}")
        return

    if not install:
        found = "not installed" if version is None else version
        raise RuntimeError(f"openpyxl>={min_version} is required; found {found}.")

    conda = _conda_executable()
    if conda is None:
        raise RuntimeError(
            f"openpyxl>={min_version} is required and conda was not found. "
            "Install/update openpyxl in this environment."
        )

    env_name = _conda_environment_name()
    _run_setup_command([
        conda,
        "install",
        "-n",
        env_name,
        "-c",
        channel,
        f"openpyxl>={min_version}",
        "-y",
    ])

    try:
        import openpyxl
        version = openpyxl.__version__
    except ImportError:
        version = None

    if version is None or _version_tuple(version) < _version_tuple(min_version):
        raise RuntimeError(
            f"openpyxl was installed/updated, but this running kernel sees {version}. "
            "Restart the kernel and run the setup cell again."
        )

    print(f"openpyxl: {version}")




def adjust_lightness(color, factor=1.0):
    r, g, b = mcolors.to_rgb(color)
    if factor >= 1:
        r = r + (1 - r) * (factor - 1)
        g = g + (1 - g) * (factor - 1)
        b = b + (1 - b) * (factor - 1)
    else:
        r *= factor
        g *= factor
        b *= factor
    return (min(max(r, 0), 1), min(max(g, 0), 1), min(max(b, 0), 1))


def make_shades(base_hex, n):
    base = np.array(to_rgb(base_hex))
    shades = []
    for i in range(n):
        t = i / max(n - 1, 1)
        dark = base * 0.6
        light = base * 0.4 + np.array([1, 1, 1]) * 0.6
        shades.append(dark * (1 - t) + light * t)
    return shades




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

        # (Optional but explicit)
        "font.family": "serif",

        # Force paper fonts via LuaLaTeX
        "pgf.preamble": r"""
\usepackage{fontspec}
\usepackage{unicode-math}

\setmainfont{Latin Modern Roman}
\setsansfont{Latin Modern Sans}
\setmonofont{Latin Modern Mono}
\setmathfont{Latin Modern Math}

\defaultfontfeatures{Ligatures=TeX}
""",
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
    """Bold via LaTeX (works reliably with text.usetex=True)."""
    return r"\textbf{" + tex_safe(s) + "}"


def get_text_color(bg_color) -> str:
    """Pick black/white text based on background luminance."""
    try:
        rgb = mcolors.to_rgb(bg_color)
    except (TypeError, ValueError):
        rgb = tuple(bg_color[:3])
        if max(rgb) > 1:
            rgb = tuple(float(c) / 255.0 for c in rgb)
    r, g, b = rgb
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.50 else "black"


def apply_house_style(ax, *, grid=False):
    """Consistent, clean axes styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)
    if grid:
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.3)




def save_and_close(
    fig=None,
    filename="quick_plot.pdf",
    save=None,
    close=None,
    *,
    legacy_tight=False,
    exact_size=False,
):
    """Save and/or close a matplotlib figure based on global configuration.

    Use ``exact_size=True`` to preserve the figure canvas size instead of
    cropping the PDF bounding box.
    """

    fig = fig or plt.gcf()

    # Resolve effective behavior
    save = SAVE_FIGURES if save is None else save
    close = CLOSE_FIGURES if close is None else close

    outpath = None

    if save:
        outpath = Path(IMG_ROOT) / filename

        if exact_size:
            fig.savefig(outpath)
        elif legacy_tight:
            fig.tight_layout()
            fig.savefig(outpath, bbox_inches="tight")
        else:
            try:
                fig.set_layout_engine("none")
            except Exception:
                pass
            fig.savefig(outpath, bbox_inches="tight")

        print(f"Saved to: {project_relative_path(outpath)}")

    if close:
        plt.close(fig)

    return outpath




def build_origin_color_map(origins):
    from .cost_model.charts import PALETTE_MUTED
    
    origins_sorted = list(origins)
    return {o: PALETTE_MUTED[i % len(PALETTE_MUTED)] for i, o in enumerate(origins_sorted)}






def _strip_latex_wrappers(s: str) -> str:
    s = str(s)
    s = s.replace(r"\textbf{", "")
    s = s.replace(r"\textit{", "")
    s = s.replace(r"\footnotesize", "")
    s = s.replace(r"\scriptsize", "")
    return s.replace("}", "").strip()


def _tex_safe_resilience_scenario(s: str) -> str:
    """Escape scenario text while preserving inline math in its title."""
    value = str(s)
    if " ($" in value and value.endswith("$)"):
        text_part, math_part = value.split(" ($", 1)
        return f"{tex_safe(text_part)} ($" + math_part
    return tex_safe(value)


def _rewrite_resilience_accounting_note(note: str | None) -> str | None:
    """Keep the accounting note current without changing its main explanation."""
    if note is None:
        return None
    old_tail = (
        r"Absolute values use $\mu=0.3$; the central $\mu=0.3$ percentage row is "
        r"supplemented by $\mu=0$, $\mu=0.1$, $\mu=0.2$, $\mu=0.4$, and $\mu=0.5$ "
        r"sensitivity rows, each relative to its corresponding baseline."
    )
    new_tail = (
        r"Absolute values correspond to $\mu=0.3$. Percentage variations are reported for "
        r"$\mu=0$--$0.5$ in increments of 0.1, relative to the unchanged baseline."
    )
    return str(note).replace(old_tail, new_tail)


def _df_to_latex_resilience_accounting_side_by_side(
    df: pd.DataFrame,
    *,
    caption=None,
    label=None,
    note: str | None = None,
) -> str:
    """Render multi-scenario resilience accounting tables side by side."""
    note = _rewrite_resilience_accounting_note(note)
    frame = df.reset_index()
    scenarios = frame["Scenario"].drop_duplicates().tolist()
    preferred_cols = ["Total", "Turbine", "BoP"]
    cols = [c for c in preferred_cols if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    if len(scenarios) == 2 and len(cols) == 3:
        scenario_cols = [(scenario, col) for scenario in scenarios for col in cols]
    else:
        raise ValueError(
            "Side-by-side resilience accounting requires two scenarios and "
            "exactly Total, Turbine, and BoP columns."
        )

    def _fmt_num(v, *, pct=False) -> str:
        return _fmt_resilience_num(v, pct=pct)

    def _fmt_sensitivity(v) -> str:
        if not isinstance(v, (tuple, list)) or len(v) != 2:
            raise ValueError("Markup sensitivity cells must contain (low, high) values.")
        return rf"\shortstack[r]{{{_fmt_num(v[0])}\\{_fmt_num(v[1])}}}"

    def _value(scenario, impact, effect, column):
        sub = frame[
            (frame["Scenario"] == scenario)
            & (frame["Impact"] == impact)
            & (frame["Effects"] == effect)
        ]
        if sub.empty:
            raise KeyError(
                f"Missing resilience accounting row: scenario={scenario!r}, "
                f"impact={impact!r}, effects={effect!r}."
            )
        return sub.iloc[0][column]

    impact_values = frame["Impact"].drop_duplicates().tolist()
    effects_by_impact = {
        impact: frame.loc[frame["Impact"] == impact, "Effects"].drop_duplicates().tolist()
        for impact in impact_values
    }
    numeric_spec = r">{\raggedleft\arraybackslash}p{1.30cm}"
    tabular_spec = "ll|" + numeric_spec * 3 + "|" + numeric_spec * 3

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\setlength{\tabcolsep}{4pt}",
        "",
        rf"\begin{{tabular}}{{{tabular_spec}}}",
        r"\toprule",
        r"\multicolumn{1}{c}{\multirow{2}{*}{\textbf{Impact}}} & "
        r"\multicolumn{1}{c|}{\multirow{2}{*}{\textbf{Effects}}} & "
        rf"\multicolumn{{3}}{{c|}}{{\textbf{{{_tex_safe_resilience_scenario(scenarios[0])}}}}} & "
        rf"\multicolumn{{3}}{{c}}{{\textbf{{{_tex_safe_resilience_scenario(scenarios[1])}}}}} \\" ,
        "& & " + " & ".join(rf"\textbf{{{tex_safe(col)}}}" for _ in scenarios for col in cols) + r" \\",
        r"\midrule",
    ]

    for impact_i, impact in enumerate(impact_values):
        effects = effects_by_impact[impact]
        for effect_i, effect in enumerate(effects):
            effect_plain = _strip_latex_wrappers(effect)
            is_markup_pct = effect_plain in {r"$\mu=0.1$", r"$\mu=0.5$"}
            is_markup_sensitivity = bool(re.fullmatch(r"\$\\mu=0(?:\.\d+)?\$", effect_plain))
            is_central_pct = "%" in effect_plain and r"$\mu=0.3$" in effect_plain
            is_pct = "%" in effect_plain or is_markup_sensitivity
            if is_markup_sensitivity:
                effect_label = effect_plain
            elif "%" in effect_plain:
                pct_name = effect_plain.split(" ($", 1)[0]
                mu_suffix = effect_plain[len(pct_name):]
                effect_label = rf"\textit{{{tex_safe(pct_name)}}}{mu_suffix}"
            else:
                effect_label = tex_safe(effect_plain)
            prefix = []
            if effect_i == 0:
                prefix.append(rf"\multirow{{{len(effects)}}}{{*}}{{{tex_safe(impact)}}}")
            else:
                prefix.append("")
            prefix.append(effect_label)
            values = [
                _fmt_num(_value(scenario, impact, effect, col), pct=is_pct)
                for scenario, col in scenario_cols
            ]
            lines.append("& ".join(prefix + values) + r" \\")
            if effect_plain.startswith(("EU27 base", "Baseline cost", "EU27 scen.", "Scenario cost", "Net effect")):
                lines.append(r"\cline{2-8}")
            if is_central_pct:
                lines.append(r"\arrayrulecolor{gray}\cline{2-8}\arrayrulecolor{black}")
        if impact_i < len(impact_values) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        note_s = str(note).replace(r"\footnotesize", "").strip()
        lines += ["", r"\par\smallskip", r"\begin{minipage}{\textwidth}", r"\scriptsize", note_s, r"\end{minipage}"]
    if caption is not None:
        lines += ["", rf"\caption{{{tex_safe(str(caption))}}}"]
    if label is not None:
        lines.append(rf"\label{{{str(label)}}}")
    lines += [r"\end{table*}", ""]
    return "\n".join(lines)


def _df_to_latex_resilience_accounting(
    df: pd.DataFrame,
    *,
    caption=None,
    label=None,
    index_names: list[str] | None = None,
    note: str | None = None,
) -> str:
    note = _rewrite_resilience_accounting_note(note)
    idx_names = list(index_names or df.index.names)
    has_scenario = idx_names[:3] == ["Scenario", "Impact", "Effects"]
    if (
        has_scenario
        and df.index.get_level_values("Scenario").nunique() == 2
        and len(df.columns) == 3
        and {"Total", "Turbine", "BoP"}.issubset({str(c) for c in df.columns})
    ):
        return _df_to_latex_resilience_accounting_side_by_side(
            df,
            caption=caption,
            label=label,
            note=note,
        )
    index_cols = ["Scenario", "Impact", "Effects"] if has_scenario else ["Impact", "Effects"]

    out = df.copy()
    out.index = out.index.set_names(idx_names)

    preferred_cols = ["Total", "Devex", "Turbine", "BoP", "Inst.", "O&M"]
    cols = [c for c in preferred_cols if c in out.columns] + [c for c in out.columns if c not in preferred_cols]
    out = out.reindex(columns=cols)

    def _fmt_num(v, *, pct=False) -> str:
        return _fmt_resilience_num(v, pct=pct)

    tabular_spec = (
        r"@{\extracolsep{\fill}}lll|r|rrrrr@{}"
        if has_scenario else
        r"@{\extracolsep{\fill}}ll|r|rrrrr@{}"
    )

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\setlength{\tabcolsep}{4pt}",
        "",
        rf"\begin{{tabular*}}{{\textwidth}}{{{tabular_spec}}}",
        r"\toprule",
    ]

    header_parts = [
        rf"\multicolumn{{1}}{{c}}{{\textbf{{{tex_safe(c)}}}}}" for c in index_cols
    ]
    if len(header_parts) >= 2:
        header_parts[-1] = header_parts[-1].replace("{c}", "{c|}", 1)
    data_headers = []
    for i, c in enumerate(cols):
        align = "c|}" if i == 0 else "c}"
        data_headers.append(rf"\multicolumn{{1}}{{{align}{{\textbf{{{tex_safe(c)}}}}}")
    lines.append("\n& ".join(header_parts + data_headers) + r" \\")
    lines.append(r"\midrule")

    frame = out.reset_index()
    scenario_values = frame["Scenario"].drop_duplicates().tolist() if has_scenario else [None]

    for scenario_i, scenario_value in enumerate(scenario_values):
        scenario_frame = frame[frame["Scenario"] == scenario_value] if has_scenario else frame
        scenario_n = len(scenario_frame)
        first_scenario_row = True

        impact_values = scenario_frame["Impact"].drop_duplicates().tolist()
        for impact_i, impact_value in enumerate(impact_values):
            impact_frame = scenario_frame[scenario_frame["Impact"] == impact_value]
            impact_n = len(impact_frame)
            first_impact_row = True

            for _, row in impact_frame.iterrows():
                effect = str(row["Effects"])
                effect_plain = _strip_latex_wrappers(effect)
                is_markup_pct = effect_plain in {r"$\mu=0.1$", r"$\mu=0.5$"}
                is_markup_sensitivity = bool(re.fullmatch(r"\$\\mu=0(?:\.\d+)?\$", effect_plain))
                is_central_pct = "%" in effect_plain and r"$\mu=0.3$" in effect_plain
                is_pct = "%" in effect_plain or is_markup_sensitivity
                if is_markup_sensitivity:
                    effect_label = effect_plain
                elif "%" in effect_plain:
                    pct_name = effect_plain.split(" ($", 1)[0]
                    mu_suffix = effect_plain[len(pct_name):]
                    effect_label = rf"\textit{{{tex_safe(pct_name)}}}{mu_suffix}"
                else:
                    effect_label = tex_safe(effect_plain)

                row_prefix = []
                if has_scenario:
                    row_prefix.append(
                        rf"\multirow[t]{{{scenario_n}}}{{*}}{{{tex_safe(scenario_value)}}}"
                        if first_scenario_row else ""
                    )
                row_prefix.append(
                    rf"\multirow[t]{{{impact_n}}}{{*}}{{{tex_safe(impact_value)}}}"
                    if first_impact_row else ""
                )
                row_prefix.append(effect_label)

                values = [_fmt_num(row[c], pct=is_pct) for c in cols]
                lines.append("& ".join(row_prefix + values) + r" \\")

                first_scenario_row = False
                first_impact_row = False

                if effect_plain.startswith(("EU27 base", "Baseline cost", "EU27 scen.", "Scenario cost", "Net effect")):
                    start_col = 3 if has_scenario else 2
                    end_col = len(index_cols) + len(cols)
                    lines.append(rf"\cline{{{start_col}-{end_col}}}")
                if is_central_pct:
                    start_col = 3 if has_scenario else 2
                    end_col = len(index_cols) + len(cols)
                    lines.append(
                        rf"\arrayrulecolor{{gray}}\cline{{{start_col}-{end_col}}}\arrayrulecolor{{black}}"
                    )

            if impact_i < len(impact_values) - 1:
                lines.append(r"\midrule")
        if scenario_i < len(scenario_values) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular*}",
    ]

    if note:
        note_s = str(note).replace(r"\footnotesize", "").strip()
        lines += [
            "",
            r"\par\smallskip",
            r"\begin{minipage}{\textwidth}",
            r"\scriptsize",
            note_s,
            r"\end{minipage}",
        ]

    if caption is not None:
        lines += ["", rf"\caption{{{tex_safe(str(caption))}}}"]
    if label is not None:
        lines.append(rf"\label{{{str(label)}}}")
    lines.append(r"\end{table*}")
    lines.append("")
    return "\n".join(lines)


def _fmt_report_num(v, *, pct: bool = False) -> str:
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    if isinstance(v, str):
        return v
    x = float(v)
    ax = abs(x)
    if pct:
        body = rf"\num[detect-all]{{{x:.0f}}}"
        return rf"\textit{{{body}}}"
    if ax >= 10:
        return rf"\num{{{x:.0f}}}"
    if ax == 0:
        return r"\num{0}"
    s = f"{x:.1f}".rstrip("0").rstrip(".")
    return rf"\num{{{s}}}"


def _fmt_resilience_num(v, *, pct: bool = False) -> str:
    """Format resilience-accounting numbers without changing other tables."""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    if isinstance(v, str):
        return v
    x = float(v)
    magnitude = abs(x) if x < 0 else x
    if abs(x) >= 10:
        body = f"{magnitude:.0f}"
    else:
        body = f"{magnitude:.1f}".rstrip("0").rstrip(".")
    if body in {"0", "0.0", "-0", "-0.0"}:
        body = "0"
    formatted = rf"\num{{{body}}}"
    if x < 0 and body != "0":
        formatted = rf"({formatted})"
    return rf"\textit{{{formatted}}}" if pct else formatted


def _df_to_latex_impact_region_effect_component(
    df: pd.DataFrame,
    *,
    caption=None,
    label=None,
    note: str | None = None,
) -> str:
    """Paper table: Impact/Region rows with effect and component columns."""
    out = df.copy()
    out.index = out.index.set_names(["Impact", "Region"])

    col_map = {"Installation": "Inst."}
    desired = ["Direct", "Indirect", "Total", "Devex", "Turbine", "BoP", "Installation", "Inst.", "O&M"]
    cols = []
    seen_display = set()
    for col in desired:
        if col in out.columns:
            display = col_map.get(col, col)
            if display not in seen_display:
                cols.append((col, display))
                seen_display.add(display)
    for col in out.columns:
        display = col_map.get(str(col), str(col))
        if display not in seen_display:
            cols.append((col, display))
            seen_display.add(display)
    out = out.reindex(columns=[c for c, _ in cols])

    def _region_label(region: str) -> tuple[str, bool]:
        region = str(region)
        pct = "%" in region
        child = region.replace("%", "").strip() in {"China", "Others"}
        indent = r"\hspace{0.4em}" if child else ""
        if pct:
            display = indent + tex_safe(region.replace("%", "").strip()) + r" \%"
            return rf"\textit{{{display}}}", True
        return indent + tex_safe(region), False

    def _row_lines(prefix: str, region: str, values: list[str]) -> list[str]:
        lines = [prefix] if prefix else []
        lines.append(f"& {region}")
        for value in values:
            lines.append(f"& {value}")
        lines[-1] = lines[-1] + r" \\"
        lines.append("")
        return lines

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\setlength{\tabcolsep}{4pt}",
        "",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll|rr|r|rrrrr@{}}",
        r"\toprule",
        r"\multicolumn{1}{c}{\textbf{Impact}}",
        r"& \multicolumn{1}{c|}{\textbf{Region}}",
        r"& \multicolumn{1}{c}{\textbf{Direct}}",
        r"& \multicolumn{1}{c|}{\textbf{Indirect}}",
        r"& \multicolumn{1}{c|}{\textbf{Total}}",
        r"& \multicolumn{1}{c}{\textbf{Devex}}",
        r"& \multicolumn{1}{c}{\textbf{Turbine}}",
        r"& \multicolumn{1}{c}{\textbf{BoP}}",
        r"& \multicolumn{1}{c}{\textbf{Inst.}}",
        r"& \multicolumn{1}{c}{\textbf{O\&M}} \\",
        r"\midrule",
        "",
    ]

    frame = out.reset_index()
    impact_values = frame["Impact"].drop_duplicates().tolist()
    for impact_i, impact_value in enumerate(impact_values):
        impact_frame = frame[frame["Impact"] == impact_value]
        row_count = len(impact_frame)
        for row_i, (_, row) in enumerate(impact_frame.iterrows()):
            region_label, is_pct = _region_label(row["Region"])
            values = [_fmt_report_num(row[col], pct=is_pct) for col, _ in cols]
            prefix = rf"\multirow[t]{{{row_count}}}{{*}}{{{tex_safe(impact_value)}}}" if row_i == 0 else ""
            lines.extend(_row_lines(prefix, region_label, values))

            region_plain = str(row["Region"])
            next_regions = impact_frame.iloc[row_i + 1 :]["Region"].astype(str).tolist()
            if region_plain == "RoW %":
                lines.append(r"\arrayrulecolor{black!20}")
                lines.append(r"\cline{2-10}")
                lines.append(r"\arrayrulecolor{black}")
                lines.append("")
            elif region_plain == "RoW" or (
                region_plain == "Global"
                and next_regions
                and "%" in next_regions[0]
            ):
                lines.append(r"\cline{2-10}")
                lines.append("")

        if impact_i < len(impact_values) - 1:
            lines.append(r"\midrule")
            lines.append("")

    while lines and lines[-1] == "":
        lines.pop()

    lines += [
        r"\bottomrule",
        r"\end{tabular*}",
    ]

    if note:
        note_s = str(note).replace(r"\footnotesize", "").strip()
        lines += [
            "",
            r"\par\smallskip",
            r"\begin{minipage}{\textwidth}",
            r"\scriptsize",
            note_s,
            r"\end{minipage}",
        ]

    if caption is not None:
        lines += ["", rf"\caption{{{tex_safe(str(caption))}}}"]
    if label is not None:
        lines.append(rf"\label{{{str(label)}}}")
    lines.append(r"\end{table*}")
    lines.append("")
    return "\n".join(lines)


def df_to_latex(
    global_comp_all: pd.DataFrame,
    *,
    caption=None,
    label=None,
    bold_totals: bool = True,
    index_names: list[str] | None = None,
    note: str | None = None,
    remove_category_clines: bool = False,
    keep_total_cline: bool = True,
    font_size: str = "\\footnotesize",
    resize_width: str = "\\textwidth",
) -> str:
    ind_order = ["Output", "GVA", "Income", "Jobs"]
    country_order = ["GB"] + [c for c in CORE_DEPLOYMENT if c != "GB"]
    type_order = ["Direct", "Indirect", "Total"]

    component_order_lvl1 = ["Devex", "BoP", "Turbine", "Installation", "O&M", "Total"]
    component_order_lvl0 = [
        "Devex", "BoP", "Blades", "Drivetrain", "Hub",
        "Nacelle", "Tower", "Installation", "O&M", "Total"
    ]

    df = global_comp_all.copy()

    idx_names_for_format = list(index_names or (df.index.names if isinstance(df.index, pd.MultiIndex) else []))
    if (
        isinstance(df.index, pd.MultiIndex)
        and idx_names_for_format in (["Scenario", "Impact", "Effects"], ["Impact", "Effects"])
        and "Total" in [str(c) for c in df.columns]
    ):
        return _df_to_latex_resilience_accounting(
            df,
            caption=caption,
            label=label,
            index_names=idx_names_for_format,
            note=note,
        )

    if (
        isinstance(df.index, pd.MultiIndex)
        and idx_names_for_format == ["Impact", "Region"]
        and {"Direct", "Indirect", "Total"}.issubset({str(c) for c in df.columns})
    ):
        return _df_to_latex_impact_region_effect_component(
            df,
            caption=caption,
            label=label,
            note=note,
        )

    is_mi_cols = isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels >= 2
    top_vals = df.columns.get_level_values(0).unique().astype(str).tolist() if is_mi_cols else []
    bottom_vals = df.columns.get_level_values(1).unique().astype(str).tolist() if is_mi_cols else []

    grouped_colfmt = None

    if is_mi_cols and any(v in ind_order for v in top_vals):
        ind_use = [i for i in ind_order if i in top_vals] + [i for i in top_vals if i not in set(ind_order)]
        df = df.reindex(ind_use, level=0, axis=1)

        ordered = []
        for ind in ind_use:
            present_c = df.loc[:, ind].columns.astype(str).tolist()
            use_c = [c for c in country_order if c in present_c] + [c for c in present_c if c not in country_order]
            ordered += [(ind, c) for c in use_c if (ind, c) in df.columns]

        leftovers = [c for c in df.columns if c not in ordered]
        df = df.reindex(columns=pd.MultiIndex.from_tuples(ordered + leftovers, names=df.columns.names))

        block_sizes = []
        for ind in ind_use:
            block_sizes.append(sum(df.columns.get_level_values(0) == ind))
        grouped_colfmt = "".join(["|" + ("r" * k) for k in block_sizes if k > 0])

        if isinstance(df.index, pd.MultiIndex):
            comp_candidates = []
            type_candidates = []
            for i in range(df.index.nlevels):
                vals = df.index.get_level_values(i).astype(str)
                comp_candidates.append(vals.isin(component_order_lvl1).sum() + vals.isin(component_order_lvl0).sum())
                type_candidates.append(vals.isin(type_order).sum())

            comp_level = int(np.argmax(comp_candidates)) if max(comp_candidates) > 0 else None
            type_level = int(np.argmax(type_candidates)) if max(type_candidates) > 0 else None

            if comp_level is not None:
                lvl_vals = df.index.get_level_values(comp_level)
                if not (isinstance(lvl_vals, pd.CategoricalIndex) and lvl_vals.ordered):
                    present = lvl_vals.astype(str).unique().tolist()
                    use_base = component_order_lvl0 if any(x in present for x in ["Blades","Drivetrain","Hub","Nacelle","Tower"]) else component_order_lvl1
                    use_comp = [c for c in use_base if c in present] + [c for c in present if c not in use_base]
                    df = df.reindex(use_comp, level=comp_level)

            if type_level is not None:
                lvl_vals = df.index.get_level_values(type_level)
                if not (isinstance(lvl_vals, pd.CategoricalIndex) and lvl_vals.ordered):
                    df = df.reindex(type_order, level=type_level)

            # apply tex-safe labels; bold "Total" only on component level
            for i in range(df.index.nlevels):
                if comp_level is not None and i == comp_level:
                    df = df.rename(
                        index=lambda x: (tex_bold("Total") if str(x).strip() == "Total" else tex_safe(str(x))),
                        level=i,
                    )
                else:
                    df = df.rename(index=lambda x: tex_safe(str(x)), level=i)

    elif is_mi_cols and set(top_vals).issubset(set(type_order)):
        use_t = [t for t in type_order if t in top_vals] + [t for t in top_vals if t not in set(type_order)]
        df = df.reindex(use_t, level=0, axis=1)

        present_c = df.columns.get_level_values(1).astype(str).unique().tolist()
        use_c = [c for c in country_order if c in present_c] + [c for c in present_c if c not in country_order]

        ordered = [(t, c) for t in use_t for c in use_c if (t, c) in df.columns]
        leftovers = [c for c in df.columns if c not in ordered]
        df = df.reindex(columns=pd.MultiIndex.from_tuples(ordered + leftovers, names=df.columns.names))

        block_sizes = [sum(df.columns.get_level_values(0) == t) for t in use_t]
        grouped_colfmt = "".join(["|" + ("r" * k) for k in block_sizes if k > 0])

        if isinstance(df.index, pd.MultiIndex):
            for i in range(df.index.nlevels):
                df = df.rename(index=lambda x: tex_safe(str(x)), level=i)
    elif is_mi_cols and set(bottom_vals).issubset({"EU27", "Global", "EU27%"}):
        # component-as-columns layout: keep component blocks with vertical separators
        comp_vals = df.columns.get_level_values(0).unique().tolist()
        ordered = []
        for comp in comp_vals:
            present_c = df.loc[:, comp].columns.astype(str).tolist()
            use_c = [c for c in ["Global", "EU27", "EU27%"] if c in present_c] + [c for c in present_c if c not in {"Global","EU27","EU27%"}]
            ordered += [(comp, c) for c in use_c if (comp, c) in df.columns]
        leftovers = [c for c in df.columns if c not in ordered]
        df = df.reindex(columns=pd.MultiIndex.from_tuples(ordered + leftovers, names=df.columns.names))

        block_sizes = [sum(df.columns.get_level_values(0) == comp) for comp in df.columns.get_level_values(0).unique()]
        grouped_colfmt = "".join(["|" + ("r" * k) for k in block_sizes if k > 0])

        if isinstance(df.index, pd.MultiIndex):
            for i in range(df.index.nlevels):
                df = df.rename(index=lambda x: tex_safe(str(x)), level=i)
        else:
            df = df.rename(index=lambda x: tex_safe(str(x)))

    else:
        if isinstance(df.index, pd.MultiIndex):
            for i in range(df.index.nlevels):
                df = df.rename(index=lambda x: tex_safe(str(x)), level=i)
        else:
            df = df.rename(index=lambda x: tex_safe(str(x)))

    # ----------------------------
    # 5) Percent columns formatting (EU27%)
    # ----------------------------
    def _num_fmt(v) -> str:
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        if isinstance(v, str):
            if v.startswith(r"\num{"):
                return v
        try:
            x = float(v)
        except Exception:
            return str(v)
        ax = abs(x)
        if 0 < ax < 10:
            return f"\\num{{{x:.1f}}}"
        r0 = round(x, 0)
        if r0 == 0:
            return f"\\num{{{round(x, 1):.1f}}}"
        return f"\\num{{{int(r0)}}}"

    def _format_pct_cell(v) -> str:
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        if isinstance(v, str):
            if "%" in v and r"\%" not in v:
                if v.startswith(r"\num{"):
                    return v.replace("%", r"\%")
                return v.replace("%", r"\%")
            return v
        try:
            x = float(v)
        except Exception:
            return str(v)
        return _num_fmt(x) + r"\%"

    if isinstance(df.columns, pd.MultiIndex):
        for col in df.columns:
            if len(col) >= 2 and str(col[1]).strip() == "EU27%":
                df[col] = df[col].map(_format_pct_cell)
    else:
        for c in df.columns:
            if str(c).strip() == "EU27%":
                df[c] = df[c].map(_format_pct_cell)

    # Presentation-only compatibility for Region/Impact component tables:
    # keep the old flat component-column layout (Total, Devex, BoP, ...)
    # even if the input arrives with a one-metric MultiIndex.
    old_component_table_layout = False
    if isinstance(df.index, pd.MultiIndex) and index_names is not None:
        idx_names_plain = [str(x).strip().casefold() for x in index_names]
        old_component_table_layout = idx_names_plain[:2] in (["region", "impact"], ["impact", "region"])

    if old_component_table_layout and isinstance(df.columns, pd.MultiIndex):
        comp_names = ["Total", "Devex", "BoP", "Turbine", "Installation", "O&M"]
        flat_cols = []
        flat_data = {}
        for comp in comp_names:
            matches = [c for c in df.columns if str(c[0]) == comp]
            if not matches:
                matches = [c for c in df.columns if str(c[-1]) == comp]
            if matches:
                col = matches[0]
                flat_cols.append(comp)
                flat_data[comp] = df[col]
        for col in df.columns:
            label = str(col[0])
            if label in flat_data:
                continue
            if str(col[-1]) in flat_data:
                continue
            flat_cols.append(label)
            flat_data[label] = df[col]
        if flat_cols:
            df = pd.DataFrame(flat_data, index=df.index)[flat_cols]

    if old_component_table_layout and not isinstance(df.columns, pd.MultiIndex):
        comp_names = ["Total", "Devex", "BoP", "Turbine", "Installation", "O&M"]
        effect_names = ["Direct", "Indirect", "Total"]
        plain_cols = [str(c) for c in df.columns]
        order_names = effect_names if any(c in plain_cols for c in ["Direct", "Indirect"]) else comp_names
        ordered = [c for c in order_names if c in df.columns]
        ordered += [c for c in df.columns if c not in ordered]
        df = df.reindex(columns=ordered)

    # ----------------------------
    # 6) Header labels (bold + centered)
    # ----------------------------
    def _display_col_label(label) -> str:
        s = str(label)
        return {"Installation": "Inst."}.get(s, s)

    def _center_header(s: str) -> str:
        return r"\multicolumn{1}{c}{" + s + "}"

    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        for a, b in df.columns.to_list():
            cols.append((tex_bold(_display_col_label(a)), _center_header(tex_bold(_display_col_label(b)))))
        df.columns = pd.MultiIndex.from_tuples(cols)
        df.columns = df.columns.set_names([None] * df.columns.nlevels)
    else:
        df.columns = [_center_header(tex_bold(_display_col_label(c))) for c in df.columns]

    # Index level names (Category/Type) if detected; otherwise preserve existing names
    if isinstance(df.index, pd.MultiIndex):
        if index_names is not None:
            actual_names_plain = [
                str(n).strip().casefold() if n is not None else ""
                for n in df.index.names
            ]
            if old_component_table_layout and actual_names_plain[:2] in (
                ["impact", "region"],
                ["region", "impact"],
            ):
                index_names = list(df.index.names)
            use_names = []
            for i in range(df.index.nlevels):
                n = index_names[i] if i < len(index_names) else None
                use_names.append(tex_bold(str(n)) if (n is not None and str(n).strip() != "") else None)
            df.index = df.index.set_names(use_names)
        else:
            names = [None] * df.index.nlevels
            if 'comp_level' in locals() and comp_level is not None:
                names[comp_level] = tex_bold("Category")
            if 'type_level' in locals() and type_level is not None:
                names[type_level] = tex_bold("Type")
            if any(n is not None for n in names):
                df.index = df.index.set_names(names)
            else:
                existing = [
                    tex_bold(str(n)) if (n is not None and str(n).strip() != "") else None
                    for n in df.index.names
                ]
                df.index = df.index.set_names(existing)
    else:
        df.index = df.index.set_names([tex_bold("Category")])

    def _format_cell(x) -> str:
        return _num_fmt(x)

    # ----------------------------
    # 7) Bold total rows
    # ----------------------------
    df_out = df.copy().astype(object)

    def _is_pct_col(c) -> bool:
        try:
            if isinstance(c, tuple) and len(c) >= 2:
                return ("EU27" in str(c[1])) and ("%" in str(c[1]))
            return ("EU27" in str(c)) and ("%" in str(c))
        except Exception:
            return False

    # format all EU27% columns with escaped percent
    for c in df_out.columns:
        if _is_pct_col(c):
            df_out[c] = df_out[c].apply(_format_pct_cell)

    # format EU27% rows (if present in index)
    pct_row_mask = None
    if isinstance(df_out.index, pd.MultiIndex):
        try:
            def _norm_pct_label(x: str) -> str:
                return str(x).replace(r"\%", "%")
            masks = []
            for lvl_i in range(df_out.index.nlevels):
                vals = df_out.index.get_level_values(lvl_i).astype(str)
                masks.append(vals.map(_norm_pct_label).str.contains("%", regex=False))
            mask = masks[0]
            for m in masks[1:]:
                mask = mask | m
            pct_row_mask = mask
        except Exception:
            pass
    if isinstance(df_out.index, pd.MultiIndex) and ('comp_level' in locals()) and comp_level is not None:
        total_mask = (df_out.index.get_level_values(comp_level).astype(str).str.lower() == "total")
    elif isinstance(df_out.index, pd.MultiIndex):
        idx_vals = df_out.index.to_frame(index=False).astype(str)
        total_mask = idx_vals.apply(lambda s: s.str.lower().eq("total")).any(axis=1).values
    else:
        idx_lower = df_out.index.astype(str).str.lower()
        try:
            total_mask = idx_lower.eq("total").values
        except AttributeError:
            total_mask = np.asarray(idx_lower == "total")

    if bold_totals and total_mask.any():
        for c in df_out.columns:
            if _is_pct_col(c):
                df_out.loc[total_mask, c] = df_out.loc[total_mask, c].apply(
                    lambda v: (r"\textbf{" + str(v) + "}") if pd.notnull(v) else v
                )
            else:
                df_out.loc[total_mask, c] = df_out.loc[total_mask, c].apply(
                    lambda v: tex_bold(_format_cell(v)) if pd.notnull(v) else v
                )

    if pct_row_mask is not None and pct_row_mask.any():
        def _italic_cell(v):
            if pd.isna(v):
                return v
            s = str(v)
            if not isinstance(v, str):
                s = _format_cell(v)
            if s.startswith(r"\textit{"):
                return s
            num_match = re.fullmatch(r"\\num\{(.+)\}", s)
            if num_match:
                return r"\textit{\num[detect-all]{" + num_match.group(1) + "}}"
            return s if s.startswith(r"\textit{") else (r"\textit{" + s + "}")

        for c in df_out.columns:
            df_out.loc[pct_row_mask, c] = df_out.loc[pct_row_mask, c].apply(_italic_cell)

        if isinstance(df_out.index, pd.MultiIndex):
            arrays = []
            for lvl_i in range(df_out.index.nlevels):
                vals = df_out.index.get_level_values(lvl_i).astype(str).tolist()
                vals = [
                    (
                        v
                        if (
                            not pct_row_mask[i]
                            or "%" not in v.replace(r"\%", "%")
                            or v.startswith(r"\textit{")
                        )
                        else r"\textit{" + v + "}"
                    )
                    for i, v in enumerate(vals)
                ]
                arrays.append(vals)
            df_out.index = pd.MultiIndex.from_arrays(arrays, names=df_out.index.names)

    if old_component_table_layout and isinstance(df_out.index, pd.MultiIndex):
        try:
            idx_vals = df_out.index.to_frame(index=False).astype(str)
            employment_mask = idx_vals.apply(
                lambda s: s.str.replace(r"\textit{", "", regex=False)
                .str.replace("}", "", regex=False)
                .eq("Employment")
            ).any(axis=1).values
            if pct_row_mask is not None:
                employment_mask = employment_mask & ~np.asarray(pct_row_mask)

            def _format_thousand_jobs_cell(v):
                if pd.isna(v):
                    return v
                if isinstance(v, str):
                    return v
                try:
                    x = float(v)
                except Exception:
                    return str(v)
                ax = abs(x)
                if ax >= 10:
                    return rf"\num{{{int(round(x))}}}"
                if ax >= 0.1:
                    return rf"\num{{{x:.1f}}}"
                if ax >= 0.01:
                    s = f"{x:.2f}".rstrip("0").rstrip(".")
                    return rf"\num{{{s}}}"
                if ax > 0:
                    s = f"{x:.3f}".rstrip("0").rstrip(".")
                    return rf"\num{{{s}}}"
                return r"\num{0}"

            if employment_mask.any():
                for c in df_out.columns:
                    df_out.loc[employment_mask, c] = df_out.loc[employment_mask, c].apply(
                        _format_thousand_jobs_cell
                    )
        except Exception:
            pass

    n_idx = df_out.index.nlevels if isinstance(df_out.index, pd.MultiIndex) else 1
    base_colfmt = ("l" * n_idx) + ("r" * df_out.shape[1])

    # If single-level columns include "Total", add vertical bars around it
    if not isinstance(df_out.columns, pd.MultiIndex):
        def _plain_col(s: str) -> str:
            s = str(s)
            # strip common LaTeX wrappers
            s = re.sub(r"\\multicolumn\{1\}\{[clr]\}\{", "", s)
            s = s.replace("\\textbf{", "")
            s = s.replace("}", "")
            s = s.replace("\\", "")
            return s.strip()

        cols = [_plain_col(c) for c in df_out.columns]
        if "Total" in cols:
            t_idx = cols.index("Total")
            bars_before = {t_idx}
            if t_idx + 1 < len(cols):
                bars_before.add(t_idx + 1)
            colfmt_cols = ""
            for i in range(len(cols)):
                colfmt_cols += ("|" if i in bars_before else "") + "r"
            base_colfmt = ("l" * n_idx) + colfmt_cols

    tabular = df_out.to_latex(
        escape=False,
        index=True,
        multicolumn=True,
        multicolumn_format="c",
        multirow=True,
        na_rep="",
        float_format=_format_cell,
        index_names=False,
        column_format=base_colfmt,
        caption=None,
        label=None,
    )

    # Put index names in the same header row as column labels
    lines = tabular.splitlines()
    header_row_idx = None
    idx_names = [n if n is not None else "" for n in (df_out.index.names or [])]
    idx_header = " & ".join([tex_bold(str(n)) if str(n).strip() != "" else "" for n in idx_names])
    for i, line in enumerate(lines):
        if "\\multicolumn" in line:
            if i + 1 < len(lines) and lines[i + 1].lstrip().startswith("&"):
                prefix = " & " * n_idx
                if idx_header.strip():
                    lines[i + 1] = lines[i + 1].replace(
                        prefix, f"{idx_header} & ", 1
                    )
                header_row_idx = i + 1
            break
    if header_row_idx is None:
        for i, line in enumerate(lines):
            if line.lstrip().startswith("&"):
                prefix = " & " * n_idx
                if idx_header.strip():
                    lines[i] = lines[i].replace(prefix, f"{idx_header} & ", 1)
                    header_row_idx = i
                break

    # Remove the separate index-names row (if any)
    if header_row_idx is not None:
        for j in range(header_row_idx + 1, len(lines)):
            ln = lines[j]
            if idx_header and idx_header in ln:
                # skip only if the rest of the line is empty headers
                tail = ln.replace(idx_header, "")
                if all(tok.strip() in {"", "&", "\\\\"} for tok in tail.split()):
                    lines.pop(j)
                    break

    # Optionally remove category-level clines (keep only before Total block)
    if remove_category_clines:
        new_lines = []
        for i, ln in enumerate(lines):
            if ln.strip().startswith("\\cline"):
                keep = False
                if keep_total_cline:
                    # look ahead to next data row
                    for j in range(i + 1, len(lines)):
                        nxt = lines[j].strip()
                        if not nxt or nxt.startswith("\\cline"):
                            continue
                        if "Total" in nxt:
                            keep = True
                        break
                if keep:
                    new_lines.append(ln)
                continue
            new_lines.append(ln)
        lines = new_lines

    if old_component_table_layout:
        cline = rf"\cline{{{2 if n_idx >= 2 else 1}-{n_idx + df_out.shape[1]}}}"
        new_lines = []
        for i, ln in enumerate(lines):
            new_lines.append(ln)
            normalized = ln.replace(r"\%", "%")
            next_data = ""
            for nxt in lines[i + 1:]:
                stripped = nxt.strip()
                if stripped:
                    next_data = stripped
                    break
            global_has_region_below = (
                " & Global & " in normalized
                and next_data
                and not next_data.startswith((r"\cline", r"\bottomrule", r"\end{tabular}"))
            )
            if (
                global_has_region_below
                or " & RoW & " in normalized
            ):
                new_lines.append(cline)
        lines = new_lines

    effects_table_layout = False
    try:
        effects_table_layout = any("Effects" in str(n) for n in (df_out.index.names or []))
    except Exception:
        effects_table_layout = False
    if not effects_table_layout:
        joined_lines = "\n".join(lines)
        effects_table_layout = (
            "EU27 base" in joined_lines
            and "Gross effect" in joined_lines
            and (
                "Up. trade effects" in joined_lines
                or "Up trade effects" in joined_lines
                or "Upstream trade effects" in joined_lines
            )
        )

    if effects_table_layout:
        cline = rf"\cline{{{2 if n_idx >= 2 else 1}-{n_idx + df_out.shape[1]}}}"
        break_after = (
            "EU27 base",
            "Up. trade effects",
            "Up trade effects",
            "Upstream trade effects",
            "Net effect",
        )
        new_lines = []
        for i, ln in enumerate(lines):
            new_lines.append(ln)
            stripped = ln.strip()
            if not stripped or stripped.startswith((r"\cline", r"\midrule", r"\toprule", r"\bottomrule")):
                continue
            normalized = stripped.replace(r"\\%", r"\%")
            if not any(pat in normalized for pat in break_after):
                continue
            next_data = ""
            for nxt in lines[i + 1:]:
                nxt_s = nxt.strip()
                if nxt_s:
                    next_data = nxt_s
                    break
            if next_data and not next_data.startswith((r"\cline", r"\bottomrule", r"\end{tabular}")):
                new_lines.append(cline)
        lines = new_lines

    tabular = "\n".join(lines)

    if grouped_colfmt is not None:
        tabular = tabular.replace(
            f"\\begin{{tabular}}{{{base_colfmt}}}",
            f"\\begin{{tabular}}{{{('l'*n_idx)}{grouped_colfmt}}}"
        )

    cap = "" if caption is None else tex_safe(str(caption))
    # Do not escape labels (LaTeX labels should be plain identifiers)
    lab = "" if label is None else str(label)

    note_block = "" if note is None else (str(note).rstrip() + "\n")
    return (
        "\\begin{table}[H]\n"
        "\\centering\n"
        f"{font_size}\n"
        f"\\resizebox{{{resize_width}}}{{!}}{{%\n"
        + tabular +
        "}\n"
        "\\par\n"
        + note_block
        + (f"\\caption{{{cap}}}\n" if caption is not None else "")
        + (f"\\label{{{lab}}}\n" if label is not None else "")
        + "\\end{table}\n"
    )


