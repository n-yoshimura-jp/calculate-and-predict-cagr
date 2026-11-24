"""分析・予測結果のグラフ（matplotlib）。

ノートブックでは最初に set_style() を呼んでから使う。各関数は Figure を返す。
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure

# 配色: 銘柄の識別用6色（この順に使う）と、プラス/マイナスの2色
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
POS_COLOR, NEG_COLOR = "#2a78d6", "#e34948"
SURFACE, INK, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"

# ヒートマップ用: 赤(マイナス) ← 灰(0) → 青(プラス) の発散カラーマップ
DIVERGING = LinearSegmentedColormap.from_list(
    "red_gray_blue",
    ["#8f2322", "#e34948", "#f4a9a8", "#f0efec", "#9ec5f4", "#2a78d6", "#104281"],
)

JAPANESE_FONTS = ["Hiragino Sans", "Yu Gothic", "Meiryo", "Noto Sans CJK JP", "IPAexGothic"]

PERCENT = mticker.PercentFormatter(xmax=1, decimals=0)


def set_style() -> None:
    """日本語フォントとグラフの共通スタイルを設定する。"""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [name for name in JAPANESE_FONTS if name in installed]
    if fonts:
        plt.rcParams["font.family"] = fonts[0]

    plt.rcParams.update(
        {
            "figure.figsize": (11, 5),
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.titlesize": 13,
            "axes.titlelocation": "left",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 11,
            "legend.frameon": False,
        }
    )


def _style_axes(ax, grid_axis: str = "y") -> None:
    """枠線と目盛りを削り、グリッドを薄く残す。"""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)


def _use_plain_log_axis(ax) -> None:
    """対数軸の目盛りを 10^2 ではなく実数で表示する。"""
    ax.set_yscale("log")
    # 1桁の範囲しかない場合も目盛りが出るよう、1・2・5 の位置に目盛りを置く
    ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def plot_price_history(close: pd.Series, symbol: str, prediction: dict | None = None) -> Figure:
    """終値の推移を対数軸で描く。prediction を渡すと1年後の予測価格を破線で重ねる。"""
    fig, ax = plt.subplots()
    ax.plot(
        close.index,
        close.values,
        color=SERIES_COLORS[0],
        linewidth=2,
        label=f"終値（直近 {close.iloc[-1]:,.2f}）",
    )

    if prediction is not None:
        target_date = close.index[-1] + pd.DateOffset(years=1)
        ax.plot(
            [close.index[-1], target_date],
            [prediction["last_close"], prediction["predicted_price"]],
            color=SERIES_COLORS[1],
            linewidth=2,
            linestyle="--",
            marker="o",
            markevery=[1],
            label=(
                f"1年後の予測 {prediction['predicted_price']:,.2f}"
                f"（{prediction['predicted_growth']:+.1%}）"
            ),
        )

    _use_plain_log_axis(ax)
    ax.set_title(f"{symbol} 終値の推移（調整後・対数軸）")
    ax.set_ylabel("終値 (USD)")
    ax.legend(loc="upper left")
    _style_axes(ax)
    fig.tight_layout()
    return fig


def plot_annual_returns(table: pd.DataFrame, symbol: str, years: int = 20) -> Figure:
    """前年比騰落率を棒グラフで描く（青=プラス / 赤=マイナス、破線は期間の CAGR）。"""
    returns = table["year_on_year_return"].dropna().tail(years)

    fig, ax = plt.subplots()
    ax.bar(
        returns.index.astype(str),
        returns.values,
        width=0.75,
        color=[POS_COLOR if v >= 0 else NEG_COLOR for v in returns.values],
    )
    ax.axhline(0, color=BASELINE, linewidth=1)

    # 表示期間の平均成長率（幾何平均 = CAGR）を参照線として重ねる
    cagr = (1 + returns).prod() ** (1 / len(returns)) - 1
    ax.axhline(cagr, color=MUTED, linewidth=1.5, linestyle="--")
    ax.annotate(
        f"CAGR {cagr:.1%}",
        xy=(1.0, cagr),
        xycoords=("axes fraction", "data"),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        color=MUTED,
        fontsize=10,
    )

    ax.set_title(f"{symbol} 前年比騰落率（年末→年末、直近{len(returns)}年）")
    ax.set_ylabel("騰落率")
    ax.yaxis.set_major_formatter(PERCENT)
    ax.tick_params(axis="x", labelsize=9)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def plot_cumulative_growth(closes: dict[str, pd.Series]) -> Figure:
    """全銘柄がそろう日を100として、累積成長を対数軸で比較する（最大6銘柄）。"""
    if len(closes) > len(SERIES_COLORS):
        raise ValueError(f"比較できるのは{len(SERIES_COLORS)}銘柄までです")

    start = max(close.index[0] for close in closes.values())

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for color, (symbol, close) in zip(SERIES_COLORS, closes.items()):
        index = close[close.index >= start]
        index = index / index.iloc[0] * 100
        ax.plot(index.index, index.values, color=color, linewidth=2, label=symbol)
        ax.annotate(  # 凡例に加えて線の右端にも銘柄名を置く
            symbol,
            xy=(index.index[-1], index.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontweight="bold",
            fontsize=10,
        )

    _use_plain_log_axis(ax)
    ax.set_title(f"累積成長の比較（{start.date()} = 100、対数軸）")
    ax.set_ylabel("指数（起点 = 100）")
    ax.legend(ncols=len(closes), loc="upper left", fontsize=10)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def plot_yoy_heatmap(yoy: pd.DataFrame, years: int = 20) -> Figure:
    """年 × 銘柄の前年比騰落率をヒートマップで描く（数値は % で併記）。"""
    df = yoy.dropna(how="all").tail(years)

    # 極端な年に色域を持っていかれないよう、絶対値の90パーセンタイルで色を頭打ちにする
    lim = float(np.nanpercentile(np.abs(df.values), 90)) or 1.0
    cmap = DIVERGING.with_extremes(bad="#f5f5f2")

    fig, ax = plt.subplots(figsize=(1.3 * len(df.columns) + 2.5, 0.42 * len(df) + 1.8))
    image = ax.imshow(
        df.values, cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), aspect="auto"
    )

    ax.set_xticks(range(len(df.columns)), df.columns, fontweight="bold", color=INK)
    ax.set_yticks(range(len(df.index)), df.index, fontsize=10)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0)
    for side in ax.spines:
        ax.spines[side].set_visible(False)

    # セル間に隙間を入れて、マス目を読みやすくする
    ax.set_xticks(np.arange(len(df.columns) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(df.index) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            value = df.values[i, j]
            if np.isnan(value):
                continue
            ax.text(
                j,
                i,
                f"{value * 100:.0f}",
                ha="center",
                va="center",
                fontsize=9,
                color="#ffffff" if abs(value) >= lim * 0.6 else INK,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, format=PERCENT)
    cbar.set_label("前年比騰落率", color=MUTED, fontsize=10)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelcolor=MUTED)

    ax.set_title("前年比騰落率の比較（年 × 銘柄、単位 %）", pad=28)
    fig.tight_layout()
    return fig


def plot_model_comparison(metrics: dict) -> Figure:
    """test 期間の MAE を、baseline（平均リターンを常に予測）との差で比べる。"""
    mae = {name: result["mae"] for name, result in metrics["models"].items()}
    names = [name for name in mae if name != "baseline"]
    # プラス = baseline より誤差が小さい（良い）
    improvement = [mae["baseline"] - mae[name] for name in names]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.barh(
        names,
        improvement,
        height=0.6,
        color=[POS_COLOR if v >= 0 else NEG_COLOR for v in improvement],
    )
    ax.axvline(0, color=BASELINE, linewidth=1)
    for y, value in enumerate(improvement):
        ax.annotate(
            f"{value:+.4f}",
            xy=(value, y),
            xytext=(6 if value >= 0 else -6, 0),
            textcoords="offset points",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=10,
        )
    ax.invert_yaxis()
    ax.margins(x=0.3)
    ax.set_title(
        f"baseline からの MAE の改善（プラス = 良い、baseline の MAE = {mae['baseline']:.4f}）"
    )
    ax.set_xlabel("baseline の MAE − モデルの MAE（対数リターン）")
    _style_axes(ax, grid_axis="x")
    fig.tight_layout()
    return fig


def plot_feature_importance(importance: pd.Series, title: str) -> Figure:
    """特徴量の重要度を横棒グラフで描く。"""
    importance = importance.sort_values()

    fig, ax = plt.subplots(figsize=(9, 0.4 * len(importance) + 1.2))
    ax.barh(importance.index, importance.values, height=0.6, color=SERIES_COLORS[0])
    ax.set_title(title)
    ax.set_xlabel("重要度")
    _style_axes(ax, grid_axis="x")
    fig.tight_layout()
    return fig


def plot_prediction_vs_actual(test_result: pd.DataFrame, model_name: str) -> Figure:
    """test 期間について、1年後の成長率の予測と実績を日ごとの全銘柄平均で比べる。"""
    growth = np.exp(test_result[["target", model_name]]) - 1
    daily = growth.groupby(level=0).mean()

    fig, ax = plt.subplots()
    lines = [("target", "実績", SERIES_COLORS[0]), (model_name, f"予測（{model_name}）", SERIES_COLORS[1])]
    for column, label, color in lines:
        ax.plot(daily.index, daily[column], color=color, linewidth=2, label=label)
        ax.annotate(
            label,
            xy=(daily.index[-1], daily[column].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontweight="bold",
            fontsize=10,
        )
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_title("1年後の成長率: 予測と実績（test 期間、全銘柄の平均）")
    ax.set_xlabel("予測した日")
    ax.set_ylabel("1年後までの成長率")
    ax.yaxis.set_major_formatter(PERCENT)
    ax.legend(loc="upper left")
    _style_axes(ax)
    fig.tight_layout()
    return fig


def plot_growth_comparison(df: pd.DataFrame) -> Figure:
    """銘柄ごとに、過去の CAGR と今後1年の予測成長率を並べて比べる。

    df の列: symbol, cagr_5y, predicted_growth
    """
    x = np.arange(len(df))
    width = 0.38

    fig, ax = plt.subplots()
    bars = [
        ("cagr_5y", "過去5年の CAGR", SERIES_COLORS[0], -width / 2),
        ("predicted_growth", "今後1年の予測成長率", SERIES_COLORS[1], width / 2),
    ]
    for column, label, color, offset in bars:
        ax.bar(x + offset, df[column], width=width - 0.04, color=color, label=label)
    ax.axhline(0, color=BASELINE, linewidth=1)

    ax.set_xticks(x, df["symbol"], rotation=45, ha="right")
    ax.set_title("過去の CAGR と今後1年の予測成長率")
    ax.set_ylabel("年率")
    ax.yaxis.set_major_formatter(PERCENT)
    ax.legend(loc="upper right")
    _style_axes(ax)
    fig.tight_layout()
    return fig
