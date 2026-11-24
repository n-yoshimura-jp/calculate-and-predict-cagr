# NASDAQ-100 の CAGR 分析と1年後の株価予測

NASDAQ-100 の構成銘柄について、**年平均成長率（CAGR）と年ごとの騰落率の分析**、**約1年後の株価・成長率の予測**、**可視化**を行うプロジェクトです。
予測には scikit-learn / LightGBM を使い、分析・予測・可視化は1本のノートブックで行います。同じモデルを FastAPI からも使えます。

> 教育・研究目的のプロジェクトです。予測結果は投資助言ではありません（[免責事項](#免責事項disclaimer)）。

## クイックスタート

```bash
uv sync                                          # 依存パッケージをインストール
uv run pytest                                    # テスト（オフラインで実行できる）
uv run cagr-train                                # モデルを学習して models/ に保存
uv run fastapi dev src/cagr_predictor/api.py     # API を起動 → http://127.0.0.1:8000/docs
```

分析・可視化は `notebooks/analysis.ipynb` で行います（[使い方](#使い方)）。

## ディレクトリ構成

```
├── data/nasdaq100.csv        # 構成銘柄のスナップショット
├── models/                   # 学習済みモデルと評価結果（git 管理外）
├── notebooks/analysis.ipynb  # 分析・予測・可視化
├── src/cagr_predictor/
│   ├── data.py               # 銘柄リストの読み込み、yfinance で株価を取得
│   ├── cagr.py               # CAGR と年次騰落率の計算
│   ├── features.py           # 特徴量と目的変数
│   ├── train.py              # 学習・評価・保存（uv run cagr-train）
│   ├── predict.py            # モデルの読み込みと1年後の価格予測
│   ├── plots.py              # グラフ（matplotlib）
│   └── api.py                # FastAPI
└── tests/
```

## セットアップ

[uv](https://docs.astral.sh/uv/) を使います。Python 3.14 以上が必要です（インストールされていなければ `uv sync` が自動でダウンロードします）。

```bash
uv sync
```

- macOS で LightGBM を使うには OpenMP が必要です（`brew install libomp`）。
- 別のプロジェクトの仮想環境が有効なままだと、uv が `VIRTUAL_ENV=... does not match the project environment path` と警告します。`deactivate` で解除してください。

## 使い方

### 1. ノートブックで分析・予測・可視化する

`notebooks/analysis.ipynb` を開き、カーネルにこのプロジェクトの `.venv` を選んで上から実行します（インターネット接続が必要です）。
先頭のセルの `symbol`（1銘柄）と `tickers`（比較する最大6銘柄）を変えると、分析する銘柄を変えられます。

| セクション | 内容 |
|---|---|
| 1. データ取得 | 全101銘柄の全期間の日足（調整後終値）を取得 |
| 2. CAGR の分析 | 全期間と直近5年の CAGR。CAGR 20% 以上かつ上場16年以上の銘柄を抽出 |
| 3. 個別銘柄の年次成長率 | 年内騰落率・前年比騰落率の表、株価の推移、前年比騰落率の棒グラフ |
| 4. 複数銘柄の比較 | 前年比騰落率の表、累積成長の比較、年 × 銘柄のヒートマップ |
| 5. 予測モデルの学習と評価 | 4モデルを学習・評価し `models/` に保存。baseline との比較、特徴量の重要度、予測と実績の比較 |
| 6. 1年後の予測 | 1年後の予測価格のグラフ、過去5年の CAGR と予測成長率の比較 |
| 7. Excel に保存 | `data/analysis.xlsx` に4シートで保存（git 管理外） |

ノートブックの中でモデルを学習するので、先に `uv run cagr-train` を実行しておく必要はありません。

JupyterLab で開く場合（プロジェクトの依存には追加せず、一時的に使います）:

```bash
uv run --with jupyterlab jupyter lab notebooks/analysis.ipynb
```

ターミナルから最後まで実行する場合（実行結果はノートブックに上書き保存されます）:

```bash
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
```

### 2. モデルだけを学習する

```bash
uv run cagr-train
```

ノートブックの5.と同じ学習を行い、`models/` に保存します（インターネット接続が必要です。20秒〜1分ほどかかります）。

```
101 銘柄の株価（全期間）をダウンロードしています...
101 銘柄のデータを取得しました
train: 1963-01-02 〜 2012-04-19 (391,060 行)
test : 2013-04-24 〜 2025-11-20 (272,350 行)
      baseline: {'mae': 0.2508..., 'rmse': 0.3630..., 'direction_accuracy': 0.7508...}
         ...
モデルと評価結果を .../models に保存しました
```

### 3. API を起動する

先にノートブックか `uv run cagr-train` でモデルを学習しておきます。

```bash
uv run fastapi dev src/cagr_predictor/api.py
```

http://127.0.0.1:8000/docs で Swagger UI が開きます。

| Method | Path | 内容 |
|---|---|---|
| GET | `/health` | 稼働確認と、読み込んだモデルの一覧 |
| GET | `/symbols` | NASDAQ-100 の銘柄一覧 |
| GET | `/cagr/{symbol}?last_years=5` | CAGR（`last_years` を省略すると全期間） |
| GET | `/annual-growth/{symbol}` | 年ごとの年内騰落率・前年比騰落率 |
| GET | `/predict/{symbol}?model=lightgbm` | 1年後（252営業日後）の予測価格と予測成長率（`model`: baseline / ridge / random_forest / lightgbm） |
| GET | `/models/metrics` | test 期間での評価指標 |

```bash
curl "http://127.0.0.1:8000/predict/AVGO?model=lightgbm"
```

```json
{
  "symbol": "AVGO",
  "model": "lightgbm",
  "as_of": "2026-11-23",
  "last_close": 313.55,
  "horizon_days": 252,
  "predicted_return": 0.1126,
  "predicted_growth": 0.1192,
  "predicted_price": 350.93
}
```

`predicted_return` は1年後までの対数リターンです。`predicted_growth = exp(predicted_return) − 1`、`predicted_price = last_close × exp(predicted_return)` です。

エラー時のステータスコード:

| Status | 原因 |
|---|---|
| 404 | NASDAQ-100 以外の銘柄、または株価データがない |
| 422 | ティッカーの形式が不正、存在しないモデル名、`last_years` が 1〜50 の範囲外 |
| 502 | yfinance での株価取得に失敗（ネットワーク接続を確認し、時間を置いて再実行） |
| 503 | モデルが未学習（`uv run cagr-train` を実行してから API を再起動） |

モデルは API の起動時に1回だけ読み込みます。学習し直したら API を再起動してください。

### 4. テストを実行する

```bash
uv run pytest
```

合成した株価データと monkeypatch を使うので、オフラインで実行でき、学習済みモデルも不要です。

## 指標

| 指標 | 計算方法 |
|---|---|
| CAGR | `(終値 / 開始時の終値) ^ (1 / 年数) − 1` |
| 年内騰落率 | その年の最後の終値 / 最初の終値 − 1 |
| 前年比騰落率 | その年の最後の終値 / 前年の最後の終値 − 1 |
| 予測成長率 | 1年後の予測価格 / 現在の終値 − 1 |

終値は yfinance の `auto_adjust=True`（株式分割・配当を調整した値）を使います。

## モデル

| 項目 | 内容 |
|---|---|
| 目的変数 | 252営業日後（約1年後）までの対数リターン `log(close[t+252] / close[t])` |
| 特徴量 | 20 / 60 / 120 / 252日リターン、60 / 252日ボラティリティ、50 / 200日移動平均からの乖離率、1年高値からの下落率、出来高の比率（20日平均 / 252日平均） |
| データ | NASDAQ-100 の101銘柄、全期間の日足 |
| 分割 | 日付の古い80%を train、新しい20%を test。間に252営業日のすき間を空けて、test 期間の価格が学習用の目的変数に入らないようにする |
| モデル | baseline（学習期間の平均リターンを常に予測）/ Ridge / RandomForest / LightGBM |

### 評価結果

2026-11-24 に学習した結果です（train: 1963-01-02 〜 2012-04-19、391,060 行 / test: 2013-04-24 〜 2025-11-20、272,350 行）。

| モデル | MAE | RMSE | 方向的中率 |
|---|---|---|---|
| baseline（平均リターン） | **0.2509** | 0.3630 | **75.1%** |
| Ridge | **0.2509** | **0.3618** | 75.0% |
| RandomForest | 0.2528 | 0.3643 | 74.6% |
| LightGBM | 0.2535 | 0.3654 | 74.5% |

どのモデルも baseline をほとんど上回りませんでした。LightGBM の予測を全銘柄で平均すると、時期によらず +10〜20% 前後でほぼ一定で、実際の1年リターンの大きな変動（-20%〜+110%）は捉えられていません。
過去の株価と出来高だけで1年後のリターンを予測するのは難しい、という結果です。方向的中率が高いのは、test 期間に上昇相場が続き「常にプラスを予測する」だけで当たりやすかったためです。
予測値は参考値として扱ってください。

## 補足

- **銘柄リスト**: `data/nasdaq100.csv` は 2026-11-23 時点の構成銘柄を Nasdaq 公式サイトから取得したものです。銘柄の入れ替えがあった場合は手動で更新してください。現在の構成銘柄で過去を分析しているため、途中で除外された銘柄が含まれない（生存者バイアス）点に注意してください。
- **モデルファイル**: `models/*.joblib` は pickle 形式で、読み込み時に任意のコードを実行できます。自分で学習して作ったファイル以外は読み込まないでください。
- **API の公開範囲**: `fastapi dev` は 127.0.0.1 だけで待ち受けます。認証やレート制限はないので、インターネットに公開する用途は想定していません。

## 免責事項/Disclaimer

### 日本語

本プロジェクトは、教育・研究目的で公開されています。
本ツールおよびその出力結果は、金融商品取引法上の投資助言・勧誘・推奨を目的としたものではありません。
株価、経済指標、CAGRなどの分析結果は過去のデータに基づくものであり、
将来の投資成果やリターンを保証・示唆するものではありません。
利用者が本プログラムまたはその結果をもとに行ういかなる投資行為についても、
その判断および結果の責任はすべて利用者本人に帰属します。
本プロジェクトの作者は、本ソフトウェアの利用または結果に起因する
いかなる損害・損失についても一切の責任を負いません。
また、本ツールを第三者への投資勧誘・販売促進などの目的で利用することを禁止します。

### English

This project is provided for educational and research purposes only.
It does not constitute investment advice, recommendation, or solicitation under any financial regulation (including the Financial Instruments and Exchange Act of Japan).
All analyses, predictions, or CAGR calculations are based on historical data, and
no guarantee or implication of future investment performance is made or intended.
Users are solely responsible for any financial decisions made based on the code or output of this project.
The author assumes no liability for any loss or damage resulting from its use.
This project must not be used for financial promotion, investment solicitation, or commercial advisory purposes.
