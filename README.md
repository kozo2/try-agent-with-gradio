# try-agent-with-gradio

```markdown
# 📊 CSV Data Visualization Agent

チャットで指示するだけで、インターネット上の公開CSVデータを自動取得・matplotlibで可視化するGradio Agentアプリです。

---

## 🔧 必要条件

- Python 3.11以上
- [uv](https://docs.astral.sh/uv/getting-started/installation/) がインストール済みであること
- OpenAI APIキー

---

## 🚀 セットアップと起動手順

### 1. uvのインストール（未インストールの場合）

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

### 2. リポジトリのクローン（またはファイルを配置）

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

またはファイルを手動で配置する場合：

```
your-project/
├── app.py
└── README.md
```

---

### 3. プロジェクトの初期化

```bash
uv init --no-package
```

---

### 4. 依存ライブラリの追加

```bash
uv add gradio pandas matplotlib requests langchain langchain-openai langchain-community
```

---

### 5. OpenAI APIキーの設定

**macOS / Linux**
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

**Windows (PowerShell)**
```powershell
$env:OPENAI_API_KEY="sk-your-api-key-here"
```

**Windows (コマンドプロンプト)**
```cmd
set OPENAI_API_KEY=sk-your-api-key-here
```

> **Note**: `.env`ファイルを使う場合は後述の方法を参照してください。

---

### 6. アプリの起動

```bash
uv run app.py
```

起動後、ブラウザで以下のURLにアクセスしてください：

```
http://localhost:7860
```

---

## 🔑 APIキーを.envファイルで管理する方法（推奨）

### 1. python-dotenvを追加

```bash
uv add python-dotenv
```

### 2. .envファイルを作成

```bash
touch .env
```

`.env`の内容：

```env
OPENAI_API_KEY=sk-your-api-key-here
```

### 3. app.pyの先頭に以下を追加

```python
from dotenv import load_dotenv
load_dotenv()
```

### 4. .gitignoreに.envを追加

```bash
echo ".env" >> .gitignore
```

---

## 📁 プロジェクト構成

```
your-project/
├── app.py          # メインアプリケーション
├── README.md       # このファイル
├── .env            # APIキー（Gitにコミットしない）
├── .gitignore
├── pyproject.toml  # uv依存関係ファイル（uv initで自動生成）
└── .venv/          # 仮想環境（uv管理）
```

---

## 📚 利用可能なデータセット

| アイコン | キーワード | 内容 |
|---------|-----------|------|
| 🚢 | titanic / タイタニック | タイタニック号の乗客データ |
| 🌸 | iris / アイリス / 花 | アイリスの花の測定データ |
| 🍽️ | tips / チップ / レストラン | レストランのチップデータ |
| ✈️ | flights / 航空 / 飛行機 | 航空機の月別乗客数データ |
| 🐧 | penguins / ペンギン | ペンギンの体測定データ |
| 🚗 | cars / 車 / 燃費 | 自動車の燃費データ |
| 🌍 | gapminder / 経済 / GDP | 世界経済・人口データ |
| 📈 | stock / 株 / 企業 | S&P500企業の財務データ |

---

## 📊 対応グラフ種類

| 種類 | 用途 |
|------|------|
| `bar` | カテゴリ比較 |
| `line` | 時系列・トレンド |
| `scatter` | 2変数の関係 |
| `hist` | 分布表示 |
| `pie` | 割合・構成比 |
| `box` | 分布の比較 |
| `heatmap` | 相関行列 |
| `area` | 面グラフ |

---

## 💡 使用例

```
タイタニックデータを取得して、客室クラス別の生存率を棒グラフで表示して
アイリスデータを取得して、がく片の長さと幅の散布図を種類別に色分けして表示
レストランのチップデータで、曜日別のチップ額の箱ひげ図を作成して
航空機乗客数データを取得して、年ごとのトレンドを折れ線グラフで表示
ペンギンデータで体重と嘴の長さの散布図を種類別に表示して
```

---

## ❓ トラブルシューティング

### `uv: command not found`
uvが正しくインストールされていません。セットアップ手順1を再度実施してください。  
インストール後にターミナルを再起動してください。

### `OPENAI_API_KEY not set`
APIキーが設定されていません。手順5または.envファイルの設定を確認してください。

### ポート7860が使用中
```bash
uv run app.py  # app.py内のserver_portを変更してください（例: 7861）
```

### 日本語フォントが表示されない（Linux環境）
```bash
uv add japanize-matplotlib
```
`app.py`の先頭に以下を追加：
```python
import japanize_matplotlib
```

### パッケージのインストールエラー
```bash
uv sync --reinstall
```
