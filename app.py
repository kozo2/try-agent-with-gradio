import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import requests
import io
import os
import json
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage

current_fig = None
_current_df = None
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key-here")

def setup_japanese_font():
    font_candidates = [
        'Noto Sans CJK JP', 'IPAGothic', 'IPAPGothic',
        'Hiragino Sans', 'Yu Gothic', 'MS Gothic',
        'TakaoGothic', 'VL Gothic'
    ]
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in font_candidates:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            return font
    plt.rcParams['font.family'] = 'DejaVu Sans'
    return 'DejaVu Sans'

setup_japanese_font()

@tool
def fetch_csv_from_url(url: str) -> str:
    """
    指定されたURLからCSVデータを取得してJSON形式で返す。
    引数:
        url: CSVファイルのURL
    戻り値:
        CSVデータのJSON文字列（最初の10行）と列名情報
    """
    global _current_df
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        encodings = ['utf-8', 'utf-8-sig', 'shift-jis', 'cp932', 'latin-1']
        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(io.StringIO(response.content.decode(enc)))
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        if df is None:
            return json.dumps({"error": "CSVの読み込みに失敗しました"}, ensure_ascii=False)
        _current_df = df
        info = {
            "status": "success",
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "head": df.head(10).to_dict(orient='records'),
            "describe": df.describe().to_dict() if len(df.select_dtypes(include='number').columns) > 0 else {}
        }
        return json.dumps(info, ensure_ascii=False, default=str)
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"データ取得エラー: {str(e)}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"予期しないエラー: {str(e)}"}, ensure_ascii=False)


@tool
def get_popular_csv_datasets(topic: str) -> str:
    """
    トピックに関連する公開CSVデータセットのURLを返す。
    引数:
        topic: データセットのトピック（例: 'population', 'covid', 'weather', 'stock', 'titanic'）
    戻り値:
        利用可能なURLリスト
    """
    datasets = {
        "titanic": {
            "url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
            "description": "タイタニック号の乗客データ"
        },
        "iris": {
            "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
            "description": "アイリスの花のデータ"
        },
        "tips": {
            "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
            "description": "レストランのチップデータ"
        },
        "covid": {
            "url": "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
            "description": "COVID-19の感染者数（世界）"
        },
        "population": {
            "url": "https://raw.githubusercontent.com/datasets/population/master/data/population.csv",
            "description": "世界の人口データ"
        },
        "gapminder": {
            "url": "https://raw.githubusercontent.com/resbaz/r-novice-gapminder-files/master/data/gapminder-FiveYearData.csv",
            "description": "GapMinderの世界経済・人口データ"
        },
        "flights": {
            "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv",
            "description": "航空機乗客数データ（月別）"
        },
        "stock": {
            "url": "https://raw.githubusercontent.com/datasets/s-and-p-500-companies-financials/master/data/constituents-financials.csv",
            "description": "S&P500企業の財務データ"
        },
        "penguins": {
            "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv",
            "description": "ペンギンの体測定データ"
        },
        "cars": {
            "url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv",
            "description": "自動車の燃費データ"
        }
    }
    topic_lower = topic.lower()
    results = []
    keyword_map = {
        "タイタニック": "titanic", "titanic": "titanic",
        "アイリス": "iris", "iris": "iris", "花": "iris",
        "チップ": "tips", "tips": "tips", "レストラン": "tips",
        "コロナ": "covid", "covid": "covid", "感染": "covid",
        "人口": "population", "population": "population",
        "gapminder": "gapminder", "経済": "gapminder", "gdp": "gapminder",
        "flights": "flights", "航空": "flights", "飛行機": "flights",
        "株": "stock", "stock": "stock", "企業": "stock",
        "ペンギン": "penguins", "penguins": "penguins",
        "車": "cars", "cars": "cars", "自動車": "cars", "燃費": "cars"
    }
    matched_keys = set()
    for keyword, key in keyword_map.items():
        if keyword in topic_lower:
            matched_keys.add(key)
    if matched_keys:
        for key in matched_keys:
            if key in datasets:
                results.append({
                    "key": key,
                    "url": datasets[key]["url"],
                    "description": datasets[key]["description"]
                })
    else:
        for key, data in datasets.items():
            results.append({
                "key": key,
                "url": data["url"],
                "description": data["description"]
            })
    return json.dumps(results, ensure_ascii=False)


@tool
def create_visualization(
    chart_type: str,
    x_column: str = "",
    y_column: str = "",
    title: str = "データ可視化",
    hue_column: str = "",
    top_n: int = 20
) -> str:
    """
    取得済みのCSVデータを可視化する。fetch_csv_from_urlで取得後に呼び出す。
    引数:
        chart_type: グラフ種類 ('bar', 'line', 'scatter', 'hist', 'pie', 'box', 'heatmap', 'area')
        x_column: X軸の列名
        y_column: Y軸の列名（複数の場合はカンマ区切り）
        title: グラフのタイトル
        hue_column: 色分けに使用する列名
        top_n: 上位N件を表示（bar/pieの場合）
    戻り値:
        作図の成否とメッセージ
    """
    global current_fig, _current_df
    try:
        if _current_df is None:
            return json.dumps({"error": "データが読み込まれていません。先にfetch_csv_from_urlを使用してください。"}, ensure_ascii=False)
        df = _current_df.copy()
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not x_column and len(df.columns) > 0:
            x_column = categorical_cols[0] if categorical_cols else df.columns[0]
        if not y_column and len(numeric_cols) > 0:
            y_column = numeric_cols[0]
        y_columns = [col.strip() for col in y_column.split(',') if col.strip() in df.columns] if y_column else [numeric_cols[0]] if numeric_cols else []
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        ax.grid(color='#0f3460', linestyle='--', linewidth=0.5, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color('#0f3460')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        colors = ['#e94560', '#0f3460', '#533483', '#e8c547', '#00a8cc', '#f5a623']

        if chart_type == 'bar':
            if x_column in df.columns:
                if x_column in categorical_cols and y_columns:
                    plot_data = df.groupby(x_column)[y_columns[0]].mean().nlargest(top_n)
                    bars = ax.bar(range(len(plot_data)), plot_data.values, color=colors[0], alpha=0.85, edgecolor='white', linewidth=0.5)
                    ax.set_xticks(range(len(plot_data)))
                    ax.set_xticklabels(plot_data.index, rotation=45, ha='right', fontsize=8, color='white')
                    ax.set_ylabel(y_columns[0] if y_columns else '', color='white')
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.1f}', ha='center', va='bottom', color='white', fontsize=7)
                else:
                    if numeric_cols:
                        plot_data = df[numeric_cols[0]].value_counts().head(top_n)
                        ax.bar(range(len(plot_data)), plot_data.values, color=colors[0], alpha=0.85)
                        ax.set_xticks(range(len(plot_data)))
                        ax.set_xticklabels(plot_data.index, rotation=45, ha='right', color='white')

        elif chart_type == 'line':
            if y_columns:
                for i, ycol in enumerate(y_columns[:6]):
                    if ycol in df.columns:
                        color = colors[i % len(colors)]
                        if x_column in df.columns:
                            ax.plot(df[x_column], df[ycol], label=ycol, color=color, linewidth=2, marker='o', markersize=3, alpha=0.9)
                        else:
                            ax.plot(df[ycol].values, label=ycol, color=color, linewidth=2, alpha=0.9)
                if len(y_columns) > 1:
                    ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
                ax.set_xlabel(x_column, color='white')

        elif chart_type == 'scatter':
            if x_column in df.columns and y_columns:
                ycol = y_columns[0]
                if ycol in df.columns:
                    if hue_column and hue_column in df.columns:
                        categories = df[hue_column].unique()[:6]
                        for i, cat in enumerate(categories):
                            mask = df[hue_column] == cat
                            ax.scatter(df.loc[mask, x_column], df.loc[mask, ycol],
                                       label=str(cat), color=colors[i % len(colors)],
                                       alpha=0.7, s=50, edgecolors='white', linewidth=0.3)
                        ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
                    else:
                        ax.scatter(df[x_column], df[ycol], c=colors[0], alpha=0.7, s=50, edgecolors='white', linewidth=0.3)
                    ax.set_xlabel(x_column, color='white')
                    ax.set_ylabel(ycol, color='white')

        elif chart_type == 'hist':
            if y_columns:
                for i, ycol in enumerate(y_columns[:3]):
                    if ycol in df.columns:
                        ax.hist(df[ycol].dropna(), bins=30, label=ycol,
                                color=colors[i % len(colors)], alpha=0.7, edgecolor='white', linewidth=0.3)
                if len(y_columns) > 1:
                    ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
                ax.set_xlabel(y_columns[0] if y_columns else '', color='white')
                ax.set_ylabel('頻度', color='white')
            elif numeric_cols:
                ax.hist(df[numeric_cols[0]].dropna(), bins=30, color=colors[0], alpha=0.8, edgecolor='white', linewidth=0.3)
                ax.set_xlabel(numeric_cols[0], color='white')
                ax.set_ylabel('頻度', color='white')

        elif chart_type == 'pie':
            if x_column in df.columns:
                if x_column in categorical_cols:
                    plot_data = df[x_column].value_counts().head(top_n)
                else:
                    plot_data = df[x_column].head(top_n)
                wedge_colors = plt.cm.Set3(range(len(plot_data)))
                wedges, texts, autotexts = ax.pie(
                    plot_data.values, labels=plot_data.index, autopct='%1.1f%%',
                    colors=wedge_colors, startangle=90, pctdistance=0.85,
                    wedgeprops=dict(edgecolor='white', linewidth=1)
                )
                for text in texts:
                    text.set_color('white')
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(8)

        elif chart_type == 'box':
            if y_columns:
                data_to_plot = [df[col].dropna().values for col in y_columns if col in df.columns]
                labels_to_plot = [col for col in y_columns if col in df.columns]
                if data_to_plot:
                    bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
                    for i, (patch, median) in enumerate(zip(bp['boxes'], bp['medians'])):
                        patch.set_facecolor(colors[i % len(colors)])
                        patch.set_alpha(0.8)
                        median.set_color('white')
                    for element in ['whiskers', 'caps', 'fliers']:
                        for item in bp[element]:
                            item.set_color('white')
                    ax.tick_params(axis='x', colors='white', labelsize=9)

        elif chart_type == 'heatmap':
            if len(numeric_cols) >= 2:
                corr_cols = numeric_cols[:10]
                corr = df[corr_cols].corr()
                im = ax.imshow(corr.values, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
                ax.set_xticks(range(len(corr_cols)))
                ax.set_yticks(range(len(corr_cols)))
                ax.set_xticklabels(corr_cols, rotation=45, ha='right', color='white', fontsize=8)
                ax.set_yticklabels(corr_cols, color='white', fontsize=8)
                plt.colorbar(im, ax=ax)
                for i in range(len(corr_cols)):
                    for j in range(len(corr_cols)):
                        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', color='white', fontsize=7)
            else:
                return json.dumps({"error": "ヒートマップには2列以上の数値データが必要です"}, ensure_ascii=False)

        elif chart_type == 'area':
            if y_columns:
                for i, ycol in enumerate(y_columns[:4]):
                    if ycol in df.columns:
                        color = colors[i % len(colors)]
                        ax.fill_between(range(len(df)), df[ycol], alpha=0.4, color=color, label=ycol)
                        ax.plot(range(len(df)), df[ycol], color=color, linewidth=1.5)
                if len(y_columns) > 1:
                    ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')

        ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        current_fig = fig
        return json.dumps({
            "status": "success",
            "message": f"{chart_type}チャートを作成しました: {title}",
            "chart_type": chart_type,
            "x_column": x_column,
            "y_columns": y_columns
        }, ensure_ascii=False)
    except Exception as e:
        import traceback
        return json.dumps({"error": f"可視化エラー: {str(e)}", "traceback": traceback.format_exc()}, ensure_ascii=False)


@tool
def analyze_dataframe(analysis_type: str = "summary") -> str:
    """
    取得済みのデータフレームを分析する。
    引数:
        analysis_type: 'summary'（統計情報）, 'missing'（欠損値）, 'correlation'（相関）, 'columns'（列情報）
    戻り値:
        分析結果のJSON文字列
    """
    global _current_df
    try:
        if _current_df is None:
            return json.dumps({"error": "データが読み込まれていません"}, ensure_ascii=False)
        df = _current_df
        if analysis_type == "summary":
            result = {
                "shape": list(df.shape),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include='number').columns) > 0 else {}
            }
        elif analysis_type == "missing":
            missing = df.isnull().sum()
            result = {
                "missing_values": {col: int(val) for col, val in missing.items()},
                "missing_percentage": {col: float(f"{(val/len(df)*100):.2f}") for col, val in missing.items()}
            }
        elif analysis_type == "correlation":
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr()
                result = {"correlation_matrix": corr.to_dict()}
            else:
                result = {"error": "相関計算には2列以上の数値データが必要です"}
        elif analysis_type == "columns":
            result = {
                "numeric_columns": df.select_dtypes(include='number').columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                "datetime_columns": df.select_dtypes(include='datetime').columns.tolist(),
                "total_rows": len(df)
            }
        else:
            result = {"error": f"不明な分析タイプ: {analysis_type}"}
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def create_agent(api_key: str):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key,
        streaming=False
    )
    tools = [fetch_csv_from_url, get_popular_csv_datasets, create_visualization, analyze_dataframe]
    system_prompt = """あなたはデータ分析と可視化の専門家AIアシスタントです。
ユーザーのリクエストに応じて以下のことができます：
1. インターネット上の公開CSVデータをURLから取得
2. 取得したデータの統計情報や特徴を分析
3. matplotlibを使ってグラフを作成

利用可能なツール:
- get_popular_csv_datasets: トピックに応じた公開データセットのURLを取得
- fetch_csv_from_url: URLからCSVデータを取得・読み込み
- analyze_dataframe: データの統計情報や列情報を分析
- create_visualization: グラフを作成（bar/line/scatter/hist/pie/box/heatmap/area）

作業フロー:
1. ユーザーのリクエストからトピックを理解
2. get_popular_csv_datasetsで適切なデータセットURLを取得
3. fetch_csv_from_urlでデータを読み込み
4. データの内容を確認して最適なグラフを選択
5. create_visualizationでグラフを作成
6. 結果をわかりやすく説明

グラフ選択の指針:
- カテゴリ比較 → bar
- 時系列・トレンド → line または area
- 2変数の関係 → scatter
- 分布 → hist
- 割合・構成比 → pie
- 複数変数の分布比較 → box
- 相関行列 → heatmap

常に日本語で丁寧に回答し、データの特徴や興味深い発見をわかりやすく説明してください。"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
        return_intermediate_steps=False
    )
    return agent_executor


def process_message(message: str, history: list, api_key: str):
    global current_fig, _current_df
    if not api_key or api_key.strip() == "":
        return history + [[message, "⚠️ OpenAI APIキーを入力してください。"]], None
    if not message.strip():
        return history, None
    try:
        agent = create_agent(api_key.strip())
        chat_history = []
        for human, ai in history:
            chat_history.append(HumanMessage(content=human))
            chat_history.append(AIMessage(content=ai))
        result = agent.invoke({"input": message, "chat_history": chat_history})
        response = result.get("output", "処理に失敗しました。")
        fig_to_show = current_fig
        new_history = history + [[message, response]]
        return new_history, fig_to_show
    except Exception as e:
        error_msg = f"❌ エラーが発生しました: {str(e)}"
        return history + [[message, error_msg]], None


def clear_all():
    global current_fig, _current_df
    current_fig = None
    _current_df = None
    plt.close('all')
    return [], None, ""


CSS = """
.gradio-container {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    color: white !important;
}
#plot-output {
    background: transparent !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
.sample-btn {
    background: rgba(233, 69, 96, 0.2) !important;
    border: 1px solid #e94560 !important;
    color: white !important;
    border-radius: 8px !important;
}
"""

SAMPLE_PROMPTS = [
    "🚢 タイタニックデータを取得して、客室クラス別の生存率を棒グラフで表示して",
    "🌸 アイリスデータを取得して、がく片の長さと幅の散布図を種類別に色分けして表示",
    "🍽️ レストランのチップデータで、曜日別のチップ額の箱ひげ図を作成して",
    "✈️ 航空機乗客数データを取得して、年ごとのトレンドを折れ線グラフで表示",
    "🐧 ペンギンデータで体重と嘴の長さの散布図を種類別に表示して",
    "🚗 自動車データで燃費（mpg）の分布をヒストグラムで表示して",
    "📊 タイタニックデータで数値列の相関ヒートマップを作成して",
    "🌍 GapMinderデータで一人当たりGDPの分布をヒストグラムで表示"
]

with gr.Blocks(css=CSS, title="📊 CSV Data Viz Agent") as demo:
    gr.HTML("""
    
        📊 CSV Data Visualization Agent
        
            チャットで指示するだけで、公開CSVデータを自動取得・可視化！
        
    
    """)
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('💬 チャット')
            api_key_input = gr.Textbox(
                label="🔑 OpenAI API Key",
                placeholder="sk-...",
                type="password",
                value=OPENAI_API_KEY if OPENAI_API_KEY != "your-api-key-here" else ""
            )
            chatbot = gr.Chatbot(label="会話履歴", height=450, bubble_full_width=False)
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="例: タイタニックデータを可視化して",
                    lines=2, scale=4, show_label=False
                )
                submit_btn = gr.Button("送信 🚀", scale=1, variant="primary")
            clear_btn = gr.Button("🗑️ 会話をクリア", variant="secondary")
            gr.HTML('💡 サンプル質問')
            for i in range(0, len(SAMPLE_PROMPTS), 2):
                with gr.Row():
                    gr.Button(SAMPLE_PROMPTS[i], elem_classes=["sample-btn"], size="sm").click(
                        fn=lambda x=SAMPLE_PROMPTS[i]: x, outputs=msg_input
                    )
                    if i + 1 < len(SAMPLE_PROMPTS):
                        gr.Button(SAMPLE_PROMPTS[i+1], elem_classes=["sample-btn"], size="sm").click(
                            fn=lambda x=SAMPLE_PROMPTS[i+1]: x, outputs=msg_input
                        )
        with gr.Column(scale=1):
            gr.HTML('📈 可視化結果')
            plot_output = gr.Plot(label="グラフ", elem_id="plot-output")
            gr.HTML("""
            
                📚 利用可能なデータセット
                
                    🚢 Titanicタイタニック乗客データ
                    🌸 Irisアイリスの花データ
                    🍽️ Tipsレストランチップデータ
                    ✈️ Flights航空機乗客数データ
                    🐧 Penguinsペンギン測定データ
                    🚗 Cars/MPG自動車燃費データ
                    🌍 Gapminder世界経済・人口データ
                    📈 S&P500S&P500財務データ
                
            
            """)

    def submit_message(message, history, api_key):
        if not message.strip():
            return history, None, ""
        new_history, fig = process_message(message, history, api_key)
        return new_history, fig, ""

    submit_btn.click(
        fn=submit_message,
        inputs=[msg_input, chatbot, api_key_input],
        outputs=[chatbot, plot_output, msg_input]
    )
    msg_input.submit(
        fn=submit_message,
        inputs=[msg_input, chatbot, api_key_input],
        outputs=[chatbot, plot_output, msg_input]
    )
    clear_btn.click(fn=clear_all, outputs=[chatbot, plot_output, msg_input])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
