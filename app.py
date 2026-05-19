import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components
import time

# =====================================================================================================
# 1. 視覺風格與 CSS 定義
# =====================================================================================================
def apply_custom_style() -> None:
    """
    注入自定義 CSS 樣式以優化 Streamlit 介面外觀。

    透過 st.markdown 將一段 CSS 字串嵌入頁面的 <style> 標籤中，
    涵蓋背景色、按鈕漸層、卡片陰影、分頁樣式等視覺設定。

    Args:
        無

    Returns:
        None：此函式僅執行 Streamlit 副作用（注入 CSS），不回傳任何值。
    """
    st.markdown("""
        <style>
        /* 主背景色 */
        .main { background-color: #fbfbfb; }

        /* 指標卡片樣式 */
        .stMetric {
            background-color: #ffffff !important;
            padding: 15px !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
            border: 1px solid #eee !important;
        }

        /* 主要按鈕樣式（藍色漸層） */
        .stButton>button {
            width: 100%;
            border-radius: 12px !important;
            background: linear-gradient(45deg, #007bff, #0056b3) !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            padding: 12px 20px !important;
            box-shadow: 0 4px 10px rgba(0,123,255,0.3) !important;
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #0056b3, #004085) !important;
        }

        /* 特定按鈕樣式（紅色漸層，用於重置等危險操作） */
        div.stButton > div.st-emotion-cache-micr9v > button {
            background: linear-gradient(45deg, #ff4b2b, #ff416c) !important;
            box-shadow: 0 4px 10px rgba(255,75,43,0.3) !important;
        }

        /* Expander 容器樣式 */
        div[data-testid="stExpander"] {
            border: none !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            background-color: white;
            border-radius: 10px;
        }

        /* 模式選擇器區塊 */
        .mode-selector {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 20px;
            border: 1px solid #dee2e6;
            margin-bottom: 30px;
            text-align: center;
        }

        /* 計算結果展示框（藍色左邊框） */
        .calc-box {
            background-color: #ffffff;
            padding: 20px;
            border-left: 6px solid #007bff;
            border-radius: 8px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }

        /* 公式解釋說明框（黃色底） */
        .explain-box {
            background-color: #fff9db;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #ffe066;
            margin-top: 10px;
            color: #856404;
            font-size: 0.95rem;
        }

        /* 路徑顯示框（灰底等寬字體） */
        .path-box {
            background-color: #f1f3f5;
            padding: 15px;
            border-radius: 12px;
            border: 2px dashed #adb5bd;
            font-family: 'Consolas', monospace;
            color: #495057;
        }

        /* 分頁標籤樣式 */
        .stTabs [data-baseweb="tab"] {
            background-color: #fff;
            border: 1px solid #dee2e6;
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
        }
        </style>
    """, unsafe_allow_html=True)


# =====================================================================================================
# 2. 數學核心邏輯
# =====================================================================================================
def build_transition_matrix(
    n: int,
    edges_with_weights: list[tuple[int, int, float]],
    allow_self_loop: bool = True
) -> tuple[np.ndarray, dict[int, list[tuple[int, float]]]]:
    """
    根據節點數、帶權邊列表與是否允許自環，建立馬可夫轉移矩陣 P。

    每個節點 i 的轉移機率計算邏輯：
        - 分母 = 所有鄰居的邊權重總和 + 自環權重（1.0 或 0.0）
        - P[i-1, j-1] = weight(i→j) / 分母
        - P[i-1, i-1] = self_weight / 分母（若允許自環）

    Args:
        n (int):
            圖中節點的總數量，節點編號範圍為 1 ~ n。
        edges_with_weights (list[tuple[int, int, float]]):
            無向邊列表，每個元素為 (u, v, w)，表示節點 u 與 v 之間
            存在權重為 w 的邊。超出 [1, n] 範圍的節點編號會被忽略。
        allow_self_loop (bool, optional):
            是否在轉移矩陣中加入自環（停留原地的機率）。
            - True：自環權重設為 1.0（預設）
            - False：自環權重設為 0.0

    Returns:
        tuple[np.ndarray, dict[int, list[tuple[int, float]]]]:
            - P (np.ndarray)：形狀為 (n, n) 的浮點數矩陣，
              P[i][j] 代表從節點 i+1 移動到節點 j+1 的機率，
              每列（row）的元素總和為 1.0。
            - adj (dict[int, list[tuple[int, float]]])：
              鄰接表，鍵為節點編號（1-indexed），
              值為 [(鄰居節點編號, 邊權重), ...] 的列表。
    """
    # 初始化 n×n 的零矩陣作為轉移矩陣
    P = np.zeros((n, n))

    # 初始化鄰接表，每個節點對應一個空列表
    adj = {i: [] for i in range(1, n + 1)}

    # 根據無向邊列表填入鄰接表（雙向）
    for u, v, w in edges_with_weights:
        if 1 <= u <= n and 1 <= v <= n:   # 過濾超出範圍的節點
            adj[u].append((v, w))
            adj[v].append((u, w))

    # 自環的預設權重（允許時為 1.0，否則為 0.0）
    self_weight = 1.0 if allow_self_loop else 0.0

    # 逐一計算每個節點的轉移機率
    for i in range(1, n + 1):
        neighbors = adj[i]
        # 分母 = 所有鄰居邊的權重總和 + 自環權重
        total_weight = sum([w for v, w in neighbors]) + self_weight

        # 若該節點完全孤立且無自環，跳過以避免除以零
        if total_weight == 0:
            continue

        # 填入自環機率（對角線元素）
        P[i-1, i-1] = self_weight / total_weight

        # 填入移動到各鄰居節點的機率
        for v, w in neighbors:
            P[i-1, v-1] = w / total_weight

    return P, adj


def find_steady_state(
    P: np.ndarray,
    threshold: float
) -> tuple[np.ndarray, int, list[float]]:
    """
    使用冪法（Power Iteration）迭代計算馬可夫鏈的穩定狀態分佈 π。

    迭代邏輯：
        v^(t+1) = v^(t) @ P
    重複直到相鄰兩次迭代的最大誤差 max|v^(t+1) - v^(t)| < threshold，
    或迭代次數超過 10,000 次為止。

    Args:
        P (np.ndarray):
            形狀為 (n, n) 的轉移矩陣，由 build_transition_matrix 產生。
        threshold (float):
            收斂判斷閾值，當最大誤差低於此值時停止迭代。
            建議範圍：1e-4 ~ 1e-8，越小精度越高但需要更多迭代次數。

    Returns:
        tuple[np.ndarray, int, list[float]]:
            - v (np.ndarray)：長度為 n 的一維陣列，代表穩定狀態機率分佈，
              v[i] 為長期停留在節點 i+1 的機率，所有元素總和接近 1.0。
            - iteration (int)：實際執行的迭代次數。
              若等於 10,000 表示未在閾值內收斂。
            - error_history (list[float])：每次迭代的最大誤差列表，
              可用於繪製收斂曲線，長度等於 iteration + 1。
    """
    n = P.shape[0]

    # 若矩陣為空（n=0），直接回傳空結果
    if n == 0:
        return np.array([]), 0, []

    # 初始狀態向量：從節點 1 出發，機率為 1.0
    v = np.zeros(n)
    v[0] = 1.0

    error_history = []   # 記錄每輪的最大誤差
    iteration = 0

    while True:
        # 執行一步矩陣乘法：v^(t+1) = v^(t) @ P
        v_next = np.dot(v, P)

        # 計算本輪最大誤差（L∞ 範數）
        error = np.max(np.abs(v_next - v))
        error_history.append(error)

        # 收斂條件：誤差低於閾值，或達到上限次數
        if error < threshold or iteration > 10000:
            break

        v = v_next
        iteration += 1

    return v, iteration, error_history


def get_convergence_history_fixed(
    P: np.ndarray,
    max_iters: int
) -> list[float]:
    """
    執行固定次數的冪法迭代，並記錄每次迭代的最大誤差，
    用於繪製收斂趨勢圖（不依賴自動收斂閾值）。

    Args:
        P (np.ndarray):
            形狀為 (n, n) 的轉移矩陣。
        max_iters (int):
            固定的迭代執行次數，無論是否收斂皆會執行完畢。

    Returns:
        list[float]：長度為 max_iters 的誤差列表，
            第 i 個元素代表第 i+1 次迭代後的 max|v_next - v|。
            若 P 為空矩陣（n=0），則回傳空列表。
    """
    n = P.shape[0]

    # 若矩陣為空，直接回傳空列表
    if n == 0:
        return []

    # 初始狀態向量：從節點 1 出發
    v = np.zeros(n)
    v[0] = 1.0

    error_history = []

    for i in range(max_iters):
        v_next = np.dot(v, P)

        # 記錄本輪最大誤差
        error = np.max(np.abs(v_next - v))
        error_history.append(error)

        v = v_next   # 更新狀態向量，進行下一輪

    return error_history


# =====================================================================================================
# 3. 視覺化模組
# =====================================================================================================
def create_interactive_graph(
    n: int,
    edges_with_weights: list[tuple[int, int, float]],
    steady_v: np.ndarray | None = None,
    fixed_pos: dict[int, tuple[float, float]] | None = None,
    label_prefix: str = "位置"
) -> str:
    """
    使用 PyVis 建立互動式網路拓撲圖，並將結果儲存為 HTML 檔案。

    節點顏色映射規則：
        - 若未提供 steady_v，所有節點均顯示為淺藍色 (#ADD8E6)。
        - 若提供 steady_v，節點顏色依穩定機率深淺變化：
          機率越高 → RGB 中紅色分量越高、綠/藍分量越低（趨近紅色）。

    Args:
        n (int):
            節點總數量（編號 1 ~ n）。
        edges_with_weights (list[tuple[int, int, float]]):
            帶權邊列表，每個元素為 (u, v, w)。
        steady_v (np.ndarray | None, optional):
            穩定狀態機率向量（長度為 n），用於節點染色與懸停提示。
            傳入 None 時，所有節點顯示預設顏色。預設為 None。
        fixed_pos (dict[int, tuple[float, float]] | None, optional):
            節點固定座標字典，鍵為節點編號（1-indexed），
            值為 (x, y) 座標（PyVis 畫布座標，單位 px）。
            傳入 None 時，使用 Barnes-Hut 自動佈局演算法。預設為 None。
        label_prefix (str, optional):
            節點標籤的前綴文字，例如 "路口" 或 "位置"。預設為 "位置"。

    Returns:
        str：儲存互動圖的 HTML 檔案路徑（固定為 "graph.html"），
            可透過 open() 讀取後傳入 st.components.v1.html() 渲染。
    """
    # 建立 PyVis Network 物件，設定畫布尺寸與背景色
    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")

    # 依照是否提供固定座標，選擇佈局模式
    if fixed_pos:
        # 停用物理引擎，使節點保持在指定位置
        net.set_options('{"physics":{"enabled":false}, "nodes":{"font":{"size":16}}}')
    else:
        # 使用 Barnes-Hut 演算法自動排列節點
        net.barnes_hut()

    # 逐一新增節點，套用顏色與懸停提示
    for i in range(1, n + 1):
        # 預設節點顏色：淺藍色
        color = "#ADD8E6"

        if steady_v is not None and len(steady_v) >= i:
            # 依機率值計算顏色強度（0~255），機率越高越紅
            intensity = int(steady_v[i-1] * 255)
            color = f"rgb(255, {255-min(intensity, 255)}, {255-min(intensity, 255)})"

        if fixed_pos:
            # 使用固定座標新增節點
            pos = fixed_pos.get(i, (0, 0))
            net.add_node(
                i,
                label=f"{label_prefix} {i}",
                color=color,
                x=pos[0],
                y=pos[1],
                title=f"機率: {steady_v[i-1]:.4f}" if steady_v is not None else ""
            )
        else:
            # 使用自動佈局新增節點
            net.add_node(
                i,
                label=f"{label_prefix} {i}",
                color=color,
                title=f"機率: {steady_v[i-1]:.4f}" if steady_v is not None else ""
            )

    # 新增所有邊（邊的粗細由權重 value 控制）
    for u, v, w in edges_with_weights:
        net.add_edge(u, v, value=w)

    # 將互動圖輸出為 HTML 檔案
    net.save_graph("graph.html")

    return "graph.html"   # 回傳 HTML 檔案的相對路徑


def draw_simulation_frame(
    n: int,
    edges: list[tuple[int, int, float]],
    current_node: int,
    steady_v: np.ndarray,
    fixed_pos: dict[int, tuple[float, float]] | None = None
) -> plt.Figure:
    """
    使用 NetworkX + Matplotlib 繪製行走模擬的單一畫格（frame）。

    目前所在節點以黃色（#FFFF00）標示，其餘節點顯示淺藍色（#ADD8E6）。

    Args:
        n (int):
            圖中節點的總數量（編號 1 ~ n）。
        edges (list[tuple[int, int, float]]):
            邊列表，每個元素為 (u, v, w)，繪圖時只使用 (u, v) 部分。
        current_node (int):
            目前行走者所在的節點編號（1-indexed），此節點將以黃色高亮。
        steady_v (np.ndarray):
            穩定狀態機率向量，目前此參數在函式內未被使用，
            保留作為未來擴充（例如節點大小映射）之用。
        fixed_pos (dict[int, tuple[float, float]] | None, optional):
            節點的固定座標字典。格式為 {節點編號: (x, y)}。
            傳入 None 時，使用 spring_layout 自動計算座標（固定 seed=42）。
            預設為 None。

    Returns:
        matplotlib.figure.Figure：包含單一畫格的 Matplotlib Figure 物件。
            呼叫端負責在適當時機執行 plt.close(fig) 以釋放記憶體。
    """
    # 建立 NetworkX 無向圖並加入節點與邊
    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    G.add_edges_from([(u, v) for u, v, w in edges])   # 忽略邊權重，僅使用拓撲結構

    # 決定節點座標佈局
    pos = fixed_pos if fixed_pos else nx.spring_layout(G, seed=42)

    # 依照目前位置設定節點顏色（黃色 = 目前位置；淺藍 = 其他節點）
    node_colors = [
        "#FFFF00" if i == current_node else "#ADD8E6"
        for i in range(1, n + 1)
    ]

    # 建立 Matplotlib Figure 並繪製圖形
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=600,
        edge_color="#D3D3D3",
        font_size=10,
        font_weight='bold',
        ax=ax
    )
    plt.axis('off')   # 隱藏座標軸

    return fig   # 回傳 Figure 物件供 Streamlit 渲染


# =====================================================================================================
# 4. Streamlit 主界面
# =====================================================================================================
st.set_page_config(page_title="Markov Analysis Suite Pro", layout="wide")
apply_custom_style()

# --- 模式選擇區 ---
st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #2c3e50;'>🛠️ 系統分析模式選擇</h2>", unsafe_allow_html=True)
mode = st.radio(
    "請選擇您要分析的對象：",
    ["👮 交通警察巡邏 (Police Patrol)", "🐁 8格迷宮老鼠 (Mouse Maze)"],
    horizontal=True,
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# --- 預設拓撲（5節點星形 + 環形混合圖）---
INITIAL_TOPO = {
    'n_nodes': 5,
    'edges': [
        (1, 2, 1.0), (2, 4, 1.0), (4, 3, 1.0), (3, 1, 1.0),   # 外環
        (1, 5, 1.0), (2, 5, 1.0), (3, 5, 1.0), (4, 5, 1.0)     # 連接中心節點 5
    ],
    'fixed_pos': {1: (0, 100), 2: (100, 100), 3: (0, 0), 4: (100, 0), 5: (50, 50)},
    'allow_self_loop': True
}

# 初始化 session state，避免每次重新渲染時重置拓撲設定
if 'topo_data' not in st.session_state:
    st.session_state.topo_data = INITIAL_TOPO.copy()

# --- 側邊欄配置 ---
st.sidebar.header("⚙️ 配置中心")

if mode == "👮 交通警察巡邏 (Police Patrol)":
    with st.sidebar.expander("📍 佈局設定", expanded=True):
        layout_type = st.selectbox("選擇佈局", ["(5節點)佈局", "3x4 網格", "自定義網格", "手動輸入"])

        if layout_type == "(5節點)佈局":
            # 使用預設的 5 節點拓撲
            st.session_state.topo_data = INITIAL_TOPO.copy()

        elif layout_type == "3x4 網格":
            # 建立 3 行 4 列的網格圖，僅含水平與垂直相鄰邊
            edges = []
            for r in range(3):
                for c in range(4):
                    u = r * 4 + c + 1
                    if c < 3: edges.append((u, u + 1, 1.0))    # 水平邊
                    if r < 2: edges.append((u, u + 4, 1.0))    # 垂直邊
            st.session_state.topo_data = {
                'n_nodes': 12, 'edges': edges,
                'fixed_pos': None, 'allow_self_loop': True
            }

        elif layout_type == "自定義網格":
            rows, cols = st.number_input("行數", 1, 10, 3), st.number_input("列數", 1, 10, 4)
            edges = []
            for r in range(rows):
                for c in range(cols):
                    u = r * cols + c + 1
                    if c < cols - 1: edges.append((u, u + 1, 1.0))        # 水平邊
                    if r < rows - 1: edges.append((u, u + cols, 1.0))     # 垂直邊
            st.session_state.topo_data = {
                'n_nodes': rows * cols, 'edges': edges,
                'fixed_pos': None, 'allow_self_loop': True
            }

        elif layout_type == "手動輸入":
            raw_input = st.text_area("編輯關係清單 (u,v,w)", "1,2,1.0\n2,3,1.0\n3,1,1.0")
            temp_edges, curr_max = [], 0
            for line in raw_input.split('\n'):
                if line.strip():
                    try:
                        u, v, w = map(float, line.split(','))
                        temp_edges.append((int(u), int(v), w))
                        curr_max = max(curr_max, int(u), int(v))
                    except:
                        pass   # 忽略格式錯誤的行
            st.session_state.topo_data = {
                'n_nodes': max(curr_max, 2), 'edges': temp_edges,
                'fixed_pos': None, 'allow_self_loop': True
            }

elif mode == "🐁 8格迷宮老鼠 (Mouse Maze)":
    with st.sidebar.expander("📍 迷宮設定", expanded=True):
        st.write("此模式為固定線性迷宮")
        # 建立 1~8 的線性鏈（路徑圖），不允許自環
        edges = [(i, i + 1, 1.0) for i in range(1, 8)]
        fixed_pos = {i: (i * 100, 0) for i in range(1, 9)}   # 水平等距排列
        st.session_state.topo_data = {
            'n_nodes': 8, 'edges': edges,
            'fixed_pos': fixed_pos, 'allow_self_loop': False
        }

# --- 數學精度設定 ---
with st.sidebar.expander("📈 數學精度設定", expanded=False):
    threshold = st.number_input("收斂閾值", value=0.000001, format="%.7f")

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ 系統管理")

# 重置按鈕：清除所有自定義配置，恢復預設拓撲
if st.sidebar.button("🔄 一鍵重置所有配置", key="reset_btn"):
    st.session_state.topo_data = INITIAL_TOPO.copy()
    st.rerun()

# --- 從 session state 讀取當前拓撲參數 ---
n_nodes = st.session_state.topo_data['n_nodes']
edges_with_weights = st.session_state.topo_data['edges']
fixed_pos = st.session_state.topo_data['fixed_pos']
allow_self = st.session_state.topo_data['allow_self_loop']

# 依模式設定節點標籤前綴
label_prefix = "路口" if mode == "👮 交通警察巡邏 (Police Patrol)" else "位置"

# --- 計算轉移矩陣與穩定狀態 ---
# P：(n_nodes × n_nodes) 轉移矩陣
# adj：鄰接表（用於計算詳情頁）
P, adj = build_transition_matrix(n_nodes, edges_with_weights, allow_self_loop=allow_self)

# steady_v：穩定狀態機率向量（長度 n_nodes）
# iters：達到收斂所需的迭代次數
# error_hist_auto：自動收斂過程的誤差歷史
steady_v, iters, error_hist_auto = find_steady_state(P, threshold)

# --- 頂部概覽指標 ---
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("路口/位置規模", f"{n_nodes} 處")
m_col2.metric("自動收斂次數", f"{iters} 次")
m_col3.metric("系統狀態", "穩定" if iters < 10000 else "未收斂")

# --- 分頁列表建立 ---
tabs_list = [
    "🌐 互動拓撲圖", "⏱️ 隨機行走模擬", "📈 步數分佈演進",
    "📊 轉移矩陣", "📉 收斂趨勢", "🎯 穩定狀態", "📝 計算詳情", "📐 數學原理"
]

# 迷宮模式額外插入矩陣運算分析頁
if mode == "🐁 8格迷宮老鼠 (Mouse Maze)":
    tabs_list.insert(4, "🧮 矩陣運算分析")

tab_objs = st.tabs(tabs_list)
# 建立 {標籤名稱: tab 物件} 的對應字典，方便以名稱索引
tab_map = {name: tab for name, tab in zip(tabs_list, tab_objs)}


# =====================================================================================================
# 分頁內容：互動拓撲圖
# =====================================================================================================
with tab_map["🌐 互動拓撲圖"]:
    st.subheader(f"{label_prefix}連接視覺化")
    # 建立並輸出 PyVis HTML，節點依穩定機率染色
    graph_html = create_interactive_graph(
        n_nodes, edges_with_weights, steady_v, fixed_pos, label_prefix
    )
    with open(graph_html, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=550)


# =====================================================================================================
# 分頁內容：隨機行走模擬
# =====================================================================================================
with tab_map["⏱️ 隨機行走模擬"]:
    st.subheader("實時行走模擬")
    col_ctrl, col_map = st.columns([1, 2])

    with col_ctrl:
        start_node = st.number_input("設定起點", 1, n_nodes, 1)
        sim_steps = st.slider("模擬時長", 1, 100, 20)
        speed = st.slider("動畫速度", 0.1, 1.0, 0.3)
        run_btn = st.button("🚀 開始模擬")

    with col_map:
        map_placeholder = st.empty()      # 用於更新模擬圖的佔位元件
        status_placeholder = st.empty()   # 用於顯示當前步數與位置
        path_placeholder = st.empty()     # 用於顯示累積路徑

        if run_btn:
            current = start_node
            visited_path = [current]

            for i in range(sim_steps):
                # 繪製目前節點高亮的畫格
                fig = draw_simulation_frame(
                    n_nodes, edges_with_weights, current, steady_v, fixed_pos
                )
                map_placeholder.pyplot(fig)

                # 更新步數與位置提示
                status_placeholder.markdown(
                    f"**狀態**：第 {i+1} 步 $\rightarrow$ 位於 **{label_prefix} {current}**"
                )

                # 更新路徑紀錄
                path_str = " $\rightarrow$ ".join(map(str, visited_path))
                path_placeholder.markdown(
                    f'<div class="path-box"><strong style="color:#007bff;">🚶 實時路徑紀錄：</strong>'
                    f'<br>{path_str}</div>',
                    unsafe_allow_html=True
                )

                # 依照轉移機率隨機選擇下一個節點
                probs = P[current-1, :]
                current = np.random.choice(
                    range(1, n_nodes + 1),
                    p=probs / np.sum(probs)   # 重新正規化以避免浮點誤差
                )
                visited_path.append(current)

                time.sleep(speed)    # 控制動畫播放速度
                plt.close(fig)       # 釋放 Figure 記憶體

            st.success(
                f"✅ 模擬結束。完整路徑：{' $\rightarrow$ '.join(map(str, visited_path))}"
            )


# =====================================================================================================
# 分頁內容：步數分佈演進（含 n 步轉移公式）
# =====================================================================================================
with tab_map["📈 步數分佈演進"]:
    st.subheader("🚶 從位置 1 出發的機率演進")

    # 展示 n 步轉移機率的數學公式
    st.markdown('<div class="calc-box">', unsafe_allow_html=True)
    st.markdown("**數學依據：n 步轉移機率計算**")
    # (P^n)_{ij} = Σ_{k=1}^{m} P_{ik}^{(n-1)} P_{kj}
    st.latex(r"(P^n)_{ij} = \sum_{k=1}^{m} P_{ik}^{(n-1)} P_{kj}")
    st.markdown(f"""
    <div class="explain-box">
    <strong>💡 公式解析：</strong><br>
    這代表從位置 $i$ 出發，經過 $n$ 步到達位置 $j$ 的機率。
    它是透過加總所有可能的中間點 $k$（總共 $m={n_nodes}$ 個點），
    計算「前 $n-1$ 步到達 $k$」且「最後一步從 $k$ 到達 $j$」的機率總和。
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("設定步數 $m$，計算在第 $m$ 步時，出現在各個位置的機率分布 $v^{(m)} = v^{(0)} P^m$")

    col_in, col_out = st.columns([1, 2])

    with col_in:
        m_steps = st.number_input("設定步數 (m)", 0, 500, 1)
        st.info("當 $m=0$ 時，機率 100% 在位置 1。")

    with col_out:
        # 初始狀態向量：從節點 1 出發
        v0 = np.zeros(n_nodes)
        v0[0] = 1.0

        # 計算 P 的 m 次方，再乘以初始向量，得到第 m 步的機率分佈
        Pm = np.linalg.matrix_power(P, m_steps)   # P^m，形狀 (n, n)
        vm = np.dot(v0, Pm)                         # v^(m) = v^(0) @ P^m，長度 n

        df_vm = pd.DataFrame({
            "位置": [f"{label_prefix} {i+1}" for i in range(n_nodes)],
            "機率": vm
        })

        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.markdown("**機率分佈表**")
            st.table(df_vm.style.format({"機率": "{:.4%}"}))
        with res_col2:
            st.markdown("**機率可視化**")
            fig_vm, ax_vm = plt.subplots(figsize=(5, 4))
            ax_vm.bar(df_vm["位置"], df_vm["機率"], color="#007bff")
            ax_vm.set_ylabel("Probability")
            ax_vm.set_ylim(0, 1.0)
            ax_vm.set_title(f"Distribution after {m_steps} steps")
            st.pyplot(fig_vm)
            plt.close(fig_vm)

    st.markdown("---")
    st.subheader("⏳ 隨步數增加的機率變化趨勢")
    max_trace = st.slider("分析總步數", 1, 100, 20)

    # 逐步迭代計算每步的機率分佈，並累積為 DataFrame
    trace_data = []
    curr_v = v0.copy()
    for s in range(max_trace + 1):
        trace_data.append(curr_v.copy())   # 記錄第 s 步的機率快照
        curr_v = np.dot(curr_v, P)         # 前進一步

    df_trace = pd.DataFrame(
        trace_data,
        columns=[f"{label_prefix} {i+1}" for i in range(n_nodes)]
    )

    # 繪製各節點機率隨步數變化的折線圖
    fig_trace, ax_trace = plt.subplots(figsize=(10, 4))
    for col in df_trace.columns:
        ax_trace.plot(df_trace.index, df_trace[col], label=col, marker='.', markersize=4)
    ax_trace.set_xlabel("Steps (m)")
    ax_trace.set_ylabel("Probability")
    ax_trace.set_title("Probability Evolution over Time (Starting from Pos 1)")
    ax_trace.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax_trace.grid(True, alpha=0.3)
    st.pyplot(fig_trace)
    plt.close(fig_trace)


# =====================================================================================================
# 分頁內容：矩陣運算分析（僅迷宮模式）
# =====================================================================================================
if "🧮 矩陣運算分析" in tab_map:
    with tab_map["🧮 矩陣運算分析"]:
        st.subheader("🎯 特定步數機率計算")
        col_input, col_res = st.columns([1, 2])

        with col_input:
            start_node_m = st.number_input("設定起始位置 $v^{(0)}$", 1, n_nodes, 1)
            target_node_m = st.number_input("設定目標位置", 1, n_nodes, 5)
            steps_m = st.number_input("計算步數 $m$", 1, 100, 2)

        with col_res:
            # 計算 P^m，再從指定起點出發得到第 m 步的機率向量
            Pm = np.linalg.matrix_power(P, steps_m)   # P^m，形狀 (n, n)
            v0 = np.zeros(n_nodes)
            v0[start_node_m - 1] = 1.0                # 起點機率為 1
            vm = np.dot(v0, Pm)                        # v^(m) = v^(0) @ P^m

            # 顯示目標位置的機率
            st.metric(
                f"經過 {steps_m} 步後，在位置 {target_node_m} 的機率",
                f"{vm[target_node_m-1]:.4%}"
            )

            # 顯示所有位置的機率長條圖
            df_vm = pd.DataFrame({"位置": range(1, n_nodes + 1), "機率": vm})
            st.bar_chart(df_vm.set_index("位置")["機率"])


# =====================================================================================================
# 分頁內容：轉移矩陣
# =====================================================================================================
with tab_map["📊 轉移矩陣"]:
    st.subheader("轉移矩陣 $P$ (行加總為 1)")

    # 計算每列的加總（應等於 1.0）以驗證矩陣正確性
    row_sums = P.sum(axis=1)

    # 建立帶標籤的 DataFrame，並附加行加總欄位供驗證
    df_P = pd.DataFrame(
        P,
        index=[f"{label_prefix} {i+1}" for i in range(n_nodes)],
        columns=[f"{label_prefix} {i+1}" for i in range(n_nodes)]
    )
    df_P['行加總 (Sum)'] = row_sums

    st.dataframe(df_P.style.format("{:.4f}"))
    st.caption("💡 驗證：每一行的『行加總』應精確等於 1.0000，代表機率分佈完整。")


# =====================================================================================================
# 分頁內容：收斂趨勢
# =====================================================================================================
with tab_map["📉 收斂趨勢"]:
    st.subheader("收斂過程分析")
    col_ctrl, col_info = st.columns([1, 1])

    with col_ctrl:
        user_iters = st.slider("調整迭代次數 (Iterations)", 1, 500, 100)
    with col_info:
        st.info(f"目前設定執行 {user_iters} 次迭代。您可以觀察 Max Error 如何隨著次數增加而下降。")

    # 執行固定次數迭代，取得誤差歷史列表（長度 = user_iters）
    error_hist_user = get_convergence_history_fixed(P, user_iters)

    if len(error_hist_user) > 0:
        fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
        ax_conv.plot(
            error_hist_user, color='#007bff', lw=2,
            marker='o', markersize=2, alpha=0.8
        )
        ax_conv.set_yscale('log')   # 使用對數刻度以清楚呈現收斂速度
        ax_conv.set_xlabel("Iterations")
        ax_conv.set_ylabel("Max Error (Log)")
        ax_conv.set_title(f"Convergence trend analysis (N={user_iters})")
        ax_conv.grid(True, which="both", ls="-", alpha=0.3)
        st.pyplot(fig_conv)
        st.metric("最終 Max Error", f"{error_hist_user[-1]:.8f}")


# =====================================================================================================
# 分頁內容：穩定狀態
# =====================================================================================================
with tab_map["🎯 穩定狀態"]:
    st.subheader("長期分佈 (穩定狀態)")

    # 將穩定狀態向量轉為 DataFrame 並顯示表格與長條圖
    df_steady = pd.DataFrame({"位置": range(1, n_nodes + 1), "機率": steady_v})
    st.table(df_steady.style.format({"機率": "{:.4%}"}))
    st.bar_chart(df_steady.set_index("位置")["機率"])


# =====================================================================================================
# 分頁內容：計算詳情
# =====================================================================================================
with tab_map["📝 計算詳情"]:
    st.subheader("🔍 數值計算過程解剖")
    calc_mode = st.selectbox(
        "選擇計算類型",
        ["轉移矩陣元素 $P_{ij}$", "穩定狀態元素 $\\pi_i$", "矩陣乘法 $(P^2)_{ij}$"]
    )

    if calc_mode == "轉移矩陣元素 $P_{ij}$":
        c1, c2 = st.columns(2)
        with c1: row = st.number_input("選擇行 (起點 $i$)", 1, n_nodes, 1)
        with c2: col = st.number_input("選擇列 (終點 $j$)", 1, n_nodes, 2)

        # 從鄰接表查找節點 row → col 的邊權重
        weight_ij = 0.0
        for v, w in adj[row]:
            if v == col:
                weight_ij = w

        self_w = 1.0 if allow_self else 0.0
        total_w = sum([w for v, w in adj[row]]) + self_w
        res = P[row-1, col-1]   # 從矩陣直接讀取結果作為對照

        st.markdown(f'<div class="calc-box">', unsafe_allow_html=True)
        st.latex(
            f"P_{{{row},{col}}} = \\frac{{\\text{{Weight}}_{{{row} \\to {col}}}}}"
            f"{{\\sum \\text{{Weights from {row}}} + \\text{{Self-loop}}}} "
        )
        st.latex(
            f"P_{{{row},{col}}} = \\frac{{{weight_ij:.1f}}}"
            f"{{{total_w - self_w:.1f} + {self_w:.1f}}} = {res:.4f}"
        )
        st.markdown(
            f'<div class="explain-box"><strong>💡 邏輯解析：</strong><br>'
            f'這代表從{label_prefix} {row} 出發，在所有可能的選擇中，'
            f'選擇移動到{label_prefix} {col} 的權重佔比。</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    elif calc_mode == "穩定狀態元素 $\\pi_i$":
        node = st.number_input("選擇位置 $i$", 1, n_nodes, 1)

        # 計算 π_i = Σ_j (π_j × P_{j,i})，展開每一項
        sum_terms, formula_terms = [], []
        for j in range(1, n_nodes + 1):
            val = steady_v[j-1] * P[j-1, node-1]
            sum_terms.append(val)
            formula_terms.append(f"{steady_v[j-1]:.4f} \\times {P[j-1, node-1]:.4f}")

        st.markdown(f'<div class="calc-box">', unsafe_allow_html=True)
        st.latex(f"\\pi_{{{node}}} = \\sum_{{j=1}}^{{{n_nodes}}} (\\pi_{{j}} \\times P_{{j,{node}}})")
        st.latex(f"\\pi_{{{node}}} = {' + '.join(formula_terms)} = {sum(sum_terms):.4f}")
        st.markdown(
            f'<div class="explain-box"><strong>💡 邏輯解析：</strong><br>'
            f'長期來看，您處於{label_prefix} {node} 的機率，等於「所有能到達這裡的節點 $j$」'
            f'的機率 $\\pi_j$ 與轉移機率 $P_{{j,node}}$ 的乘積之總和。</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    elif calc_mode == "矩陣乘法 $(P^2)_{ij}$":
        c1, c2 = st.columns(2)
        with c1: r = st.number_input("選擇行 $i$", 1, n_nodes, 1)
        with c2: c = st.number_input("選擇列 $j$", 1, n_nodes, 1)

        # 展開 (P^2)_{r,c} = Σ_k P_{r,k} × P_{k,c} 的每一項
        terms, formula_terms = [], []
        for k in range(1, n_nodes + 1):
            val = P[r-1, k-1] * P[k-1, c-1]
            terms.append(val)
            formula_terms.append(f"{P[r-1, k-1]:.2f} \\times {P[k-1, c-1]:.2f}")

        st.markdown(f'<div class="calc-box">', unsafe_allow_html=True)
        st.latex(
            f"(P^2)_{{{r},{c}}} = \\sum_{{k=1}}^{{{n_nodes}}} "
            f"(P_{{{r},{k}}} \\times P_{{{k},{c}}})"
        )
        st.latex(
            f"(P^2)_{{{r},{c}}} = {' + '.join(formula_terms)} = {sum(terms):.4f}"
        )
        st.markdown(
            f'<div class="explain-box"><strong>💡 邏輯解析：</strong><br>'
            f'這是在計算「經過恰好 2 步」從 {label_prefix} {r} 到達 {label_prefix} {c} 的機率。</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================================================
# 分頁內容：數學原理
# =====================================================================================================
with tab_map["📐 數學原理"]:
    st.subheader("📐 數學模型與解析")

    if mode == "🐁 8格迷宮老鼠 (Mouse Maze)":
        st.markdown(
            "### 迷宮問題分析\n"
            "- **無自環限制**：$P_{ii} = 0$。\n"
            "- **多步轉移**：使用 $P^m$ 求解分佈。"
        )
    else:
        st.markdown(
            "#### 巡邏問題分析\n"
            "- **自環權重**：$w_{ii} = 1.0$。"
        )
