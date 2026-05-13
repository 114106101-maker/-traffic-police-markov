import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components
import time

# ==========================================
# 1. 視覺風格與 CSS 注入 (UI 優化)
# ==========================================
def apply_custom_style():
    st.markdown("""
        <style>
        .main { background-color: #fbfbfb; }
        .stMetric {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border: 1px solid #eee;
        }
        .stButton>button {
            width: 100%;
            border-radius: 25px;
            background-color: #007bff;
            color: white;
            font-weight: bold;
            border: none;
            padding: 10px 20px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #0056b3;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        div[data-testid="stExpander"] {
            border: none !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            background-color: white;
            border-radius: 10px;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #eee;
            border-radius: 10px 10px 0 0;
            padding: 10px 20px;
        }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 數學核心邏輯 (馬可夫鏈)
# ==========================================
def build_transition_matrix(n, edges_with_weights):
    P = np.zeros((n, n))
    adj = {i: [] for i in range(1, n + 1)}
    for u, v, w in edges_with_weights:
        if 1 <= u <= n and 1 <= v <= n:
            adj[u].append((v, w))
            adj[v].append((u, w))

    for i in range(1, n + 1):
        neighbors = adj[i]
        self_weight = 1.0 
        total_weight = sum([w for v, w in neighbors]) + self_weight
        P[i-1, i-1] = self_weight / total_weight
        for v, w in neighbors:
            P[i-1, v-1] = w / total_weight
    return P

def find_steady_state(P, threshold):
    n = P.shape[0]
    if n == 0: return np.array([]), 0, []
    v = np.ones(n) / n
    error_history = []
    iteration = 0
    while True:
        v_next = np.dot(v, P)
        error = np.max(np.abs(v_next - v))
        error_history.append(error)
        if error < threshold or iteration > 10000:
            break
        v = v_next
        iteration += 1
    return v, iteration, error_history

# ==========================================
# 3. 視覺化與動畫模組 (新增座標控制)
# ==========================================
def create_interactive_graph(n, edges_with_weights, steady_v=None, fixed_pos=None):
    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # 如果有固定座標，關閉物理模擬
    if fixed_pos:
        net.set_options('{"physics":{"enabled":false}}')
    else:
        net.barnes_hut()

    for i in range(1, n + 1):
        color = "#ADD8E6"
        if steady_v is not None and len(steady_v) >= i:
            intensity = int(steady_v[i-1] * 255 * 2)
            color = f"rgb(255, {255-min(intensity, 255)}, {255-min(intensity, 255)})"
        
        # 設置座標
        pos = fixed_pos.get(i, None) if fixed_pos else None
        x = pos[0] * 100 if pos else None
        y = pos[1] * 100 if pos else None
        
        net.add_node(i, label=f"路口 {i}", color=color, x=x, y=y, 
                     title=f"長期機率: {steady_v[i-1]:.4f}" if steady_v is not None else "")
    
    for u, v, w in edges_with_weights:
        if 1 <= u <= n and 1 <= v <= n:
            net.add_edge(u, v, value=w, title=f"權重: {w}")
    
    net.save_graph("graph.html")
    return "graph.html"

def draw_simulation_frame(n, edges, current_node, steady_v, fixed_pos=None):
    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    G.add_edges_from([(u, v) for u, v, w in edges])
    
    # 使用固定座標或隨機佈局
    pos = fixed_pos if fixed_pos else nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    node_colors = ["#FFFF00" if i == current_node else "#ADD8E6" for i in range(1, n + 1)]

    nx.draw(G, pos, with_labels=True, node_color=node_colors,
            node_size=600, edge_color="#D3D3D3", font_size=10, font_weight='bold', ax=ax)
    ax.set_title(f"Police officer's current location: intersection {current_node}", fontsize=12, fontweight='bold')
    plt.axis('off')
    return fig

# ==========================================
# 4. Streamlit 應用主界面
# ==========================================
st.set_page_config(page_title="Police Markov Station Pro", layout="wide")
apply_custom_style()
st.title("👮 交通警察值勤分析工作站")
st.markdown("本系統整合了**動態拓撲偵測、加權轉移矩陣、隨機行走動畫與穩定狀態分析**。")

# --- 側邊欄：配置中心 ---
st.sidebar.header("⚙️ 配置中心")
with st.sidebar.expander("📍 佈局與連動設定", expanded=True):
    layout_type = st.selectbox("選擇連接佈局", ["(5節點) 佈局", "3x4 網格", "自定義網格", "手動輸入 (u,v,w)"])

    n_nodes = 0
    edges_with_weights = []
    fixed_pos = None

    if layout_type == "(5節點) 佈局":
        n_nodes = 5
        # 外圈正方形: 1-2, 2-4, 4-3, 3-1
        # 內線星形: 1-5, 2-5, 3-5, 4-5
        edges_with_weights = [
            (1, 2, 1.0), (2, 4, 1.0), (4, 3, 1.0), (3, 1, 1.0),
            (1, 5, 1.0), (2, 5, 1.0), (3, 5, 1.0), (4, 5, 1.0)
        ]
        # 定義座標讓它看起來像照片 (x, y)
        fixed_pos = {
            1: (0, 1),   # 左上
            2: (1, 1),   # 右上
            3: (0, 0),   # 左下
            4: (1, 0),   # 右下
            5: (0.5, 0.5)# 中心
        }

    elif layout_type == "3x4 網格":
        n_nodes = 12
        for r in range(3):
            for c in range(4):
                u = r * 4 + c + 1
                if c < 3: edges_with_weights.append((u, u + 1, 1.0))
                if r < 2: edges_with_weights.append((u, u + 4, 1.0))

    elif layout_type == "自定義網格":
        rows = st.number_input("行數", min_value=1, value=3)
        cols = st.number_input("列數", min_value=1, value=4)
        n_nodes = rows * cols
        for r in range(rows):
            for c in range(cols):
                u = r * cols + c + 1
                if c < cols - 1: edges_with_weights.append((u, u + 1, 1.0))
                if r < rows - 1: edges_with_weights.append((u, u + cols, 1.0))

    elif layout_type == "手動輸入 (u,v,w)":
        st.caption("格式`路口,路口,權重` (例如: 1,2,1.5)")
        raw_input = st.text_area("編輯關係清單", "1,2,1.0\n2,3,1.0\n3,1,1.0")
        temp_edges = []
        current_max = 0
        for line in raw_input.split('\n'):
            if line.strip():
                try:
                    u, v, w = map(float, line.split(','))
                    temp_edges.append((int(u), int(v), w))
                    current_max = max(current_max, int(u), int(v))
                except: pass
        n_nodes = current_max if current_max >= 2 else 2
        edges_with_weights = temp_edges
        st.success(f"🎯 自動偵測路口數: {n_nodes}")

with st.sidebar.expander("📈 數學精度設定", expanded=False):
    threshold = st.number_input("收斂閾值", value=0.000001, format="%.7f")

# --- 核心計算流 ---
P = build_transition_matrix(n_nodes, edges_with_weights)
steady_v, iters, error_hist = find_steady_state(P, threshold)

# --- 頂部 KPI 儀表板 ---
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("路口規模", f"{n_nodes} 處")
m_col2.metric("迭代次數", f"{iters} 次")
m_col3.metric("系統狀態", "穩定" if iters < 10000 else "未收斂")

# --- 主功能分頁 ---
tab_graph, tab_sim, tab_matrix, tab_conv, tab_steady = st.tabs([
    "🌐 互動拓撲圖", "⏱️ 隨機行走模擬", "📊 轉移矩陣", "📉 收斂趨勢", "🎯 穩定狀態"
])

with tab_graph:
    st.subheader("路口連接視覺化")
    st.info("💡 提示：可拖拽節點。顏色越紅 $\rightarrow$ 警察長期停留機率越高。")
    graph_html = create_interactive_graph(n_nodes, edges_with_weights, steady_v, fixed_pos)
    with open(graph_html, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=550)

with tab_sim:
    st.subheader("實時隨機行走模擬")
    col_ctrl, col_map = st.columns([1, 2])

    with col_ctrl:
        st.markdown("### 🎮 控制面板")
        start_node = st.number_input("設定起點", 1, n_nodes, 1)
        sim_steps = st.slider("模擬時長 (小時)", 1, 100, 20)
        speed = st.slider("動畫速度", 0.1, 1.0, 0.3)
        run_btn = st.button("🚀 開始模擬行走")
        st.markdown("---")
        st.caption("地圖說明：亮黃色節點為警察當前位置。")
    with col_map:
        map_placeholder = st.empty()
        status_placeholder = st.empty()
        if run_btn:
            current = start_node
            path = [current]
            for i in range(sim_steps):
                fig = draw_simulation_frame(n_nodes, edges_with_weights, current, steady_v, fixed_pos)
                map_placeholder.pyplot(fig)
                status_placeholder.markdown(f"**狀態**：第 {i+1} 小時 $\rightarrow$ 位於 **路口 {current}**")

                probs = P[current-1, :]
                probs = probs / np.sum(probs)
                current = np.random.choice(range(1, n_nodes + 1), p=probs)
                path.append(current)
                time.sleep(speed)
                plt.close(fig)
            st.success(f"模擬結束。完整路徑: {' $\rightarrow$ '.join(map(str, path))}")

with tab_matrix:
    st.subheader("轉移矩陣 $P$")
    with st.expander("展開查看矩陣詳細數據", expanded=False):
        df_P = pd.DataFrame(P, index=[f"路口 {i+1}" for i in range(n_nodes)],
                            columns=[f"路口 {i+1}" for i in range(n_nodes)])
        st.dataframe(df_P.style.format("{:.4f}"))
        st.download_button("📥 下載矩陣 CSV", df_P.to_csv().encode('utf-8'), "matrix.csv")

with tab_conv:
    st.subheader("收斂過程分析")
    if len(error_hist) > 0:
        fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
        ax_conv.plot(error_hist, color='#007bff', lw=2)
        ax_conv.set_yscale('log')
        ax_conv.set_xlabel("Number of iterations")
        ax_conv.set_ylabel("Maximum error (Log)")
        ax_conv.grid(True, alpha=0.3)
        st.pyplot(fig_conv)

with tab_steady:
    st.subheader("長期值勤分佈")
    c_t1, c_t2 = st.columns([1, 1])
    with c_t1:
        df_steady = pd.DataFrame({"路口": [f"路口 {i+1}" for i in range(n_nodes)], "機率": steady_v})
        st.table(df_steady.style.format({"機率": "{:.4%}"}))
        st.download_button("📥 下載分佈 CSV", df_steady.to_csv(index=False).encode('utf-8'), "steady_state.csv")
    with c_t2:
        st.bar_chart(df_steady.set_index("路口")["機率"])
