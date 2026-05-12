import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components
import random
import time

# ================== 核心邏輯模組 ==================

def build_transition_matrix(n, edges_with_weights):
    """
    n: 路口總數
    edges_with_weights: list of tuples (u, v, weight)
    """
    P = np.zeros((n, n))
    # 建立鄰接表
    adj = {i: [] for i in range(1, n + 1)}
    
    for u, v, w in edges_with_weights:
        # 【修正點 1】增加邊界檢查，防止 KeyError
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

def run_simulation(P, start_node, steps):
    n = P.shape[0]
    current = start_node
    path = [current]
    for _ in range(steps):
        probs = P[current-1, :]
        # 確保機率總和為 1 (避免浮點數精度問題)
        probs = probs / np.sum(probs)
        current = np.random.choice(range(1, n + 1), p=probs)
        path.append(current)
    return path

# ================= 視覺化模組 =================

def create_interactive_graph(n, edges_with_weights, steady_v=None):
    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
    net.barnes_hut()
    
    for i in range(1, n + 1):
        color = "#ADD8E6" 
        if steady_v is not None:
            # 顏色深度映射
            intensity = int(steady_v[i-1] * 255 * 2) 
            color = f"rgb(255, {255-min(intensity, 255)}, {255-min(intensity, 255)})"
        
        net.add_node(i, label=f"路口 {i}", color=color, 
                     title=f"穩定機率: {steady_v[i-1]:.4f}" if steady_v is not None else "")

    for u, v, w in edges_with_weights:
        if 1 <= u <= n and 1 <= v <= n: # 【修正點 2】防止繪圖時報錯
            net.add_edge(u, v, value=w, title=f"權重: {w}")

    net.save_graph("graph.html")
    return "graph.html"

# ================= Streamlit 介面 =================

st.set_page_config(page_title="馬可夫鏈分析工作站", layout="wide")

st.title("👮 交通警察值勤分析工作站 (Pro)")
st.markdown("整合了**互動圖形、加權路徑、隨機模擬與收斂分析**。")

# --- 側邊欄 ---
st.sidebar.header("⚙️ 系統配置")
n_nodes = st.sidebar.number_input("路口總數 (n)", min_value=2, max_value=50, value=12)

layout_type = st.sidebar.selectbox("連接佈局", ["3x4 網格", "自定義網格", "手動輸入 (u,v,w)"])

edges_with_weights = []
if layout_type == "3x4 網格":
    for r in range(3):
        for c in range(4):
            u = r * 4 + c + 1
            if c < 3: edges_with_weights.append((u, u + 1, 1.0))
            if r < 2: edges_with_weights.append((u, u + 4, 1.0))
elif layout_type == "自定義網格":
    rows = st.sidebar.number_input("行數", value=3)
    cols = st.sidebar.number_input("列數", value=4)
    if rows * cols == n_nodes:
        for r in range(rows):
            for c in range(cols):
                u = r * cols + c + 1
                if c < cols - 1: edges_with_weights.append((u, u + 1, 1.0))
                if r < rows - 1: edges_with_weights.append((u, u + cols, 1.0))
    else:
        st.sidebar.warning(f"⚠️ 行x列({rows*cols}) 必須等於總數({n_nodes})")
elif layout_type == "手動輸入 (u,v,w)":
    st.sidebar.info("格式：`路口1,路口2,權重` (例如: 1,2,1.5)")
    raw_input = st.sidebar.text_area("輸入關係", "1,2,1.0\n2,3,2.0\n3,4,1.0\n4,1,1.0\n1,5,0.5")
    for line in raw_input.split('\n'):
        if line.strip():
            try:
                parts = line.split(',')
                u, v, w = float(parts[0]), float(parts[1]), float(parts[2])
                # 【修正點 3】輸入端直接攔截超出範圍的編號
                if 1 <= u <= n_nodes and 1 <= v <= n_nodes:
                    edges_with_weights.append((int(u), int(v), w))
                else:
                    st.sidebar.error(f"❌ 錯誤：路口 {int(u)} 或 {int(v)} 超出了總數 {n_nodes}")
            except Exception as e:
                st.sidebar.error(f"❌ 格式錯誤: {line}")

threshold = st.sidebar.number_input("收斂閾值", value=0.000001, format="%.7f")

# --- 核心計算 ---
# 只要 n_nodes 和 edges_with_weights 正確，這裡就不會再崩潰
P = build_transition_matrix(n_nodes, edges_with_weights)
steady_v, iters, error_hist = find_steady_state(P, threshold)

# --- 分頁介面 ---
tab_graph, tab_sim, tab_matrix, tab_conv, tab_steady = st.tabs([
    "🌐 互動拓撲圖", "⏱️ 隨機行走模擬", "📊 轉移矩陣", "📉 收斂分析", "🎯 穩定狀態"
])

with tab_graph:
    st.subheader("互動式路口連接圖")
    st.info("💡 提示：您可以拖拽節點。顏色越紅 $\rightarrow$ 長期值勤機率越高。")
    graph_html = create_interactive_graph(n_nodes, edges_with_weights, steady_v)
    with open(graph_html, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=550)

with tab_sim:
    st.subheader("警察隨機行走模擬")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1: start_node = st.number_input("起始路口", 1, n_nodes, 1)
    with col_s2: sim_steps = st.slider("模擬時間 (小時)", 1, 100, 20)
    with col_s3: speed = st.slider("動畫速度", 0.1, 1.0, 0.3)
    
    if st.button("開始模擬行走"):
        path = run_simulation(P, start_node, sim_steps)
        st.write(f"🚶 模擬路徑: {' $\rightarrow$ '.join(map(str, path))}")
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(len(path)):
            status_text.text(f"目前位置: 路口 {path[i]}")
            progress_bar.progress((i + 1) / len(path))
            time.sleep(speed)
        st.success("模擬完成！")

with tab_matrix:
    st.subheader("轉移矩陣 $P$")
    df_P = pd.DataFrame(P, index=[f"路口 {i+1}" for i in range(n_nodes)], 
                        columns=[f"路口 {i+1}" for i in range(n_nodes)])
    st.dataframe(df_P.style.format("{:.4f}"))
    csv = df_P.to_csv().encode('utf-8')
    st.download_button("📥 下載矩陣為 CSV", csv, "transition_matrix.csv", "text/csv")

with tab_conv:
    st.subheader("迭代收斂過程")
    st.write(f"達到穩定狀態所需迭代次數: {iters}")
    fig_conv, ax_conv = plt.subplots()
    ax_conv.plot(error_hist, color='blue', lw=2)
    ax_conv.set_yscale('log') 
    ax_conv.set_xlabel("迭代次數")
    ax_conv.set_ylabel("最大誤差 (Log Scale)")
    ax_conv.grid(True, which="both", ls="-", alpha=0.5)
    st.pyplot(fig_conv)

with tab_steady:
    st.subheader("長期值勤分佈")
    df_steady = pd.DataFrame({"路口": [f"路口 {i+1}" for i in range(n_nodes)], "機率": steady_v})
    st.table(df_steady.style.format({"機率": "{:.4%}"}))
    st.bar_chart(df_steady.set_index("路口")["機率"])
    csv_steady = df_steady.to_csv(index=False).encode('utf-8')
    st.download_button("📥 下載穩定狀態為 CSV", csv_steady, "steady_state.csv", "text/csv")
