import streamlit as st
import numpy as np
import pandas as pd

# ----------------- 1. 核心計算邏輯 -----------------

def build_transition_matrix(n, edges):
    """
    根據節點數和邊清單建立轉移矩陣
    edges: list of tuples [(1,2), (2,3), ...]
    """
    P = np.zeros((n, n))
    # 建立鄰接表
    adj = {i: [] for i in range(1, n + 1)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(1, n + 1):
        neighbors = adj[i]
        # 機率 = 1 / (鄰居數 + 1) -> 包含留在原處
        prob = 1.0 / (len(neighbors) + 1)
        P[i-1, i-1] = prob # 留在原處
        for nb in neighbors:
            P[i-1, nb-1] = prob
            
    return P

def get_prob_after_steps(P, start_node, end_node, steps):
    n = P.shape[0]
    v0 = np.zeros(n)
    v0[start_node - 1] = 1.0
    Pn = np.linalg.matrix_power(P, steps)
    vn = np.dot(v0, Pn)
    return vn[end_node - 1]

def find_steady_state(P, threshold):
    n = P.shape[0]
    v = np.ones(n) / n 
    iteration = 0
    while True:
        v_next = np.dot(v, P)
        if np.max(np.abs(v_next - v)) < threshold:
            break
        v = v_next
        iteration += 1
        if iteration > 10000: # 防止死循環
            break
    return v, iteration

# ----------------- 2. Streamlit 介面 -----------------

st.set_page_config(page_title="通用馬可夫鏈分析器", layout="wide")

st.title("👮 交通警察值勤路口分析系統 (自定義版)")
st.markdown("您可以自定義路口數量、連接方式及計算參數。")

# --- 側邊欄：參數輸入 ---
st.sidebar.header("⚙️ 參數設定")

# A. 路口數量
n_nodes = st.sidebar.number_input("路口總數 (n)", min_value=2, max_value=50, value=12)

# B. 連接方式
layout_type = st.sidebar.selectbox("選擇路口連接佈局", ["3x4 網格 (預設)", "自定義網格 (R x C)", "手動輸入連接清單"])

edges = []
if layout_type == "3x4 網格 (預設)":
    # 快速生成 3x4 網格
    for r in range(3):
        for c in range(4):
            u = r * 4 + c + 1
            if c < 3: edges.append((u, u + 1)) # 右連
            if r < 2: edges.append((u, u + 4)) # 下連

elif layout_type == "自定義網格 (R x C)":
    rows = st.sidebar.number_input("行數 (Rows)", min_value=1, max_value=10, value=3)
    cols = st.sidebar.number_input("列數 (Cols)", min_value=1, max_value=10, value=4)
    if rows * cols != n_nodes:
        st.sidebar.error(f"⚠️ 警告：{rows}x{cols} 不等於您設定的路口總數 {n_nodes}")
    else:
        for r in range(rows):
            for c in range(cols):
                u = r * cols + c + 1
                if c < cols - 1: edges.append((u, u + 1))
                if r < rows - 1: edges.append((u, u + cols))

elif layout_type == "手動輸入連接清單":
    st.sidebar.info("格式：每行一組，例如 `1,2` 代表路口 1 與 2 相連")
    raw_input = st.sidebar.text_area("輸入連接關係 (u,v)", "1,2\n2,3\n3,4\n4,1\n1,5")
    for line in raw_input.split('\n'):
        if line.strip():
            try:
                u, v = map(int, line.split(','))
                if 1 <= u <= n_nodes and 1 <= v <= n_nodes:
                    edges.append((u, v))
                else:
                    st.sidebar.warning(f"路口編號 {u} 或 {v} 超出範圍 (1-{n_nodes})")
            except:
                st.sidebar.error(f"格式錯誤: {line}")

# C. 精度設定
threshold = st.sidebar.number_input("穩定狀態閾值", value=0.000001, format="%.7f")

# --- 主界面計算 ---

# 建立矩陣
P = build_transition_matrix(n_nodes, edges)

tab1, tab2, tab3 = st.tabs(["📊 轉移矩陣", "⏱️ 機率預測", "🎯 穩定狀態"])

with tab1:
    st.subheader("轉移矩陣 $P$")
    df_P = pd.DataFrame(P, index=[f"路口 {i+1}" for i in range(n_nodes)], 
                        columns=[f"路口 {i+1}" for i in range(n_nodes)])
    st.dataframe(df_P.style.format("{:.4f}"))
    st.caption("說明：$P_{ij}$ 表示從路口 $i$ 移動到路口 $j$ 的機率。")

with tab2:
    st.subheader("特定時間後的機率")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_n = st.number_input("起始位置", min_value=1, max_value=n_nodes, value=1)
    with col2:
        end_n = st.number_input("目標位置", min_value=1, max_value=n_nodes, value=5 if n_nodes >= 5 else 2)
    with col3:
        steps = st.number_input("時間 (小時)", min_value=1, max_value=1000, value=10)
    
    if st.button("計算機率"):
        prob = get_prob_after_steps(P, start_n, end_n, steps)
        st.success(f"從路口 {start_n} 出發，經過 {steps} 小時後在路口 {end_n} 的機率為: **{prob:.6f}**")

with tab3:
    st.subheader("長期值勤分佈 (穩定狀態)")
    steady_v, iters = find_steady_state(P, threshold)
    st.write(f"系統在經過 {iters} 次迭代後達到穩定狀態 (誤差 < {threshold})")
    
    df_steady = pd.DataFrame({
        "路口": [f"路口 {i+1}" for i in range(n_nodes)],
        "長期值勤比例": steady_v
    })
    st.table(df_steady.style.format({"長期值勤比例": "{:.4%}"}))
    
    # 簡單的可視化
    st.bar_chart(df_steady.set_index("路口")["長期值勤比例"])
