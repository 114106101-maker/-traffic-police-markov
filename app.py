import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components
import time

# ==========================================================================================
# 1. 視覺風格與 CSS
# ===========================================================================================
def apply_custom_style():
    st.markdown("""
        <style>
        .main { background-color: #fbfbfb; }
        .stMetric { background-color: #ffffff !important; padding: 15px !important; border-radius: 12px !important; box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important; border: 1px solid #eee !important; }
        .stButton>button { width: 100%; border-radius: 12px !important; background: linear-gradient(45deg, #007bff, #0056b3) !important; color: white !important; font-weight: 600 !important; border: none !important; padding: 12px 20px !important; box-shadow: 0 4px 10px rgba(0,123,255,0.3) !important; }
        .stButton>button:hover { background: linear-gradient(45deg, #0056b3, #004085) !important; }
        div.stButton > div.st-emotion-cache-micr9v > button { background: linear-gradient(45deg, #ff4b2b, #ff416c) !important; box-shadow: 0 4px 10px rgba(255,75,43,0.3) !important; }
        div[data-testid="stExpander"] { border: none !important; box-shadow: 0 2px 8px rgba(0,0,0,0.05); background-color: white; border-radius: 10px; }
        .mode-selector { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 25px; border-radius: 20px; border: 1px solid #dee2e6; margin-bottom: 30px; text-align: center; }
        .calc-box { background-color: #ffffff; padding: 20px; border-left: 6px solid #007bff; border-radius: 8px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
        .explain-box { background-color: #fff9db; padding: 15px; border-radius: 8px; border: 1px solid #ffe066; margin-top: 10px; color: #856404; font-size: 0.95rem; }
        .path-box { background-color: #f1f3f5; padding: 15px; border-radius: 12px; border: 2px dashed #adb5bd; font-family: 'Consolas', monospace; color: #495057; }
        .stTabs [data-baseweb="tab"] { background-color: #fff; border: 1px solid #dee2e6; border-radius: 8px 8px 0 0; padding: 10px 20px; }
        </style>
    """, unsafe_allow_html=True)

# ==========================================================================================
# 2. 數學核心邏輯
# ==========================================================================================
def build_transition_matrix(n, edges_with_weights, allow_self_loop=True):
    P = np.zeros((n, n))
    adj = {i: [] for i in range(1, n + 1)}
    for u, v, w in edges_with_weights:
        if 1 <= u <= n and 1 <= v <= n:
            adj[u].append((v, w))
            adj[v].append((u, w))

    self_weight = 1.0 if allow_self_loop else 0.0
    for i in range(1, n + 1):
        neighbors = adj[i]
        total_weight = sum([w for v, w in neighbors]) + self_weight
        if total_weight == 0: continue
        P[i-1, i-1] = self_weight / total_weight
        for v, w in neighbors:
            P[i-1, v-1] = w / total_weight
    return P, adj

def find_steady_state(P, threshold):
    """計算直到收斂的穩定狀態 """
    n = P.shape[0]
    if n == 0: return np.array([]), 0, []
    v = np.zeros(n)
    v[0] = 1.0 # 使用點分佈增加收斂過程的可視化長度
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

def get_convergence_history_fixed(P, max_iters):
    """精確執行 N 次迭代並回傳誤差紀錄，不因閾值提前停止"""
    n = P.shape[0]
    if n == 0: return []
    v = np.zeros(n)
    v[0] = 1.0
    error_history = []
    for i in range(max_iters):
        v_next = np.dot(v, P)
        error = np.max(np.abs(v_next - v))
        error_history.append(error)
        v = v_next
    return error_history

# ==========================================================================================
# 3. 視覺化模組 
# ==========================================================================================
def create_interactive_graph(n, edges_with_weights, steady_v=None, fixed_pos=None, label_prefix="位置"):
    net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
    if fixed_pos:
        net.set_options('{"physics":{"enabled":false}, "nodes":{"font":{"size":16}}}')
    else:
        net.barnes_hut()
        
    for i in range(1, n + 1):
        color = "#ADD8E6"
        if steady_v is not None and len(steady_v) >= i:
            intensity = int(steady_v[i-1] * 255)
            color = f"rgb(255, {255-min(intensity, 255)}, {255-min(intensity, 255)})"
        if fixed_pos:
            pos = fixed_pos.get(i, (0,0))
            net.add_node(i, label=f"{label_prefix} {i}", color=color, x=pos[0], y=pos[1],
                         title=f"機率: {steady_v[i-1]:.4f}" if steady_v is not None else "")
        else:
            net.add_node(i, label=f"{label_prefix} {i}", color=color,
                         title=f"機率: {steady_v[i-1]:.4f}" if steady_v is not None else "")
    for u, v, w in edges_with_weights:
        net.add_edge(u, v, value=w)
    net.save_graph("graph.html")
    return "graph.html"

def draw_simulation_frame(n, edges, current_node, steady_v, fixed_pos=None):
    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    G.add_edges_from([(u, v) for u, v, w in edges])
    pos = fixed_pos if fixed_pos else nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    node_colors = ["#FFFF00" if i == current_node else "#ADD8E6" for i in range(1, n + 1)]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=600, edge_color="#D3D3D3", font_size=10, font_weight='bold', ax=ax)
    plt.axis('off')
    return fig

# ==========================================================================================
# 4. Streamlit 主界面
# ==========================================================================================
st.set_page_config(page_title="Markov Analysis Suite Pro", layout="wide")
apply_custom_style()

st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #2c3e50;'>🛠️ 系統分析模式選擇</h2>", unsafe_allow_html=True)
mode = st.radio("請選擇您要分析的對象：", ["👮 交通警察巡邏 (Police Patrol)", "🐁 8格迷宮老鼠 (Mouse Maze)"], horizontal=True, label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

INITIAL_TOPO = {
    'n_nodes': 5, 'edges': [(1, 2, 1.0), (2, 4, 1.0), (4, 3, 1.0), (3, 1, 1.0), (1, 5, 1.0), (2, 5, 1.0), (3, 5, 1.0), (4, 5, 1.0)],
    'fixed_pos': {1: (0, 100), 2: (100, 100), 3: (0, 0), 4: (100, 0), 5: (50, 50)}, 'allow_self_loop': True
}

if 'topo_data' not in st.session_state:
    st.session_state.topo_data = INITIAL_TOPO.copy()

st.sidebar.header("⚙️ 配置中心")
if mode == "👮 交通警察巡邏 (Police Patrol)":
    with st.sidebar.expander("📍 佈局設定", expanded=True):
        layout_type = st.selectbox("選擇佈局", ["(5節點)佈局", "3x4 網格", "自定義網格", "手動輸入"])
        if layout_type == "(5節點)佈局":
            st.session_state.topo_data = INITIAL_TOPO.copy()
        elif layout_type == "3x4 網格":
            edges = []
            for r in range(3):
                for c in range(4):
                    u = r * 4 + c + 1
                    if c < 3: edges.append((u, u + 1, 1.0))
                    if r < 2: edges.append((u, u + 4, 1.0))
            st.session_state.topo_data = {'n_nodes': 12, 'edges': edges, 'fixed_pos': None, 'allow_self_loop': True}
        elif layout_type == "自定義網格":
            rows, cols = st.number_input("行數", 1, 10, 3), st.number_input("列數", 1, 10, 4)
            edges = []
            for r in range(rows):
                for c in range(cols):
                    u = r * cols + c + 1
                    if c < cols - 1: edges.append((u, u + 1, 1.0))
                    if r < rows - 1: edges.append((u, u + cols, 1.0))
            st.session_state.topo_data = {'n_nodes': rows*cols, 'edges': edges, 'fixed_pos': None, 'allow_self_loop': True}
        elif layout_type == "手動輸入":
            raw_input = st.text_area("編輯關係清單 (u,v,w)", "1,2,1.0\n2,3,1.0\n3,1,1.0")
            temp_edges, curr_max = [], 0
            for line in raw_input.split('\n'):
                if line.strip():
                    try:
                        u, v, w = map(float, line.split(','))
                        temp_edges.append((int(u), int(v), w))
                        curr_max = max(curr_max, int(u), int(v))
                    except: pass
            st.session_state.topo_data = {'n_nodes': max(curr_max, 2), 'edges': temp_edges, 'fixed_pos': None, 'allow_self_loop': True}
elif mode == "🐁 8格迷宮老鼠 (Mouse Maze)":
    with st.sidebar.expander("📍 迷宮設定", expanded=True):
        st.write("此模式為固定線性迷宮")
        edges = [(i, i+1, 1.0) for i in range(1, 8)]
        fixed_pos = {i: (i * 100, 0) for i in range(1, 9)}
        st.session_state.topo_data = {'n_nodes': 8, 'edges': edges, 'fixed_pos': fixed_pos, 'allow_self_loop': False}

with st.sidebar.expander("📈 數學精度設定", expanded=False):
    threshold = st.number_input("收斂閾值", value=0.000001, format="%.7f")

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ 系統管理")
if st.sidebar.button("🔄 一鍵重置所有配置", key="reset_btn"):
    st.session_state.topo_data = INITIAL_TOPO.copy()
    st.rerun()

n_nodes = st.session_state.topo_data['n_nodes']
edges_with_weights = st.session_state.topo_data['edges']
fixed_pos = st.session_state.topo_data['fixed_pos']
allow_self = st.session_state.topo_data['allow_self_loop']
label_prefix = "路口" if mode == "👮 交通警察巡邏 (Police Patrol)" else "位置"

P, adj = build_transition_matrix(n_nodes, edges_with_weights, allow_self_loop=allow_self)
steady_v, iters, error_hist_auto = find_steady_state(P, threshold)

m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("路口/位置規模", f"{n_nodes} 處")
m_col2.metric("自動收斂次數", f"{iters} 次")
m_col3.metric("系統狀態", "穩定" if iters < 10000 else "未收斂")

tabs_list = ["🌐 互動拓撲圖", "⏱️ 隨機行走模擬", "📊 轉移矩陣", "📉 收斂趨勢", "🎯 穩定狀態", "📝 計算詳情", "📐 數學原理"]
if mode == "🐁 8格迷宮老鼠 (Mouse Maze)":
    tabs_list.insert(3, "🧮 矩陣運算分析")

tab_objs = st.tabs(tabs_list)
tab_map = {name: tab for name, tab in zip(tabs_list, tab_objs)}

with tab_map["🌐 互動拓撲圖"]:
    st.subheader(f"{label_prefix}連接視覺化")
    graph_html = create_interactive_graph(n_nodes, edges_with_weights, steady_v, fixed_pos, label_prefix)
    with open(graph_html, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=550)

with tab_map["⏱️ 隨機行走模擬"]:
    st.subheader("實時行走模擬")
    col_ctrl, col_map = st.columns([1, 2])
    with col_ctrl:
        start_node = st.number_input("設定起點", 1, n_nodes, 1)
        sim_steps = st.slider("模擬時長", 1, 100, 20)
        speed = st.slider("動畫速度", 0.1, 1.0, 0.3)
        run_btn = st.button("🚀 開始模擬")
    with col_map:
        map_placeholder = st.empty()
        status_placeholder = st.empty()
        path_placeholder = st.empty()
        if run_btn:
            current = start_node
            visited_path = [current]
            for i in range(sim_steps):
                fig = draw_simulation_frame(n_nodes, edges_with_weights, current, steady_v, fixed_pos)
                map_placeholder.pyplot(fig)
                status_placeholder.markdown(f"**狀態**：第 {i+1} 步 $\rightarrow$ 位於 **{label_prefix} {current}**")
                path_str = " $\rightarrow$ ".join(map(str, visited_path))
                path_placeholder.markdown(f'<div class="path-box"><strong style="color:#007bff;">🚶 實時路徑紀錄：</strong><br>{path_str}</div>', unsafe_allow_html=True)
                probs = P[current-1, :]
                current = np.random.choice(range(1, n_nodes + 1), p=probs/np.sum(probs))
                visited_path.append(current)
                time.sleep(speed)
                plt.close(fig)
            st.success(f"✅ 模擬結束。完整路徑：{' $\rightarrow$ '.join(map(str, visited_path))}")

if "🧮 矩陣運算分析" in tab_map:
    with tab_map["🧮 矩陣運算分析"]:
        st.subheader("🎯 特定步數機率計算")
        col_input, col_res = st.columns([1, 2])
        with col_input:
            start_node_m = st.number_input("設定起始位置 $v^{(0)}$", 1, n_nodes, 1)
            target_node_m = st.number_input("設定目標位置", 1, n_nodes, 5)
            steps_m = st.number_input("計算步數 $m$", 1, 100, 2)
        with col_res:
            Pm = np.linalg.matrix_power(P, steps_m)
            v0 = np.zeros(n_nodes); v0[start_node_m-1] = 1.0
            vm = np.dot(v0, Pm)
            st.metric(f"經過 {steps_m} 步後，在位置 {target_node_m} 的機率", f"{vm[target_node_m-1]:.4%}")
            df_vm = pd.DataFrame({"位置": range(1, n_nodes+1), "機率": vm})
            st.bar_chart(df_vm.set_index("位置")["機率"])

with tab_map["📊 轉移矩陣"]:
    st.subheader("轉移矩陣 $P$ (行加總為 1)")
    row_sums = P.sum(axis=1)
    df_P = pd.DataFrame(P, index=[f"{label_prefix} {i+1}" for i in range(n_nodes)],
                        columns=[f"{label_prefix} {i+1}" for i in range(n_nodes)])
    df_P['行加總 (Sum)'] = row_sums
    st.dataframe(df_P.style.format("{:.4f}"))
    st.caption("💡 驗證：每一行的『行加總』應精確等於 1.0000，代表機率分佈完整。")

with tab_map["📉 收斂趨勢"]:
    st.subheader("收斂過程分析")
    
    # --- 【新增功能：可調整的迭代次數】 ---
    col_ctrl, col_info = st.columns([1, 1])
    with col_ctrl:
        user_iters = st.slider("調整迭代次數 (Iterations)", 1, 500, 100)
    with col_info:
        st.info(f"目前設定執行 {user_iters} 次迭代。您可以觀察 Max Error 如何隨著次數增加而下降。")
    
    # 使用新函數計算誤差，確保不會因為 threshold 而提前停止
    error_hist_user = get_convergence_history_fixed(P, user_iters)
    
    if len(error_hist_user) > 0:
        fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
        ax_conv.plot(error_hist_user, color='#007bff', lw=2, marker='o', markersize=2, alpha=0.8)
        ax_conv.set_yscale('log')
        ax_conv.set_xlabel("Iterations")
        ax_conv.set_ylabel("Max Error (Log)")
        ax_conv.set_title(f"Convergence trend analysis (N={user_iters})")
        ax_conv.grid(True, which="both", ls="-", alpha=0.3)
        st.pyplot(fig_conv)
        st.metric("最終 Max Error", f"{error_hist_user[-1]:.8f}")

with tab_map["🎯 穩定狀態"]:
    st.subheader("長期分佈 (穩定狀態)")
    df_steady = pd.DataFrame({"位置": range(1, n_nodes+1), "機率": steady_v})
    st.table(df_steady.style.format({"機率": "{:.4%}"}))
    st.bar_chart(df_steady.set_index("位置")["機率"])

with tab_map["📝 計算詳情"]:
    st.subheader("🔍 數值計算過程解剖")
    calc_mode = st.selectbox("選擇計算類型", ["轉移矩陣元素 $P_{ij}$", "穩定狀態元素 $\\pi_i$", "矩陣乘法 $(P^2)_{ij}$"])
    if calc_mode == "轉移矩陣元素 $P_{ij}$":
        c1, c2 = st.columns(2)
        with c1: row = st.number_input("選擇行 (起點 $i$)", 1, n_nodes, 1)
        with c2: col = st.number_input("選擇列 (終點 $j$)", 1, n_nodes, 2)
        weight_ij = 0.0
        for v, w in adj[row]:
            if v == col: weight_ij = w
        self_w = 1.0 if allow_self else 0.0
        total_w = sum([w for v, w in adj[row]]) + self_w
        res = P[row-1, col-1]
        st.markdown(f'<div class="calc-box">', unsafe_allow_html=True)
        st.latex(f"P_{{{row},{col}}} = \\frac{{\\text{{Weight}}_{{{row} \\to {col}}}}}{{\\sum \\text{{Weights from {row}}} + \\text{{Self-loop}}}} ")
        st.latex(f"P_{{{row},{col}}} = \\frac{{{weight_ij:.1f}}}{{{total_w - self_w:.1f} + {self_w:.1f}}} = {res:.4f}")
        st.markdown(f'<div class="explain-box"><strong>💡 邏輯解析：</strong><br>這代表從{label_prefix} {row} 出發，在所有可能的選擇中，選擇移動到{label_prefix} {col} 的權重佔比。</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    elif calc_mode == "穩定狀態元素 $\\pi_i$":
        node = st.number_input("選擇位置 $i$", 1, n_nodes, 1)
        sum_terms, formula_terms = [], []
        for j in range(1, n_nodes + 1):
            val = steady_v[j-1] * P[j-1, node-1]
            sum_terms.append(val)
            formula_terms.append(f"{steady_v[j-1]:.4f} \\times {P[j-1, node-1]:.4f}")
        st.markdown(f'<div class="calc-box">', unsafe_allow_html=True)
        st.latex(f"\\pi_{{{node}}} = \\sum_{{j=1}}^{{{n_nodes}}} (\\pi_{{j}} \\times P_{{j,{node}}})")
        st.latex(f"\\pi_{{{node}}} = {' + '.join(formula_terms)} = {sum(sum_terms):.4f}")
        st.markdown(f'<div class="explain-box"><strong>💡 邏輯解析：</strong><br>長期來看，您處於{label_prefix} {node} 的機率，等於「所有能到達這裡的節點 $j$」的機率 $\\pi_j$ 與轉移機率 $P_{j,node}$ 的乘積之總和。</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    elif calc_mode == "矩陣乘法 $(P^2)_{ij}$":
        c1, c2 = st.columns(2)
        with c1: r = st.number_input("選擇行 $i$", 1, n_nodes, 1)
        with c2: c = st.number_input("選擇列 $j$", 1, n_nodes, 1)
        terms, formula_terms = [], []
        for k in range(1, n_nodes + 1):
            val = P[r-1, k-1] * P[k-1, c-1]
            terms.append(val)
            formula_terms.append(f"{P[r-1, k-1]:.2f} \\times {P[k-1, c-1]:.2f}")
        st.markdown(f'<div class="calc-box">', unsafe_allow_html=True)
        st.latex(f"(P^2)_{{{r},{c}}} = \\sum_{{k=1}}^{{{n_nodes}}} (P_{{{r},{k}}} \\times P_{{{k},{c}}})")
        st.latex(f"(P^2)_{{{r},{c}}} = {' + '.join(formula_terms)} = {sum(terms):.4f}")
        st.markdown(f'<div class="explain-box"><strong>💡 邏輯解析：</strong><br>這是在計算「經過恰好 2 步」從 {label_prefix} {r} 到達 {label_prefix} {c} 的機率。</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab_map["📐 數學原理"]:
    st.subheader("📐 數學模型與解析")
    if mode == "🐁 8格迷宮老鼠 (Mouse Maze)":
        st.markdown("#### 迷宮問題分析\n- **無自環限制**：$P_{ii} = 0$。\n- **多步轉移**：使用 $P^m$ 求解分佈。")
    else:
        st.markdown("#### 巡邏問題分析\n- **自環權重**：$w_{ii} = 1.0$。")
