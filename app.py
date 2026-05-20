import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components
import time
import json

# ======================================================================================================
# 1. 視覺風格與 CSS 定義
# ======================================================================================================
def apply_custom_style() -> None:
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e4e7eb 100%); }
        .glass-card {
            background: rgba(255, 255, 255, 0.7) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 24px !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07) !important;
            padding: 25px !important;
            margin-bottom: 20px !important;
            color: #1d1d1f;
        }
        .stMetric {
            background: rgba(255, 255, 255, 0.8) !important;
            padding: 20px !important;
            border-radius: 20px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.03) !important;
            border: none !important;
        }
        .stButton>button {
            width: 100%; border-radius: 16px !important;
            background: linear-gradient(180deg, #007AFF, #005BBF) !important;
            color: white !important; font-weight: 600 !important;
            border: none !important; padding: 14px 20px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(0, 122, 255, 0.3) !important;
        }
        .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0, 122, 255, 0.4) !important; }
        div.stButton > div.st-emotion-cache-micr9v > button {
            background: linear-gradient(180deg, #FF3B30, #D70015) !important;
            box-shadow: 0 4px 15px rgba(255, 59, 48, 0.3) !important;
        }
        .mode-selector {
            background: rgba(255, 255, 255, 0.5); backdrop-filter: blur(15px);
            padding: 30px; border-radius: 30px; border: 1px solid rgba(255, 255, 255, 0.4);
            margin-bottom: 30px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        }
        .calc-box {
            background: rgba(255, 255, 255, 0.9); padding: 20px; border-radius: 20px;
            border-left: 8px solid #007AFF; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin: 15px 0;
        }
        .explain-box {
            background-color: #fffbe6; padding: 15px; border-radius: 16px;
            border: 1px solid #ffe58f; margin-top: 10px; color: #856404; font-size: 0.9rem;
        }
        .path-box {
            background: rgba(242, 242, 247, 0.8); padding: 15px; border-radius: 16px;
            border: 1px solid #d1d1d6; font-family: 'SF Mono', monospace; color: #3a3a3c;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important; border: none !important;
            border-radius: 12px 12px 0 0 !important; padding: 10px 20px !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(255, 255, 255, 0.8) !important;
            border-bottom: 3px solid #007AFF !important; font-weight: bold !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ======================================================================================================
# 2. 數學核心邏輯
# ======================================================================================================
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
    n = P.shape[0]
    if n == 0: return np.array([]), 0, []
    v = np.zeros(n); v[0] = 1.0
    error_history = []
    iteration = 0
    while True:
        v_next = np.dot(v, P)
        error = np.max(np.abs(v_next - v))
        error_history.append(error)
        if error < threshold or iteration > 10000: break
        v = v_next
        iteration += 1
    return v, iteration, error_history

def get_convergence_history_fixed(P, max_iters):
    n = P.shape[0]
    if n == 0: return []
    v = np.zeros(n); v[0] = 1.0
    error_history = []
    for i in range(max_iters):
        v_next = np.dot(v, P)
        error = np.max(np.abs(v_next - v))
        error_history.append(error)
        v = v_next
    return error_history

# ======================================================================================================
# 3. 視覺化與動畫組件
# ======================================================================================================
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
            pos = fixed_pos.get(i, (0, 0))
            net.add_node(i, label=f"{label_prefix} {i}", color=color, x=pos[0], y=pos[1],
                         title=f"機率: {steady_v[i-1]:.4f}" if steady_v is not None else "")
        else:
            net.add_node(i, label=f"{label_prefix} {i}", color=color,
                         title=f"機率: {steady_v[i-1]:.4f}" if steady_v is not None else "")
    for u, v, w in edges_with_weights:
        net.add_edge(u, v, value=w)
    net.save_graph("graph.html")
    return "graph.html"

def render_smooth_simulation(n_nodes, edges, P_matrix, start_node, speed, label_prefix="位置", max_steps=500):
    P_json = P_matrix.tolist()
    nodes_js = [{"id": i, "label": f"{label_prefix}{i}", "color": {"background": "#ADD8E6", "border": "#7EB8D4"}} for i in range(1, n_nodes + 1)]
    edges_js = []
    for u, v, w in edges:
        edges_js.append({"from": u, "to": v, "width": max(1, w * 2), "color": {"color": "#C8C8D0"}})

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
  body {{ background: #f5f7fa; }}
  #wrap {{ display: flex; flex-direction: column; height: 780px; gap: 10px; padding: 10px; }}

  /* ── Top stats bar ── */
  #statsbar {{
    display: flex; gap: 8px; flex-shrink: 0;
  }}
  .stat-pill {{
    flex: 1; background: white; border-radius: 14px;
    padding: 10px 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    display: flex; flex-direction: column; align-items: center;
  }}
  .stat-label {{ font-size: 10px; color: #8e8e93; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; }}
  .stat-value {{ font-size: 22px; font-weight: 700; color: #1d1d1f; margin-top: 2px; }}

  /* ── Main area: graph + side panel ── */
  #main {{ display: flex; gap: 10px; flex: 1; min-height: 0; }}
  #graph-wrap {{ flex: 1; background: white; border-radius: 20px; box-shadow: 0 4px 16px rgba(0,0,0,0.06); overflow: hidden; position: relative; }}
  #mynetwork {{ width: 100%; height: 100%; }}

  /* ── Side panel ── */
  #sidepanel {{ width: 180px; display: flex; flex-direction: column; gap: 8px; flex-shrink: 0; }}
  .panel-box {{ background: white; border-radius: 16px; padding: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  .panel-title {{ font-size: 10px; font-weight: 700; color: #8e8e93; letter-spacing: 0.6px; text-transform: uppercase; margin-bottom: 8px; }}

  /* Controls */
  #controls {{ display: flex; gap: 6px; }}
  .ctrl-btn {{
    flex: 1; border: none; border-radius: 12px; padding: 10px 6px;
    font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.2s;
  }}
  #btn-play  {{ background: linear-gradient(180deg,#34C759,#248A3D); color:white; box-shadow:0 3px 10px rgba(52,199,89,0.35); }}
  #btn-pause {{ background: linear-gradient(180deg,#FF9500,#CC7700); color:white; box-shadow:0 3px 10px rgba(255,149,0,0.35); }}
  #btn-reset {{ background: linear-gradient(180deg,#FF3B30,#CC2218); color:white; box-shadow:0 3px 10px rgba(255,59,48,0.30); }}
  .ctrl-btn:hover {{ transform: translateY(-1px); opacity: 0.92; }}
  .ctrl-btn:disabled {{ opacity: 0.4; cursor: default; transform: none; }}

  /* Visit freq bars */
  .freq-row {{ display: flex; align-items: center; gap: 6px; margin-bottom: 5px; }}
  .freq-label {{ font-size: 11px; color: #3a3a3c; width: 38px; flex-shrink: 0; }}
  .freq-bar-wrap {{ flex: 1; background: #f2f2f7; border-radius: 4px; height: 8px; overflow: hidden; }}
  .freq-bar {{ height: 100%; border-radius: 4px; background: linear-gradient(90deg,#007AFF,#5AC8FA); transition: width 0.4s ease; }}
  .freq-pct {{ font-size: 10px; color: #8e8e93; width: 32px; text-align: right; flex-shrink: 0; }}

  /* Path history */
  #path-scroll {{ max-height: 110px; overflow-y: auto; font-size: 11px; color: #3a3a3c; line-height: 1.8; }}
  #path-scroll::-webkit-scrollbar {{ width: 4px; }}
  #path-scroll::-webkit-scrollbar-thumb {{ background: #c7c7cc; border-radius: 2px; }}
  .path-step {{ display: inline-block; background: #f2f2f7; border-radius: 6px; padding: 1px 6px; margin: 1px; font-weight: 600; }}
  .path-step.current {{ background: #FFD60A; color: #1d1d1f; }}

  /* Speed slider */
  #speed-slider {{ width: 100%; accent-color: #007AFF; }}
  .speed-label {{ font-size: 11px; color: #8e8e93; text-align: center; margin-top: 4px; }}

  /* Legend */
  .legend-row {{ display: flex; align-items: center; gap: 6px; margin-bottom: 4px; font-size: 11px; color: #3a3a3c; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}

  /* Bottom: probability bar chart */
  #probchart {{ background: white; border-radius: 16px; padding: 12px 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); flex-shrink: 0; }}
  #probchart-title {{ font-size: 10px; font-weight: 700; color: #8e8e93; letter-spacing: 0.6px; text-transform: uppercase; margin-bottom: 8px; }}
  #barchart {{ display: flex; align-items: flex-end; gap: 4px; height: 60px; }}
  .bar-col {{ flex: 1; display: flex; flex-direction: column; align-items: center; gap: 2px; }}
  .bar-fill {{ width: 100%; border-radius: 4px 4px 0 0; transition: height 0.5s ease; min-height: 2px; }}
  .bar-lbl  {{ font-size: 9px; color: #8e8e93; }}
  .bar-pct  {{ font-size: 8px; color: #3a3a3c; font-weight: 600; }}
</style>
</head>
<body>
<div id="wrap">

  <!-- Stats bar -->
  <div id="statsbar">
    <div class="stat-pill"><div class="stat-label">步數</div><div class="stat-value" id="stat-steps">0</div></div>
    <div class="stat-pill"><div class="stat-label">目前位置</div><div class="stat-value" id="stat-pos">{start_node}</div></div>
    <div class="stat-pill"><div class="stat-label">最常拜訪</div><div class="stat-value" id="stat-top">—</div></div>
    <div class="stat-pill"><div class="stat-label">狀態</div><div class="stat-value" id="stat-status" style="font-size:13px;color:#34C759;">就緒</div></div>
  </div>

  <!-- Main -->
  <div id="main">
    <div id="graph-wrap"><div id="mynetwork"></div></div>

    <div id="sidepanel">
      <!-- Controls -->
      <div class="panel-box">
        <div class="panel-title">控制</div>
        <div id="controls">
          <button class="ctrl-btn" id="btn-play"  onclick="startSim()">▶ 開始</button>
          <button class="ctrl-btn" id="btn-pause" onclick="pauseSim()" disabled>⏸</button>
          <button class="ctrl-btn" id="btn-reset" onclick="resetSim()">↺</button>
        </div>
        <div style="margin-top:10px;">
          <div class="panel-title">速度</div>
          <input type="range" id="speed-slider" min="100" max="2000" value="{int(speed*1000)}" oninput="updateSpeed(this.value)">
          <div class="speed-label" id="speed-display">{speed:.1f} 秒/步</div>
        </div>
      </div>

      <!-- Visit frequency -->
      <div class="panel-box" style="flex:1; overflow:hidden; display:flex; flex-direction:column;">
        <div class="panel-title">拜訪頻率</div>
        <div id="freq-bars" style="flex:1; overflow-y:auto;"></div>
      </div>

      <!-- Legend -->
      <div class="panel-box">
        <div class="panel-title">圖例</div>
        <div class="legend-row"><div class="legend-dot" style="background:#FFD60A;border:2px solid #FFB800;"></div>目前位置</div>
        <div class="legend-row"><div class="legend-dot" style="background:#007AFF;border:2px solid #005BBF;"></div>上一步</div>
        <div class="legend-row"><div class="legend-dot" style="background:#ADD8E6;border:2px solid #7EB8D4;"></div>一般節點</div>
        <div class="legend-row"><div class="legend-dot" style="background:linear-gradient(135deg,#FF6B6B,#FFD60A);"></div>高頻節點</div>
      </div>
    </div>
  </div>

  <!-- Path history -->
  <div class="panel-box" style="flex-shrink:0;">
    <div class="panel-title">路徑紀錄（最近 40 步）</div>
    <div id="path-scroll"></div>
  </div>

  <!-- Prob bar chart -->
  <div id="probchart">
    <div id="probchart-title">實際拜訪比例 vs 理論穩態（藍=實際 / 灰=理論）</div>
    <div id="barchart"></div>
  </div>

</div>

<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<script>
const N = {n_nodes};
const P = {P_json};
const startNode = {start_node};
const steadyV = {list(P_matrix) and [float(x) for x in P_matrix.T[0]]};  // placeholder, recalc below
const labelPrefix = "{label_prefix}";
const MAX_STEPS = {max_steps};

// Compute steady state via power iteration
function computeSteady(P, n) {{
  let v = new Array(n).fill(1/n);
  for (let iter = 0; iter < 3000; iter++) {{
    let vn = new Array(n).fill(0);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        vn[j] += v[i] * P[i][j];
    let err = Math.max(...vn.map((x,i) => Math.abs(x - v[i])));
    v = vn;
    if (err < 1e-8) break;
  }}
  return v;
}}
const steady = computeSteady(P, N);

// Build vis.js network
const nodesData = {nodes_js};
const edgesData = {edges_js};
const visNodes = new vis.DataSet(nodesData);
const visEdges = new vis.DataSet(edgesData);
const network = new vis.Network(
  document.getElementById('mynetwork'),
  {{ nodes: visNodes, edges: visEdges }},
  {{
    nodes: {{ shape:'circle', size:28, font:{{size:14,face:'-apple-system',bold:{{color:'#1d1d1f'}}}}, borderWidth:2, shadow:{{enabled:true,size:6,x:0,y:2}} }},
    edges: {{ smooth:{{type:'continuous'}}, color:{{inherit:false}}, selectionWidth:0 }},
    physics: {{ enabled:true, barnesHut:{{gravitationalConstant:-2500,centralGravity:0.3,springLength:160}} }},
    interaction: {{ hover:true, tooltipDelay:100 }}
  }}
);

// State
let currentNode = startNode;
let stepCount = 0;
let visitCount = new Array(N+1).fill(0);
let pathHistory = [];
let intervalId = null;
let running = false;
let simSpeed = {int(speed*1000)};
let edgeMap = {{}};
let prevNode = -1;

// Build edge lookup
visEdges.getIds().forEach(id => {{
  const e = visEdges.get(id);
  edgeMap[`${{e.from}}-${{e.to}}`] = id;
  edgeMap[`${{e.to}}-${{e.from}}`] = id;
}});

// Init visit count
visitCount[startNode]++;
pathHistory.push(startNode);
highlightNode(startNode, 'current');
renderFreqBars();
renderProbChart();
renderPathHistory();

function heatColor(ratio) {{
  // ratio 0→1 maps to cool blue → warm red
  const r = Math.round(173 + (255-173)*ratio);
  const g = Math.round(216 - 216*ratio*0.7);
  const b = Math.round(230 - 230*ratio*0.8);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function highlightNode(nodeId, type) {{
  if (type === 'current') {{
    visNodes.update({{id: nodeId, color:{{background:'#FFD60A',border:'#FFB800'}}, shadow:{{enabled:true,size:10}}, size:34}});
  }} else if (type === 'prev') {{
    const ratio = visitCount[nodeId] / Math.max(1, Math.max(...visitCount.slice(1)));
    const bg = heatColor(ratio);
    visNodes.update({{id: nodeId, color:{{background:bg,border:'#7EB8D4'}}, shadow:{{enabled:true,size:5}}, size:28}});
  }} else {{
    visNodes.update({{id: nodeId, color:{{background:'#ADD8E6',border:'#7EB8D4'}}, shadow:{{enabled:false}}, size:28}});
  }}
}}

async function step() {{
  if (!running) return;
  const probs = P[currentNode - 1];
  let rand = Math.random(), nextNode = 1, cum = 0;
  for (let i = 0; i < probs.length; i++) {{
    cum += probs[i];
    if (rand <= cum) {{ nextNode = i + 1; break; }}
  }}

  // Animate edge
  const edgeKey = `${{currentNode}}-${{nextNode}}`;
  const edgeId = edgeMap[edgeKey];
  if (edgeId !== undefined) {{
    visEdges.update({{id:edgeId, color:{{color:'#007AFF'}}, width:5}});
    await delay(simSpeed * 0.35);
    if (!running) {{ visEdges.update({{id:edgeId, color:{{color:'#C8C8D0'}}, width:2}}); return; }}
    visEdges.update({{id:edgeId, color:{{color:'#C8C8D0'}}, width:2}});
  }} else {{
    await delay(simSpeed * 0.35);
  }}

  // Update nodes
  if (prevNode > 0) highlightNode(prevNode, 'heat');
  highlightNode(currentNode, 'prev');
  highlightNode(nextNode, 'current');

  prevNode = currentNode;
  currentNode = nextNode;
  stepCount++;
  visitCount[currentNode]++;
  pathHistory.push(currentNode);
  if (pathHistory.length > 40) pathHistory.shift();

  updateStats();
  renderFreqBars();
  renderProbChart();
  renderPathHistory();

  if (stepCount >= MAX_STEPS) {{
    pauseSim();
    document.getElementById('stat-status').textContent = '完成';
    document.getElementById('stat-status').style.color = '#FF9500';
  }}
}}

function delay(ms) {{ return new Promise(r => setTimeout(r, ms)); }}

function startSim() {{
  if (running) return;
  running = true;
  document.getElementById('btn-play').disabled = true;
  document.getElementById('btn-pause').disabled = false;
  document.getElementById('stat-status').textContent = '執行中';
  document.getElementById('stat-status').style.color = '#34C759';
  scheduleNext();
}}

function scheduleNext() {{
  if (!running) return;
  step().then(() => {{
    if (running) intervalId = setTimeout(scheduleNext, simSpeed * 0.65);
  }});
}}

function pauseSim() {{
  running = false;
  clearTimeout(intervalId);
  document.getElementById('btn-play').disabled = false;
  document.getElementById('btn-pause').disabled = true;
  document.getElementById('stat-status').textContent = '暫停';
  document.getElementById('stat-status').style.color = '#FF9500';
}}

function resetSim() {{
  pauseSim();
  currentNode = startNode;
  prevNode = -1;
  stepCount = 0;
  visitCount = new Array(N+1).fill(0);
  visitCount[startNode]++;
  pathHistory = [startNode];
  // Reset all nodes
  for (let i = 1; i <= N; i++) highlightNode(i, 'none');
  highlightNode(startNode, 'current');
  updateStats();
  renderFreqBars();
  renderProbChart();
  renderPathHistory();
  document.getElementById('stat-status').textContent = '就緒';
  document.getElementById('stat-status').style.color = '#34C759';
}}

function updateSpeed(val) {{
  simSpeed = parseInt(val);
  document.getElementById('speed-display').textContent = (simSpeed/1000).toFixed(1) + ' 秒/步';
}}

function updateStats() {{
  document.getElementById('stat-steps').textContent = stepCount;
  document.getElementById('stat-pos').textContent = labelPrefix + currentNode;
  const maxVisit = Math.max(...visitCount.slice(1));
  const topNode = visitCount.indexOf(maxVisit);
  document.getElementById('stat-top').textContent = labelPrefix + topNode;
}}

function renderFreqBars() {{
  const total = visitCount.slice(1).reduce((a,b)=>a+b,0) || 1;
  const maxV = Math.max(...visitCount.slice(1)) || 1;
  let html = '';
  for (let i = 1; i <= N; i++) {{
    const pct = visitCount[i] / total;
    html += `<div class="freq-row">
      <div class="freq-label">${{labelPrefix}}${{i}}</div>
      <div class="freq-bar-wrap"><div class="freq-bar" style="width:${{(visitCount[i]/maxV*100).toFixed(1)}}%"></div></div>
      <div class="freq-pct">${{(pct*100).toFixed(1)}}%</div>
    </div>`;
  }}
  document.getElementById('freq-bars').innerHTML = html;
}}

function renderProbChart() {{
  const total = visitCount.slice(1).reduce((a,b)=>a+b,0) || 1;
  const maxSteady = Math.max(...steady);
  let html = '';
  for (let i = 1; i <= N; i++) {{
    const actual = visitCount[i] / total;
    const theory = steady[i-1];
    const aH = Math.round(actual / maxSteady * 55);
    const tH = Math.round(theory / maxSteady * 55);
    html += `<div class="bar-col">
      <div style="display:flex;align-items:flex-end;gap:1px;height:55px;">
        <div class="bar-fill" style="height:${{aH}}px;background:linear-gradient(180deg,#007AFF,#5AC8FA);width:48%;"></div>
        <div class="bar-fill" style="height:${{tH}}px;background:#d1d1d6;width:48%;"></div>
      </div>
      <div class="bar-lbl">${{i}}</div>
    </div>`;
  }}
  document.getElementById('barchart').innerHTML = html;
}}

function renderPathHistory() {{
  const el = document.getElementById('path-scroll');
  el.innerHTML = pathHistory.map((n, idx) =>
    `<span class="path-step${{idx === pathHistory.length-1 ? ' current' : ''}}">${{labelPrefix}}${{n}}</span>`
  ).join(' → ');
  el.scrollLeft = el.scrollWidth;
}}
</script>
</body>
</html>"""
    return components.html(html_content, height=820)

# ======================================================================================================
# 4. Streamlit 主界面
# ======================================================================================================
st.set_page_config(page_title="Markov Analysis Suite Pro", layout="wide")
apply_custom_style()

st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #1d1d1f; font-family: sans-serif;'>🛠️ 系統分析模式選擇</h2>", unsafe_allow_html=True)
mode = st.radio("請選擇您要分析的對象：", ["👮 交通警察巡邏 (Police Patrol)", "🐁 8格迷宮老鼠 (Mouse Maze)"], horizontal=True, label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

INITIAL_TOPO = {
    'n_nodes': 5,
    'edges': [(1, 2, 1.0), (2, 4, 1.0), (4, 3, 1.0), (3, 1, 1.0), (1, 5, 1.0), (2, 5, 1.0), (3, 5, 1.0), (4, 5, 1.0)],
    'fixed_pos': {1: (0, 100), 2: (100, 100), 3: (0, 0), 4: (100, 0), 5: (50, 50)},
    'allow_self_loop': True
}
if 'topo_data' not in st.session_state:
    st.session_state.topo_data = INITIAL_TOPO.copy()

# 側邊欄
st.sidebar.header("⚙️ 配置中心")
if mode == "👮 交通警察巡邏 (Police Patrol)":
    with st.sidebar.expander("📍 佈局設定", expanded=True):
        layout_type = st.selectbox("選擇佈局", ["(5節點)佈局", "3x4 網格", "自定義網格", "手動輸入"])
        if layout_type == "(5節點)佈局": st.session_state.topo_data = INITIAL_TOPO.copy()
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
            st.session_state.topo_data = {'n_nodes': rows * cols, 'edges': edges, 'fixed_pos': None, 'allow_self_loop': True}
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
        edges = [(i, i + 1, 1.0) for i in range(1, 8)]
        fixed_pos = {i: (i * 100, 0) for i in range(1, 9)}
        st.session_state.topo_data = {'n_nodes': 8, 'edges': edges, 'fixed_pos': fixed_pos, 'allow_self_loop': False}

with st.sidebar.expander("📈 數學精度設定", expanded=False):
    threshold = st.number_input("收斂閾值", value=0.000001, format="%.7f")

if st.sidebar.button("🔄 一鍵重置所有配置"):
    st.session_state.topo_data = INITIAL_TOPO.copy()
    st.rerun()

# 計算
n_nodes = st.session_state.topo_data['n_nodes']
edges_with_weights = st.session_state.topo_data['edges']
fixed_pos = st.session_state.topo_data['fixed_pos']
allow_self = st.session_state.topo_data['allow_self_loop']
label_prefix = "路口" if mode == "👮 交通警察巡邏 (Police Patrol)" else "位置"
P, adj = build_transition_matrix(n_nodes, edges_with_weights, allow_self_loop=allow_self)
steady_v, iters, _ = find_steady_state(P, threshold)

# 指標
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("規模", f"{n_nodes} 處")
m_col2.metric("收斂次數", f"{iters} 次")
m_col3.metric("狀態", "穩定" if iters < 10000 else "未收斂")

# 分頁
tabs_list = ["🌐 互動拓撲圖", "⏱️ 隨機行走模擬", "📈 步數分佈演進", "📊 轉移矩陣", "📉 收斂趨勢", "🎯 穩定狀態", "📝 計算詳情", "📐 數學原理"]
if mode == "🐁 8格迷宮老鼠 (Mouse Maze)":
    tabs_list.insert(4, "🧮 矩陣運算分析")
tab_objs = st.tabs(tabs_list)
tab_map = {name: tab for name, tab in zip(tabs_list, tab_objs)}

# --- 🌐 互動拓撲圖 ---
with tab_map["🌐 互動拓撲圖"]:
    st.subheader(f"{label_prefix}連接視覺化")
    graph_html = create_interactive_graph(n_nodes, edges_with_weights, steady_v, fixed_pos, label_prefix)
    with open(graph_html, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=550)

# --- ⏱️ 隨機行走模擬 ---
with tab_map["⏱️ 隨機行走模擬"]:
    st.subheader("🚀 隨機行走模擬")
    col_ctrl, col_map = st.columns([1, 3])
    with col_ctrl:
        start_node_sim = st.number_input("設定起點", 1, n_nodes, 1)
        sim_speed = st.slider("初始速度 (秒/步)", 0.1, 2.0, 0.8)
        max_steps_sim = st.number_input("最大步數上限", 50, 5000, 500, step=50)
        st.markdown("""
        <div class="explain-box">
        <strong>✨ 新功能：</strong><br>
        • ▶ / ⏸ / ↺ 即時控制<br>
        • 滑桿動態調速（不需重啟）<br>
        • 節點顏色熱度圖（拜訪越多越紅）<br>
        • 拜訪頻率即時排行<br>
        • 實際比例 vs 理論穩態對比圖<br>
        • 路徑歷史紀錄（最近 40 步）
        </div>
        """, unsafe_allow_html=True)
        run_sim_btn = st.button("🎬 載入模擬")
    with col_map:
        if run_sim_btn:
            render_smooth_simulation(n_nodes, edges_with_weights, P, start_node_sim, sim_speed, label_prefix, max_steps_sim)
        else:
            st.info("設定好參數後，點擊左側「載入模擬」即可啟動。模擬載入後可用畫面內按鈕控制開始 / 暫停 / 重置，並即時調整速度。")

# --- 📈 步數分佈演進 (強化版) ---
with tab_map["📈 步數分佈演進"]:
    st.subheader("🚶 隨機行走機率演進分析")
    # ✅ 已修正公式：對應圖片中的遞迴展開式
    st.markdown(f"""<div class="glass-card"><div class="calc-box"><strong style="font-size:1.1rem;">📏 數學依據：n 步轉移機率計算</strong><div style="margin: 10px 0;">$$(P^n)_{{ij}} = \\sum_{{k=1}}^{{m}} P_{{ik}}^{{(n-1)}} P_{{kj}}$$</div><div class="explain-box"><strong>💡 邏輯解析：</strong><br>這是矩陣冪的遞迴展開式：$n$ 步轉移機率等於「先走 $n-1$ 步到中間節點 $k$，再從 $k$ 一步到 $j$」所有路徑的機率總和。</div></div></div>""", unsafe_allow_html=True)
    col_s1, col_s2 = st.columns([1, 1])
    with col_s1: start_pos_evo = st.number_input("選擇出發位置 (Starting Node)", 1, n_nodes, 1)
    with col_s2: hours = st.number_input("設定時間 (小時/步數)", 0, 500, 1)
    v0 = np.zeros(n_nodes); v0[start_pos_evo - 1] = 1.0
    Pm = np.linalg.matrix_power(P, hours)
    vm = np.dot(v0, Pm)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    res_c1, res_c2 = st.columns([1, 1])
    with res_c1:
        st.markdown(f"**⏱️ 第 {hours} 小時分佈表 (從 {label_prefix} {start_pos_evo} 出發)**")
        df_vm = pd.DataFrame({"位置": [f"{label_prefix} {i+1}" for i in range(n_nodes)], "機率": vm})
        st.table(df_vm.style.format({"機率": "{:.4%}"}))
    with res_c2:
        fig_vm, ax_vm = plt.subplots(figsize=(5, 4))
        ax_vm.bar(df_vm["位置"], df_vm["機率"], color="#007AFF")
        ax_vm.set_ylim(0, 1.0); ax_vm.set_title(f"Distribution after {hours} step(s)")
        st.pyplot(fig_vm); plt.close(fig_vm)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("⏳ 長期演進趨勢圖")
    max_trace = st.slider("分析總時長 (小時)", 1, 100, 20)
    trace_data = []
    curr_v = v0.copy()
    for s in range(max_trace + 1):
        trace_data.append(curr_v.copy()); curr_v = np.dot(curr_v, P)
    df_trace = pd.DataFrame(trace_data, columns=[f"{label_prefix} {i+1}" for i in range(n_nodes)])
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    fig_trace, ax_trace = plt.subplots(figsize=(10, 4))
    for col in df_trace.columns:
        ax_trace.plot(df_trace.index, df_trace[col], label=col, marker='.', markersize=4, alpha=0.8)
    ax_trace.set_title(f"Evolution from {label_prefix} {start_pos_evo}"); ax_trace.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    st.pyplot(fig_trace); plt.close(fig_trace)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 📊 轉移矩陣 ---
with tab_map["📊 轉移矩陣"]:
    st.subheader("轉移矩陣 $P$ (行加總為 1)")
    df_P = pd.DataFrame(P, index=[f"{label_prefix} {i+1}" for i in range(n_nodes)], columns=[f"{label_prefix} {i+1}" for i in range(n_nodes)])
    df_P['行加總 (Sum)'] = P.sum(axis=1)
    st.dataframe(df_P.style.format("{:.4f}"))

# --- 📉 收斂趨勢 ---
with tab_map["📉 收斂趨勢"]:
    st.subheader("收斂過程分析")
    u_iters = st.slider("調整迭代次數 (Iterations)", 1, 500, 100)
    err_hist = get_convergence_history_fixed(P, u_iters)
    if err_hist:
        fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
        ax_conv.plot(err_hist, color='#007AFF', lw=2, marker='o', markersize=2)
        ax_conv.set_yscale('log'); ax_conv.grid(True, alpha=0.3)
        st.pyplot(fig_conv); plt.close(fig_conv)

# --- 🎯 穩定狀態 ---
with tab_map["🎯 穩定狀態"]:
    st.subheader("長期分佈 (穩定狀態)")
    df_steady = pd.DataFrame({"位置": range(1, n_nodes+1), "機率": steady_v})
    st.table(df_steady.style.format({"機率": "{:.4%}"}))
    st.bar_chart(df_steady.set_index("位置")["機率"])

# --- 📝 計算詳情 ---
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
        st.latex(f"P_{{{row},{col}}} = \\frac{{w_{{{row} \\to {col}}}}}{{\\sum_{{\\text{{neighbors}}}} w + w_{{\\text{{self}}}}}}")
        st.latex(f"P_{{{row},{col}}} = \\frac{{{weight_ij:.1f}}}{{{total_w - self_w:.1f} + {self_w:.1f}}} = {res:.4f}")
        st.markdown(f'<div class="explain-box"><strong>💡 邏輯解析：</strong><br>這代表從{label_prefix} {row} 出發，選擇移動到{label_prefix} {col} 的權重佔比。</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    elif calc_mode == "穩定狀態元素 $\\pi_i$":
        node = st.number_input("選擇位置 $i$", 1, n_nodes, 1)
        sum_terms, formula_terms = [], []
        for j in range(1, n_nodes + 1):
            val = steady_v[j-1] * P[j-1, node-1]
            sum_terms.append(val); formula_terms.append(f"{steady_v[j-1]:.4f} \\times {P[j-1, node-1]:.4f}")
        st.markdown(f'<div class="calc-box">', unsafe_allow_html=True)
        st.latex(f"\\pi_{{{node}}} = \\sum_{{j=1}}^{{{n_nodes}}} (\\pi_{{j}} \\times P_{{j,{node}}})")
        st.latex(f"\\pi_{{{node}}} = {' + '.join(formula_terms)} = {sum(sum_terms):.4f}")
        st.markdown(f'<div class="explain-box"><strong>💡 邏輯解析：</strong><br>長期來看，您處於{label_prefix} {node} 的機率，等於所有能到達這裡的節點 $j$ 的貢獻總和。</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    elif calc_mode == "矩陣乘法 $(P^2)_{ij}$":
        c1, c2 = st.columns(2)
        with c1: r = st.number_input("選擇行 $i$", 1, n_nodes, 1)
        with c2: c = st.number_input("選擇列 $j$", 1, n_nodes, 1)
        terms, formula_terms = [], []
        for k in range(1, n_nodes + 1):
            val = P[r-1, k-1] * P[k-1, c-1]
            terms.append(val); formula_terms.append(f"{P[r-1, k-1]:.2f} \\times {P[k-1, c-1]:.2f}")
        st.markdown(f'<div class="calc-box">', unsafe_allow_html=True)
        st.latex(f"(P^2)_{{{r},{c}}} = \\sum_{{k=1}}^{{{n_nodes}}} (P_{{{r},{k}}} \\times P_{{{k},{c}}})")
        st.latex(f"(P^2)_{{{r},{c}}} = {' + '.join(formula_terms)} = {sum(terms):.4f}")
        st.markdown(f'<div class="explain-box"><strong>💡 邏輯解析：</strong><br>這是在計算「經過恰好 2 步」從 {label_prefix} {r} 到達 {label_prefix} {c} 的機率。</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 📐 數學原理 ---
with tab_map["📐 數學原理"]:
    st.subheader("📐 數學模型與解析")
    if mode == "🐁 8格迷宮老鼠 (Mouse Maze)":
        st.markdown("### 迷宮問題分析\n- **無自環限制**：$P_{ii} = 0$。\n- **多步轉移**：使用 $P^m$ 求解分佈。")
    else:
        st.markdown("### 巡邏問題分析\n- **自環權重**：$w_{ii} = 1.0$。")

# --- 🧮 矩陣運算分析 (迷宮專用) ---
if "🧮 矩陣運算分析" in tab_map:
    with tab_map["🧮 矩陣運算分析"]:
        st.subheader("🎯 特定步數機率計算")
        col_input, col_res = st.columns([1, 2])
        with col_input:
            start_node_m = st.number_input("設定起始位置 $v^{(0)}$", 1, n_nodes, 1)
            target_node_m = st.number_input("設定目標位置", 1, n_nodes, 5)
            steps_m = st.number_input("計算步數 $m$", 1, 100, 2)
        with col_res:
            Pm_m = np.linalg.matrix_power(P, steps_m)
            v0_m = np.zeros(n_nodes); v0_m[start_node_m-1] = 1.0
            vm_m = np.dot(v0_m, Pm_m)
            st.metric(f"經過 {steps_m} 步後，在位置 {target_node_m} 的機率", f"{vm_m[target_node_m-1]:.4%}")
            df_vm_m = pd.DataFrame({"位置": range(1, n_nodes+1), "機率": vm_m})
            st.bar_chart(df_vm_m.set_index("位置")["機率"])
