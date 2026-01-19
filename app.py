# app.py
import json
import os
import math
import tempfile
from typing import Dict, Any, List, Set, Tuple, Optional

import streamlit as st
import networkx as nx
from pyvis.network import Network

st.set_page_config(page_title="Knowledge Graph Viewer", page_icon="🧠", layout="wide")

# -----------------------------
# Perf defaults (tune as needed)
# -----------------------------
DEFAULT_MAX_NODES = 350
DEFAULT_MAX_EDGES = 1200

# -----------------------------
# Colors
# -----------------------------
TYPE_COLOR = {
    "Paper": "#9D9DA1",
    "Material": "#4C78A8",
    "Interface": "#F58518",
    "Phenomenon": "#54A24B",
    "Mechanism": "#E45756",
    "Condition": "#72B7B2",
    "Quantity": "#B279A2",
    "Relationship": "#FF9DA6",
    "PhaseFieldIngredient": "#A0CBE8",
    "Unknown": "#9D9DA1",
}

# -----------------------------
# Loaders
# -----------------------------
@st.cache_data(show_spinner=False)
def load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_graph(graph: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Graph JSON must contain list fields: nodes, edges")
    return nodes, edges

def build_nx_graph(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for n in nodes:
        nid = str(n.get("id"))
        ntype = str(n.get("type", "Unknown"))
        label = str(n.get("label", nid))
        G.add_node(nid, type=ntype, label=label)

    for e in edges:
        s = str(e.get("source"))
        t = str(e.get("target"))
        r = str(e.get("relation", "related_to"))

        if not G.has_node(s):
            G.add_node(s, type="Unknown", label=s)
        if not G.has_node(t):
            G.add_node(t, type="Unknown", label=t)

        G.add_edge(
            s,
            t,
            relation=r,
            evidence=e.get("evidence", []),
            papers=e.get("papers", []),
            certainty=e.get("certainty", []),
            confidence=e.get("confidence", []),
        )
    return G

# -----------------------------
# Graph utils
# -----------------------------
def to_undirected_simple(G: nx.MultiDiGraph) -> nx.Graph:
    """Collapse MultiDiGraph into undirected simple graph for distances/degrees."""
    und = nx.Graph()
    und.add_nodes_from(G.nodes(data=True))
    for u, v, k in G.edges(keys=True):
        und.add_edge(u, v)
    return und

def ego_subgraph(G: nx.MultiDiGraph, center: str, hops: int) -> nx.MultiDiGraph:
    """Neighborhood subgraph within N hops (treat as undirected for neighborhood)."""
    if center not in G:
        return G

    und = to_undirected_simple(G)

    nodes: Set[str] = {center}
    frontier: Set[str] = {center}
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            nxt |= set(und.neighbors(n))
        nxt -= nodes
        nodes |= nxt
        frontier = nxt

    return G.subgraph(nodes).copy()

def limit_graph_size(G: nx.MultiDiGraph, center: str, max_nodes: int, max_edges: int) -> nx.MultiDiGraph:
    """
    If subgraph is too large, keep:
    - center node
    - highest-degree nodes (in undirected projection)
    - highest "importance" edges (degree(u)+degree(v))
    """
    if G.number_of_nodes() <= max_nodes and G.number_of_edges() <= max_edges:
        return G

    und = to_undirected_simple(G)
    deg = dict(und.degree())

    # Always keep center, then high-degree nodes
    nodes_sorted = sorted(G.nodes(), key=lambda n: (n != center, -deg.get(n, 0), str(n)))
    keep_nodes = set(nodes_sorted[:max_nodes])
    keep_nodes.add(center)

    SG = G.subgraph(keep_nodes).copy()

    # If too many edges, keep top edges by endpoint degrees
    if SG.number_of_edges() > max_edges:
        scored = []
        for u, v, k in SG.edges(keys=True):
            scored.append((deg.get(u, 0) + deg.get(v, 0), u, v, k))
        scored.sort(reverse=True)

        H = nx.MultiDiGraph()
        H.add_nodes_from(SG.nodes(data=True))
        for _, u, v, k in scored[:max_edges]:
            H.add_edge(u, v, **SG.edges[u, v, k])
        SG = H

    return SG

def shortest_distances(und: nx.Graph, center: str, cutoff: int) -> Dict[str, int]:
    if center not in und:
        return {}
    return nx.single_source_shortest_path_length(und, center, cutoff=cutoff)

def short_label(s: str, n: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"

# -----------------------------
# Payload (cache-friendly)
# -----------------------------
def graph_to_payload(
    G_display: nx.MultiDiGraph,
    G_full: nx.MultiDiGraph,
    center: str,
    hops: int,
    relation_filter: Optional[Set[str]],
    label_char_limit: int,
    faint_alpha: float,
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    nodes_payload item:
      (nid, label, ntype, size, font_rgba, is_focus_or_neighbor)
    edges_payload item:
      (u, v, rel, papers_n, ev0)
    """
    und_display = to_undirected_simple(G_display)
    dist = shortest_distances(und_display, center, cutoff=hops)

    und_full = to_undirected_simple(G_full)
    deg_full = dict(und_full.degree())

    # Normalize degree -> size
    # Use sqrt scaling for stability
    max_deg = max(deg_full.values()) if deg_full else 1
    max_sqrt = math.sqrt(max_deg) if max_deg > 0 else 1.0

    def node_size(nid: str) -> float:
        d = deg_full.get(nid, 0)
        # base 10..36 approx
        base = 10.0
        span = 26.0
        return base + span * (math.sqrt(d) / max_sqrt if max_sqrt > 0 else 0.0)

    nodes_payload: List[Tuple] = []
    for nid in G_display.nodes():
        label = G_display.nodes[nid].get("label", nid)
        ntype = G_display.nodes[nid].get("type", "Unknown")

        d = dist.get(nid, 999999)
        focus_or_neighbor = (nid == center) or (d == 1)

        if focus_or_neighbor:
            alpha = 1.0
            label_use = short_label(label, label_char_limit)
        else:
            alpha = max(0.0, min(1.0, faint_alpha))
            label_use = short_label(label, max(6, min(label_char_limit, 14)))  # slightly shorter for faint nodes

        # font color with alpha
        font_rgba = f"rgba(40,40,40,{alpha:.2f})"

        size = node_size(nid)
        if nid == center:
            size *= 1.25
        elif d == 1:
            size *= 1.10

        nodes_payload.append((nid, label_use, ntype, float(size), font_rgba, focus_or_neighbor))

    edges_payload: List[Tuple] = []
    for u, v, k, d in G_display.edges(keys=True, data=True):
        rel = d.get("relation", "related_to")
        if relation_filter and rel not in relation_filter:
            continue

        papers = d.get("papers", []) or []
        evidence = d.get("evidence", []) or []
        ev0 = ""
        if evidence:
            ev0 = str(evidence[0])
            if len(ev0) > 240:
                ev0 = ev0[:240] + "..."
        edges_payload.append((u, v, rel, len(papers), ev0))

    return nodes_payload, edges_payload

# -----------------------------
# Rendering (cached)
# -----------------------------
def pyvis_html_from_payload(
    nodes_payload,
    edges_payload,
    height_px: int,
    physics: bool,
    show_edge_labels: bool,
    show_only_focus_labels: bool,
    center: str,
    spring_length: int,
    avoid_overlap: float,
) -> str:
    net = Network(height=f"{height_px}px", width="100%", directed=True, bgcolor="#FFFFFF", font_color="#1f1f1f")

    # Nodes
    for nid, label, ntype, size, font_rgba, focus_or_neighbor in nodes_payload:
        color = TYPE_COLOR.get(ntype, TYPE_COLOR["Unknown"])

        # If user wants only focus labels visible, hide others' label completely (still have hover)
        label_to_show = label if (not show_only_focus_labels or focus_or_neighbor) else ""

        title = f"<b>{G_LABEL(nid, nodes_payload)}</b><br>id: {nid}<br>type: {ntype}"
        if nid == center:
            title = f"<b>{G_LABEL(nid, nodes_payload)}</b> (CENTER)<br>id: {nid}<br>type: {ntype}"

        net.add_node(
            nid,
            label=label_to_show,
            title=title,
            color=color,
            size=size,
            borderWidth=(5 if nid == center else (3 if focus_or_neighbor else 1)),
            font={"color": font_rgba, "size": (18 if nid == center else (14 if focus_or_neighbor else 12))},
        )

    # Edges
    for u, v, rel, papers_n, ev0 in edges_payload:
        tooltip_lines = [f"{u} —({rel})→ {v}"]
        if papers_n:
            tooltip_lines.append(f"papers: {papers_n}")
        if ev0:
            tooltip_lines.append("evidence:")
            tooltip_lines.append(ev0)
        title = "<br>".join(tooltip_lines)

        if show_edge_labels:
            net.add_edge(u, v, label=rel, title=title, arrows="to")
        else:
            net.add_edge(u, v, title=title, arrows="to")

    # Options: space + overlap control
    # - avoidOverlap helps reduce node collisions (best with physics on)
    # - springLength creates more space
    options = f"""
    var options = {{
      "interaction": {{
        "hover": true,
        "navigationButtons": true,
        "keyboard": true
      }},
      "nodes": {{
        "shape": "dot",
        "scaling": {{
          "min": 8,
          "max": 60
        }}
      }},
      "edges": {{
        "smooth": {{
          "type": "dynamic"
        }},
        "arrows": {{
          "to": {{
            "enabled": true,
            "scaleFactor": 0.7
          }}
        }}
      }},
      "physics": {{
        "enabled": {str(physics).lower()},
        "barnesHut": {{
          "gravitationalConstant": -12000,
          "centralGravity": 0.15,
          "springLength": {spring_length},
          "springConstant": 0.03,
          "damping": 0.12,
          "avoidOverlap": {avoid_overlap}
        }},
        "stabilization": {{
          "enabled": true,
          "iterations": 200
        }}
      }}
    }}
    """
    net.set_options(options)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        tmp_path = f.name
    net.save_graph(tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as rf:
        return rf.read()

def G_LABEL(nid: str, nodes_payload) -> str:
    # nodes_payload: (nid, label, ntype, size, font_rgba, focus_or_neighbor)
    # we want full label-ish; we only have truncated label here, but better than nid.
    for item in nodes_payload:
        if item[0] == nid:
            return item[1] if item[1] else nid
    return nid

@st.cache_data(show_spinner=False)
def render_cached(
    nodes_payload,
    edges_payload,
    height_px: int,
    physics: bool,
    show_edge_labels: bool,
    show_only_focus_labels: bool,
    center: str,
    spring_length: int,
    avoid_overlap: float,
) -> str:
    return pyvis_html_from_payload(
        nodes_payload=nodes_payload,
        edges_payload=edges_payload,
        height_px=height_px,
        physics=physics,
        show_edge_labels=show_edge_labels,
        show_only_focus_labels=show_only_focus_labels,
        center=center,
        spring_length=spring_length,
        avoid_overlap=avoid_overlap,
    )

# -----------------------------
# UI
# -----------------------------
st.title("🧠 Knowledge Graph Viewer (Focus Labels + Spatial Context + Importance Sizing)")

with st.sidebar:
    st.header("Graph Source")
    mode = st.radio("Choose graph mode", ["Merged global graph", "Per-paper subgraph"], index=0)

    st.caption("If you ran build_kg.py, put outputs under ./kg_out/")

    merged_path = os.path.join("kg_out", "kg_merged.json")
    bypaper_path = os.path.join("kg_out", "kg_by_paper.json")

    uploaded_merged = st.file_uploader("Upload kg_merged.json (optional)", type=["json"])
    uploaded_bypaper = st.file_uploader("Upload kg_by_paper.json (optional)", type=["json"])

    st.divider()
    st.header("Performance")
    st.caption("Tip: render only neighborhood subgraphs; increase limits carefully.")
    auto_limit = st.toggle("Auto-limit subgraph size", value=True)
    max_nodes = st.slider("Max nodes to render", 100, 800, DEFAULT_MAX_NODES, 50)
    max_edges = st.slider("Max edges to render", 200, 3000, DEFAULT_MAX_EDGES, 100)

    st.divider()
    st.header("Layout & Labels")
    physics = st.toggle("Physics (better spacing, slower)", value=True)
    spring_length = st.slider("Spacing (spring length)", 120, 420, 260, 20)
    avoid_overlap = st.slider("Avoid overlap", 0.0, 2.0, 1.0, 0.1)

    show_edge_labels = st.toggle("Show edge labels (can clutter)", value=False)

    st.caption("Label strategy: show clear labels for center + neighbors; others are faint.")
    show_only_focus_labels = st.toggle("Hide non-focus labels completely", value=False)
    faint_alpha = st.slider("Non-focus label opacity", 0.0, 0.6, 0.18, 0.02)
    label_char_limit = st.slider("Max label length", 8, 40, 22, 1)

    st.divider()
    st.header("View")
    height_px = st.slider("Graph height (px)", 450, 1100, 780, 50)

# Load graphs
try:
    if uploaded_merged is not None:
        merged_graph = json.load(uploaded_merged)
    else:
        merged_graph = load_json_file(merged_path) if os.path.exists(merged_path) else None

    if uploaded_bypaper is not None:
        bypaper_graphs = json.load(uploaded_bypaper)
    else:
        bypaper_graphs = load_json_file(bypaper_path) if os.path.exists(bypaper_path) else None

except Exception as e:
    st.error(f"Failed to load graphs: {e}")
    st.stop()

# Select graph
if mode == "Merged global graph":
    if merged_graph is None:
        st.info("No merged graph found. Upload kg_merged.json or place it under ./kg_out/")
        st.stop()
    nodes, edges = parse_graph(merged_graph)
    G_full = build_nx_graph(nodes, edges)
    graph_label = "Merged graph"
else:
    if bypaper_graphs is None:
        st.info("No per-paper graph found. Upload kg_by_paper.json or place it under ./kg_out/")
        st.stop()

    paper_ids = sorted(list(bypaper_graphs.keys()))
    paper_id = st.selectbox("Select a paper/file subgraph", paper_ids)
    nodes, edges = parse_graph(bypaper_graphs[paper_id])
    G_full = build_nx_graph(nodes, edges)
    graph_label = f"Subgraph: {paper_id}"

# Controls row
all_nodes = sorted(list(G_full.nodes()))
all_relations = sorted({d.get("relation", "related_to") for _, _, _, d in G_full.edges(keys=True, data=True)})

c1, c2, c3 = st.columns([1.4, 1.0, 1.8])
with c1:
    center = st.selectbox("Center node (required for rendering)", options=["(none)"] + all_nodes, index=0)
with c2:
    hops = st.slider("Path length (hops)", 1, 6, 2)
with c3:
    rel_selected = st.multiselect("Filter relations", options=all_relations, default=all_relations)

relation_filter = set(rel_selected) if rel_selected else None

# Stats about full selected graph
with st.expander("Graph stats (full selected graph)", expanded=False):
    st.write(f"**{graph_label}**")
    st.write(f"- Nodes: {G_full.number_of_nodes()}")
    st.write(f"- Edges: {G_full.number_of_edges()}")
    st.write(f"- Relation types: {len(all_relations)}")

# Full-graph rendering disabled for performance
if center == "(none)":
    st.info(
        "To keep the app fast and readable, full-graph rendering is disabled. "
        "Please select a **Center node**, then adjust **hops** to expand/shrink its neighborhood."
    )
    st.stop()

# Build neighborhood subgraph
G_display = ego_subgraph(G_full, center, hops)

# Auto-limit
before_n, before_e = G_display.number_of_nodes(), G_display.number_of_edges()
if auto_limit:
    G_display = limit_graph_size(G_display, center, max_nodes=max_nodes, max_edges=max_edges)
after_n, after_e = G_display.number_of_nodes(), G_display.number_of_edges()

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rendered nodes", after_n)
m2.metric("Rendered edges", after_e)
m3.metric("Center", center)
m4.metric("Hops", hops)

if auto_limit and (before_n != after_n or before_e != after_e):
    st.warning(
        f"Subgraph was auto-limited for performance: "
        f"{before_n}→{after_n} nodes, {before_e}→{after_e} edges. "
        f"Increase limits in the sidebar if needed."
    )

st.divider()

# Build cache-friendly payload (includes: focus labels visible, others faint; sizes by importance)
nodes_payload, edges_payload = graph_to_payload(
    G_display=G_display,
    G_full=G_full,
    center=center,
    hops=hops,
    relation_filter=relation_filter,
    label_char_limit=label_char_limit,
    faint_alpha=faint_alpha,
)

html = render_cached(
    nodes_payload=nodes_payload,
    edges_payload=edges_payload,
    height_px=height_px,
    physics=physics,
    show_edge_labels=show_edge_labels,
    show_only_focus_labels=show_only_focus_labels,
    center=center,
    spring_length=spring_length,
    avoid_overlap=avoid_overlap,
)

st.components.v1.html(html, height=height_px + 40, scrolling=True)

with st.expander("Expected graph JSON format"):
    st.code(
        json.dumps(
            {
                "nodes": [
                    {"id": "A", "type": "Material", "label": "Lithium metal"},
                    {"id": "B", "type": "Mechanism", "label": "Dendrite growth"},
                ],
                "edges": [
                    {
                        "source": "A",
                        "target": "B",
                        "relation": "PROMOTES",
                        "papers": ["Paper_xxx", "Paper_yyy"],
                        "evidence": ["some sentence..."],
                        "certainty": ["explicit"],
                        "confidence": ["0.8"],
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        language="json",
    )
