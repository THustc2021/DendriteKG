# app.py
import json
import os
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

# Node type colors (extend freely)
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
# Subgraph + limiting
# -----------------------------
def ego_subgraph(G: nx.MultiDiGraph, center: str, hops: int) -> nx.MultiDiGraph:
    """Neighborhood subgraph within N hops (treat as undirected for neighborhood)."""
    if center not in G:
        return G

    und = nx.Graph()
    und.add_nodes_from(G.nodes(data=True))
    for u, v, k, d in G.edges(keys=True, data=True):
        und.add_edge(u, v)

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

    und = nx.Graph()
    und.add_nodes_from(G.nodes(data=True))
    for u, v, k, d in G.edges(keys=True, data=True):
        und.add_edge(u, v)

    deg = dict(und.degree())

    # Always keep center, then high-degree nodes
    nodes_sorted = sorted(G.nodes(), key=lambda n: (n != center, -deg.get(n, 0), str(n)))
    keep_nodes = set(nodes_sorted[:max_nodes])

    if center in G:
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

# -----------------------------
# Rendering (cached)
# -----------------------------
def graph_to_payload(G: nx.MultiDiGraph, relation_filter: Optional[Set[str]] = None):
    """
    Convert graph to a cache-friendly payload.
    For performance, store minimal edge info + a short tooltip snippet.
    """
    nodes_payload = [(n, G.nodes[n].get("label", n), G.nodes[n].get("type", "Unknown")) for n in G.nodes()]
    edges_payload = []
    for u, v, k, d in G.edges(keys=True, data=True):
        rel = d.get("relation", "related_to")
        if relation_filter and rel not in relation_filter:
            continue

        # compact tooltip info
        papers = d.get("papers", []) or []
        evidence = d.get("evidence", []) or []
        ev0 = ""
        if evidence:
            ev0 = str(evidence[0])
            if len(ev0) > 240:
                ev0 = ev0[:240] + "..."
        papers_n = len(papers)

        edges_payload.append((u, v, rel, papers_n, ev0))
    return nodes_payload, edges_payload

def pyvis_html_from_payload(
    nodes_payload,
    edges_payload,
    height_px: int,
    physics: bool,
    show_edge_labels: bool,
    show_node_labels: bool,
    highlight: str,
) -> str:
    net = Network(height=f"{height_px}px", width="100%", directed=True, bgcolor="#FFFFFF", font_color="#1f1f1f")

    # Physics is the biggest browser-side cost; keep off by default
    net.toggle_physics(physics)
    if physics:
        net.barnes_hut(gravity=-8000, central_gravity=0.2, spring_length=140, spring_strength=0.02, damping=0.09)

    # Nodes
    for nid, label, ntype in nodes_payload:
        color = TYPE_COLOR.get(ntype, TYPE_COLOR["Unknown"])
        size = 18
        border = 1
        if highlight and nid == highlight:
            size = 30
            border = 5

        title = f"<b>{label}</b><br>id: {nid}<br>type: {ntype}"
        net.add_node(
            nid,
            label=(label if show_node_labels else ""),
            title=title,
            color=color,
            size=size,
            borderWidth=border,
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

    net.set_options(
        """
        var options = {
          "interaction": {"hover": true, "navigationButtons": true, "keyboard": true},
          "nodes": {"shape": "dot"},
          "edges": {"smooth": {"type": "dynamic"}}
        }
        """
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        tmp_path = f.name
    net.save_graph(tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as rf:
        return rf.read()

@st.cache_data(show_spinner=False)
def render_cached(
    nodes_payload,
    edges_payload,
    height_px: int,
    physics: bool,
    show_edge_labels: bool,
    show_node_labels: bool,
    highlight: str,
) -> str:
    return pyvis_html_from_payload(
        nodes_payload,
        edges_payload,
        height_px=height_px,
        physics=physics,
        show_edge_labels=show_edge_labels,
        show_node_labels=show_node_labels,
        highlight=highlight,
    )

# -----------------------------
# UI
# -----------------------------
st.title("🧠 Knowledge Graph Viewer (Fast Neighborhood Rendering)")

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
    st.caption("Tip: keep Physics OFF and render only a neighborhood subgraph.")
    auto_limit = st.toggle("Auto-limit subgraph size", value=True)
    max_nodes = st.slider("Max nodes to render", 100, 800, DEFAULT_MAX_NODES, 50)
    max_edges = st.slider("Max edges to render", 200, 3000, DEFAULT_MAX_EDGES, 100)

    physics = st.toggle("Physics (slow)", value=False)
    show_edge_labels = st.toggle("Show edge labels (slow)", value=False)
    show_node_labels = st.toggle("Show node labels", value=True)

    st.divider()
    st.header("View")
    height_px = st.slider("Graph height (px)", 450, 1100, 750, 50)

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
    G = build_nx_graph(nodes, edges)
    graph_label = "Merged graph"
else:
    if bypaper_graphs is None:
        st.info("No per-paper graph found. Upload kg_by_paper.json or place it under ./kg_out/")
        st.stop()

    paper_ids = sorted(list(bypaper_graphs.keys()))
    paper_id = st.selectbox("Select a paper/file subgraph", paper_ids)
    nodes, edges = parse_graph(bypaper_graphs[paper_id])
    G = build_nx_graph(nodes, edges)
    graph_label = f"Subgraph: {paper_id}"

# Controls row
all_nodes = sorted(list(G.nodes()))
all_relations = sorted({d.get("relation", "related_to") for _, _, _, d in G.edges(keys=True, data=True)})

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
    st.write(f"- Nodes: {G.number_of_nodes()}")
    st.write(f"- Edges: {G.number_of_edges()}")
    st.write(f"- Relation types: {len(all_relations)}")

# IMPORTANT: Do not render full graph by default (huge speed win)
if center == "(none)":
    st.info(
        "To keep the app fast, full-graph rendering is disabled. "
        "Please select a **Center node**, then adjust **hops** to expand/shrink its neighborhood."
    )
    st.stop()

# Build neighborhood subgraph
display_G = ego_subgraph(G, center, hops)

# Auto-limit
before_n, before_e = display_G.number_of_nodes(), display_G.number_of_edges()
if auto_limit:
    display_G = limit_graph_size(display_G, center, max_nodes=max_nodes, max_edges=max_edges)

after_n, after_e = display_G.number_of_nodes(), display_G.number_of_edges()

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

# Payload + cached render
nodes_payload, edges_payload = graph_to_payload(display_G, relation_filter=relation_filter)

html = render_cached(
    nodes_payload=nodes_payload,
    edges_payload=edges_payload,
    height_px=height_px,
    physics=physics,
    show_edge_labels=show_edge_labels,
    show_node_labels=show_node_labels,
    highlight=center,
)

st.components.v1.html(html, height=height_px + 40, scrolling=True)

with st.expander("Expected graph JSON format"):
    st.code(
        json.dumps(
            {
                "nodes": [{"id": "A", "type": "Material", "label": "A"}, {"id": "B", "type": "Mechanism", "label": "B"}],
                "edges": [
                    {
                        "source": "A",
                        "target": "B",
                        "relation": "INFLUENCES",
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
