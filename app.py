import json
import os
import tempfile
from typing import Dict, Any, List, Set, Tuple

import streamlit as st
import networkx as nx
from pyvis.network import Network

st.set_page_config(page_title="Knowledge Graph Viewer", page_icon="🧠", layout="wide")

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

        # evidence bundle
        evidence_list = e.get("evidence", [])
        papers_list = e.get("papers", [])
        certainty_list = e.get("certainty", [])
        confidence_list = e.get("confidence", [])

        G.add_edge(
            s,
            t,
            relation=r,
            evidence=evidence_list,
            papers=papers_list,
            certainty=certainty_list,
            confidence=confidence_list,
        )
    return G

def ego_subgraph(G: nx.MultiDiGraph, center: str, hops: int) -> nx.MultiDiGraph:
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

def pyvis_html(
    G: nx.MultiDiGraph,
    height_px: int,
    physics: bool,
    show_edge_labels: bool,
    highlight: str,
    relation_filter: Set[str] | None,
) -> str:
    net = Network(height=f"{height_px}px", width="100%", directed=True, bgcolor="#FFFFFF", font_color="#1f1f1f")
    net.toggle_physics(physics)
    if physics:
        net.barnes_hut(gravity=-8000, central_gravity=0.2, spring_length=140, spring_strength=0.02, damping=0.09)

    # Nodes
    for nid, data in G.nodes(data=True):
        ntype = data.get("type", "Unknown")
        label = data.get("label", nid)
        color = TYPE_COLOR.get(ntype, TYPE_COLOR["Unknown"])

        size = 18
        border = 1
        if highlight and nid == highlight:
            size = 30
            border = 5

        title = f"<b>{label}</b><br>id: {nid}<br>type: {ntype}"
        net.add_node(nid, label=label, title=title, color=color, size=size, borderWidth=border)

    # Edges
    for u, v, k, d in G.edges(keys=True, data=True):
        rel = d.get("relation", "related_to")
        if relation_filter and rel not in relation_filter:
            continue

        evidence = d.get("evidence", [])
        papers = d.get("papers", [])
        certainty = d.get("certainty", [])
        confidence = d.get("confidence", [])

        # Keep tooltip concise
        tooltip_lines = [f"{u} —({rel})→ {v}"]
        if papers:
            tooltip_lines.append(f"papers: {', '.join(papers[:5])}" + ("..." if len(papers) > 5 else ""))
        if certainty:
            tooltip_lines.append(f"certainty: {', '.join(certainty)}")
        if confidence:
            tooltip_lines.append(f"confidence: {', '.join(confidence)}")
        if evidence:
            tooltip_lines.append("evidence:")
            tooltip_lines.append(evidence[0][:240] + ("..." if len(evidence[0]) > 240 else ""))

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

# -----------------------------
# UI
# -----------------------------
st.title("🧠 Knowledge Graph App Demo")

with st.sidebar:
    st.header("Graph Source")
    mode = st.radio(
        "Choose graph mode",
        ["Merged global graph", "Per-paper subgraph"],
        index=0,
    )

    st.caption("If you already ran build_kg.py, put outputs in ./kg_out/ .")

    merged_path = os.path.join("kg_out", "kg_merged.json")
    bypaper_path = os.path.join("kg_out", "kg_by_paper.json")

    uploaded_merged = st.file_uploader("Upload kg_merged.json (optional)", type=["json"])
    uploaded_bypaper = st.file_uploader("Upload kg_by_paper.json (optional)", type=["json"])

    st.divider()
    st.header("View")
    height_px = st.slider("Graph height (px)", 450, 1100, 750, 50)
    physics = st.toggle("Enable physics", value=True)
    show_edge_labels = st.toggle("Show edge labels", value=True)

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

if mode == "Merged global graph":
    if merged_graph is None:
        st.info("No merged graph found. Please upload kg_merged.json or place it under ./kg_out/")
        st.stop()
    nodes, edges = parse_graph(merged_graph)
    G = build_nx_graph(nodes, edges)
else:
    if bypaper_graphs is None:
        st.info("No per-paper graph found. Please upload kg_by_paper.json or place it under ./kg_out/")
        st.stop()

    paper_ids = sorted(list(bypaper_graphs.keys()))
    paper_id = st.selectbox("Select a paper/file subgraph", paper_ids)
    nodes, edges = parse_graph(bypaper_graphs[paper_id])
    G = build_nx_graph(nodes, edges)

# Controls row
all_nodes = sorted(list(G.nodes()))
all_relations = sorted({d.get("relation", "related_to") for _, _, _, d in G.edges(keys=True, data=True)})

c1, c2, c3 = st.columns([1.2, 1.0, 1.8])
with c1:
    center = st.selectbox("Center node", options=["(none)"] + all_nodes, index=0)
with c2:
    hops = st.slider("Path length (hops)", 1, 6, 2)
with c3:
    rel_selected = st.multiselect("Filter relations", options=all_relations, default=all_relations)

relation_filter = set(rel_selected) if rel_selected else None

# Apply ego-view if center chosen
display_G = G
highlight = ""
if center != "(none)":
    highlight = center
    display_G = ego_subgraph(G, center, hops)

# Stats
m1, m2, m3 = st.columns(3)
m1.metric("Nodes", display_G.number_of_nodes())
m2.metric("Edges", display_G.number_of_edges())
m3.metric("Relations", len(all_relations))

st.divider()

html = pyvis_html(
    display_G,
    height_px=height_px,
    physics=physics,
    show_edge_labels=show_edge_labels,
    highlight=highlight,
    relation_filter=relation_filter,
)
st.components.v1.html(html, height=height_px + 40, scrolling=True)

with st.expander("Expected record format (one JSON block per triple)"):
    st.code(
        """{
  "subject_id": "Paper_chapter_12",
  "subject_type": "Paper",
  "predicate": "STUDIES",
  "object_id": "Material_lithium_metal",
  "object_type": "Material",
  "paper_id": "Paper_chapter_12",
  "evidence": "....",
  "certainty": "explicit",
  "confidence": "null",
  "qualifiers": {}
}""",
        language="json",
    )
