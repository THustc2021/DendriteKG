import json
import tempfile
from typing import Dict, List, Any, Tuple, Set

import streamlit as st
import networkx as nx
from pyvis.network import Network

st.set_page_config(page_title="Knowledge Graph Demo", page_icon="🧠", layout="wide")

# -----------------------------
# Sample KG data
# -----------------------------
SAMPLE_DATA = {
    "nodes": [
        {"id": "Albert Einstein", "type": "Person"},
        {"id": "Physics", "type": "Field"},
        {"id": "Relativity", "type": "Theory"},
        {"id": "Nobel Prize", "type": "Award"},
        {"id": "Max Planck", "type": "Person"},
        {"id": "Germany", "type": "Place"},
        {"id": "Switzerland", "type": "Place"},
        {"id": "ETH Zurich", "type": "Organization"},
    ],
    "edges": [
        {"source": "Albert Einstein", "target": "Physics", "relation": "studied"},
        {"source": "Albert Einstein", "target": "Relativity", "relation": "proposed"},
        {"source": "Albert Einstein", "target": "Nobel Prize", "relation": "won"},
        {"source": "Albert Einstein", "target": "Germany", "relation": "born_in"},
        {"source": "Albert Einstein", "target": "Switzerland", "relation": "worked_in"},
        {"source": "Albert Einstein", "target": "ETH Zurich", "relation": "affiliated_with"},
        {"source": "Max Planck", "target": "Physics", "relation": "contributed_to"},
        {"source": "Max Planck", "target": "Germany", "relation": "born_in"},
    ],
}

# A small color palette for node types
TYPE_COLOR = {
    "Person": "#4C78A8",
    "Field": "#F58518",
    "Theory": "#54A24B",
    "Award": "#E45756",
    "Place": "#72B7B2",
    "Organization": "#B279A2",
    "Unknown": "#9D9DA1",
}

# -----------------------------
# Helpers
# -----------------------------
def parse_kg_data(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Invalid JSON format: 'nodes' and 'edges' must be lists.")
    # basic validation
    node_ids = set()
    for n in nodes:
        if "id" not in n:
            raise ValueError("Each node must have an 'id'.")
        node_ids.add(str(n["id"]))
    for e in edges:
        if "source" not in e or "target" not in e:
            raise ValueError("Each edge must have 'source' and 'target'.")
        # allow edges pointing to missing nodes; we will add them as Unknown
    return nodes, edges


def build_graph(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for n in nodes:
        nid = str(n.get("id"))
        ntype = str(n.get("type", "Unknown"))
        G.add_node(nid, type=ntype)

    for e in edges:
        s = str(e.get("source"))
        t = str(e.get("target"))
        r = str(e.get("relation", "related_to"))

        if not G.has_node(s):
            G.add_node(s, type="Unknown")
        if not G.has_node(t):
            G.add_node(t, type="Unknown")

        # Multi-edge support
        G.add_edge(s, t, relation=r)

    return G


def subgraph_around(G: nx.MultiDiGraph, center: str, hops: int) -> nx.MultiDiGraph:
    """Return an ego-subgraph around 'center' within 'hops' hops (treat as undirected for neighborhood)."""
    if center not in G:
        return G

    und = nx.Graph()
    und.add_nodes_from(G.nodes(data=True))
    for u, v, k, d in G.edges(keys=True, data=True):
        und.add_edge(u, v)

    nodes: Set[str] = set([center])
    frontier: Set[str] = set([center])
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            nxt |= set(und.neighbors(n))
        nxt -= nodes
        nodes |= nxt
        frontier = nxt

    SG = G.subgraph(nodes).copy()
    return SG


def to_pyvis_html(
    G: nx.MultiDiGraph,
    height_px: int,
    physics: bool,
    show_edge_labels: bool,
    highlight_node: str = "",
    relation_filter: Set[str] | None = None,
) -> str:
    net = Network(height=f"{height_px}px", width="100%", directed=True, bgcolor="#FFFFFF", font_color="#1f1f1f")

    # physics config
    net.toggle_physics(physics)
    if physics:
        net.barnes_hut(gravity=-8000, central_gravity=0.2, spring_length=140, spring_strength=0.02, damping=0.09)

    # Determine which edges to include based on relation filter
    edges_to_add = []
    for u, v, k, d in G.edges(keys=True, data=True):
        rel = d.get("relation", "related_to")
        if relation_filter and rel not in relation_filter:
            continue
        edges_to_add.append((u, v, rel))

    # Add nodes
    for nid, data in G.nodes(data=True):
        ntype = data.get("type", "Unknown")
        color = TYPE_COLOR.get(ntype, TYPE_COLOR["Unknown"])
        title = f"<b>{nid}</b><br>type: {ntype}"

        size = 18
        border_width = 1

        if highlight_node and nid == highlight_node:
            size = 30
            border_width = 5

        net.add_node(
            nid,
            label=nid,
            title=title,
            color=color,
            size=size,
            borderWidth=border_width,
        )

    # Add edges
    for u, v, rel in edges_to_add:
        title = f"{u} —({rel})→ {v}"
        if show_edge_labels:
            net.add_edge(u, v, label=rel, title=title, arrows="to")
        else:
            net.add_edge(u, v, title=title, arrows="to")

    # allow interactive controls
    net.set_options(
        """
        var options = {
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          },
          "nodes": {
            "shape": "dot"
          },
          "edges": {
            "smooth": {
              "type": "dynamic"
            }
          }
        }
        """
    )

    # Save to a temp HTML and read back
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        tmp_path = f.name
    net.save_graph(tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as rf:
        html = rf.read()
    return html


# -----------------------------
# UI
# -----------------------------
st.title("🧠 Knowledge Graph Visualization Demo (Streamlit)")

with st.sidebar:
    st.header("Data")
    use_sample = st.toggle("Use sample knowledge graph", value=True)

    uploaded = st.file_uploader("Upload JSON (optional)", type=["json"], disabled=use_sample)

    st.caption(
        "JSON format:\n"
        "- nodes: [{id, type?}, ...]\n"
        "- edges: [{source, target, relation?}, ...]"
    )

    st.divider()
    st.header("View")
    height_px = st.slider("Graph height (px)", 400, 1100, 700, 50)
    physics = st.toggle("Enable physics (force layout)", value=True)
    show_edge_labels = st.toggle("Show relation labels on edges", value=True)

# load data
try:
    if use_sample:
        data = SAMPLE_DATA
    else:
        if uploaded is None:
            st.info("Upload a JSON file in the sidebar, or switch on 'Use sample knowledge graph'.")
            st.stop()
        data = json.load(uploaded)

    nodes, edges = parse_kg_data(data)
    G = build_graph(nodes, edges)
except Exception as e:
    st.error(f"Failed to load/parse data: {e}")
    st.stop()

# controls row
col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.0])

all_nodes = sorted(list(G.nodes()))
all_relations = sorted({d.get("relation", "related_to") for _, _, _, d in G.edges(keys=True, data=True)})

with col1:
    search_node = st.selectbox("Search / focus node", options=["(none)"] + all_nodes, index=0)

with col2:
    hops = st.slider("Neighborhood hops", 1, 4, 2)

with col3:
    rel_selected = st.multiselect("Filter relations", options=all_relations, default=all_relations)

with col4:
    st.write("")
    st.write("")
    show_stats = st.toggle("Show stats", value=True)

relation_filter = set(rel_selected) if rel_selected else None

# optionally subgraph
display_graph = G
highlight = ""
if search_node != "(none)":
    highlight = search_node
    display_graph = subgraph_around(G, search_node, hops)

# stats
if show_stats:
    c1, c2, c3 = st.columns(3)
    c1.metric("Nodes", display_graph.number_of_nodes())
    c2.metric("Edges", display_graph.number_of_edges())
    c3.metric("Relations", len(all_relations))

st.divider()

# render
html = to_pyvis_html(
    display_graph,
    height_px=height_px,
    physics=physics,
    show_edge_labels=show_edge_labels,
    highlight_node=highlight,
    relation_filter=relation_filter,
)

st.components.v1.html(html, height=height_px + 40, scrolling=True)

with st.expander("Example JSON format"):
    st.code(
        json.dumps(
            {
                "nodes": [{"id": "A", "type": "Entity"}, {"id": "B", "type": "Entity"}],
                "edges": [{"source": "A", "target": "B", "relation": "related_to"}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        language="json",
    )
