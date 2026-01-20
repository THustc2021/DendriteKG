# build_kg.py
import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, Any, List, Tuple

JSON_BLOCK_RE = re.compile(r"```json(.*?)```", re.DOTALL)

def iter_json_objects_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract JSON objects from a text that contains multiple JSON blocks."""
    objs = []
    for m in JSON_BLOCK_RE.finditer(text):
        block = m.group(1).strip()
        try:
            objs.append(json.loads(block))
        except json.JSONDecodeError:
            # Skip malformed blocks silently; you can log if needed.
            continue
    return objs

def normalize_node(node_id: str, node_type: str) -> Dict[str, Any]:
    return {
        "id": node_id,
        "type": node_type or "Unknown",
        "label": node_id,  # you can later prettify: strip prefixes like "Material_"
    }

def edge_key(src: str, rel: str, dst: str) -> Tuple[str, str, str]:
    return (src, rel, dst)

def add_evidence(edge_obj: Dict[str, Any], paper_id: str, evidence: str, certainty: str, confidence: Any):
    # Keep a compact list; de-duplicate
    if paper_id:
        edge_obj["papers"].add(paper_id)
    if evidence:
        edge_obj["evidence"].add(evidence)
    if certainty:
        edge_obj["certainty"].add(certainty)
    if confidence is not None and str(confidence).lower() != "null":
        edge_obj["confidence"].add(str(confidence))

def build_graph_from_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build one graph JSON from a list of triple records."""
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for r in records:
        s_id = str(r.get("subject_id", "")).strip()
        s_type = str(r.get("subject_type", "Unknown")).strip()
        p = str(r.get("predicate", "related_to")).strip()
        o_id = str(r.get("object_id", "")).strip()
        o_type = str(r.get("object_type", "Unknown")).strip()

        paper_id = str(r.get("paper_id", "")).strip()
        evidence = str(r.get("evidence", "")).strip()
        certainty = str(r.get("certainty", "")).strip()
        confidence = r.get("confidence", None)

        if not s_id or not o_id:
            continue

        if s_id not in nodes:
            nodes[s_id] = normalize_node(s_id, s_type)
        if o_id not in nodes:
            nodes[o_id] = normalize_node(o_id, o_type)

        k = edge_key(s_id, p, o_id)
        if k not in edges:
            edges[k] = {
                "source": s_id,
                "target": o_id,
                "relation": p,
                "papers": set(),
                "evidence": set(),
                "certainty": set(),
                "confidence": set(),
            }
        add_evidence(edges[k], paper_id, evidence, certainty, confidence)

    # finalize sets -> lists
    edge_list = []
    for e in edges.values():
        e["papers"] = sorted(list(e["papers"]))
        e["evidence"] = sorted(list(e["evidence"]))
        e["certainty"] = sorted(list(e["certainty"]))
        e["confidence"] = sorted(list(e["confidence"]))
        edge_list.append(e)

    return {
        "nodes": list(nodes.values()),
        "edges": edge_list,
    }

def remove_paper_nodes_and_incident_edges(graph: Dict[str, Any]) -> Dict[str, Any]:
    """For merged graph: remove Paper nodes and edges connected to them."""
    nodes = graph["nodes"]
    edges = graph["edges"]

    paper_ids = {n["id"] for n in nodes if n.get("type") == "Paper" or str(n["id"]).startswith("Paper_")}
    kept_nodes = [n for n in nodes if n["id"] not in paper_ids]

    kept_node_ids = {n["id"] for n in kept_nodes}

    kept_edges = []
    for e in edges:
        if e["source"] in paper_ids or e["target"] in paper_ids:
            continue
        if e["source"] in kept_node_ids and e["target"] in kept_node_ids:
            kept_edges.append(e)

    return {"nodes": kept_nodes, "edges": kept_edges}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory containing *.txt files")
    ap.add_argument("--out_dir", default="kg_out", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Collect records by paper_id
    records_by_paper: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_records: List[Dict[str, Any]] = []

    for fn in os.listdir(args.in_dir):
        if not fn.lower().endswith(".txt"):
            continue
        path = os.path.join(args.in_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        objs = iter_json_objects_from_text(text)
        for r in objs:
            paper_id = str(r.get("paper_id", "")).strip() or fn  # fallback
            records_by_paper[paper_id].append(r)
            all_records.append(r)

    # Build per-paper graphs
    kg_by_paper = {}
    for paper_id, recs in records_by_paper.items():
        kg_by_paper[paper_id] = build_graph_from_records(recs)

    with open(os.path.join(args.out_dir, "kg_by_paper.json"), "w", encoding="utf-8") as f:
        json.dump(kg_by_paper, f, ensure_ascii=False, indent=2)

    # Build merged
    merged = build_graph_from_records(all_records)
    merged_clean = remove_paper_nodes_and_incident_edges(merged)

    with open(os.path.join(args.out_dir, "kg_merged.json"), "w", encoding="utf-8") as f:
        json.dump(merged_clean, f, ensure_ascii=False, indent=2)

    print(f"Done. Wrote:\n- {os.path.join(args.out_dir, 'kg_by_paper.json')}\n- {os.path.join(args.out_dir, 'kg_merged.json')}")

if __name__ == "__main__":
    main()
