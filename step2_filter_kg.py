# postprocess_kg.py
import argparse
import json
import os
import math
import re
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Set

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


# ---------------------------
# Prefix stripping + normalization (generic, non-domain)
# ---------------------------
TYPE_PREFIX_RE = re.compile(r"^[A-Za-z]+_")   # e.g., Material_xxx -> xxx
SEP_RE = re.compile(r"[_\-]+")               # underscores/hyphens -> space


def strip_type_prefix(text: str) -> str:
    """
    Remove leading 'Type_' prefix if present.
    Example: 'Material_lithium_metal' -> 'lithium_metal'
    """
    if text is None:
        return ""
    t = str(text).strip()
    t = TYPE_PREFIX_RE.sub("", t)
    return t


def label_no_prefix(text: str) -> str:
    """
    Human-facing label without prefix.
    - strip Type_ prefix
    - convert '_' and '-' to spaces
    - collapse spaces
    (No lowercasing here; keep pretty output.)
    """
    t = strip_type_prefix(text)
    t = SEP_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_for_merge(text: str) -> str:
    """
    Merge key normalization (used for embeddings):
    - strip Type_ prefix
    - '_'/'-' -> space
    - lowercase
    - collapse spaces
    """
    t = label_no_prefix(text)
    t = t.lower()
    return t


# ---------------------------
# Union-Find for clustering
# ---------------------------
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# ---------------------------
# IO utils
# ---------------------------
def load_graph(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


# ---------------------------
# 1) Synonym merge by embeddings (no explicit synonym table)
# ---------------------------
def cluster_nodes_by_semantics(
    nodes: List[Dict[str, Any]],
    model_name: str,
    similarity_threshold: float,
    knn_k: int,
    same_type_only: bool,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Return:
      - id_map: original_node_id -> canonical_node_id
      - report: details of clusters

    IMPORTANT CHANGE:
      - embeddings are computed from "no-prefix" labels (and normalized), not raw ids.
    """
    ids = [str(n["id"]) for n in nodes]
    types = [str(n.get("type", "Unknown")) for n in nodes]

    # Prefer node["label"], fallback to id; BUT merge basis should be no-prefix
    raw_labels = [str(n.get("label", n["id"])) for n in nodes]

    # Use no-prefix label for merge text; if empty fallback to no-prefix id
    texts = []
    for lab, nid in zip(raw_labels, ids):
        base = label_no_prefix(lab)
        if not base:
            base = label_no_prefix(nid)
        texts.append(normalize_for_merge(base))

    model = SentenceTransformer(model_name)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)

    n = len(nodes)
    if n == 0:
        return {}, {"clusters": [], "note": "no nodes"}

    k = min(knn_k, n)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(emb)
    dists, idxs = nn.kneighbors(emb, return_distance=True)

    uf = UnionFind(n)

    for i in range(n):
        for jpos in range(1, k):  # skip itself at pos 0
            j = int(idxs[i, jpos])
            dist = float(dists[i, jpos])
            sim = 1.0 - dist
            if sim < similarity_threshold:
                continue
            if same_type_only and types[i] != types[j]:
                continue
            uf.union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    # Choose canonical id per cluster:
    # Keep canonical id stable as an existing node id (we do NOT rewrite ids here).
    # But the canonical label we output later will be no-prefix.
    id_map: Dict[str, str] = {}
    cluster_report = []

    for root, members in clusters.items():
        if len(members) == 1:
            i = members[0]
            id_map[ids[i]] = ids[i]
            continue

        member_ids = [ids[i] for i in members]
        member_types = [types[i] for i in members]
        member_texts = [texts[i] for i in members]
        member_labels = [raw_labels[i] for i in members]

        # canonical merge-text: most common normalized no-prefix text
        canonical_text = Counter(member_texts).most_common(1)[0][0]

        candidates = [i for i in members if texts[i] == canonical_text]
        if not candidates:
            candidates = members

        # canonical id: shortest id among candidates (stable)
        canonical_idx = min(candidates, key=lambda i: (len(ids[i]), ids[i]))
        canonical_id = ids[canonical_idx]

        for mid in member_ids:
            id_map[mid] = canonical_id

        cluster_report.append({
            "canonical_id": canonical_id,
            "canonical_text": canonical_text,
            "members": [
                {
                    "id": ids[i],
                    "type": types[i],
                    "label_raw": raw_labels[i],
                    "label_no_prefix": label_no_prefix(raw_labels[i]) or label_no_prefix(ids[i]),
                    "norm_for_merge": texts[i],
                }
                for i in members
            ],
            "type_distribution": dict(Counter(member_types)),
        })

    report = {
        "model": model_name,
        "similarity_threshold": similarity_threshold,
        "same_type_only": same_type_only,
        "knn_k": knn_k,
        "num_nodes_in": n,
        "num_clusters": len(clusters),
        "merged_clusters": cluster_report,
    }
    return id_map, report


# ---------------------------
# 2) Merge nodes + resolve type conflicts
# ---------------------------
def merge_nodes(
    nodes: List[Dict[str, Any]],
    id_map: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Merge nodes that map to same canonical id.
    Type conflict resolution: majority vote; keep conflict info.

    IMPORTANT CHANGE:
      - output node["label"] is ALWAYS the no-prefix label.
      - label voting/selection uses the no-prefix label space.
    """
    buckets = defaultdict(list)
    for n in nodes:
        oid = str(n["id"])
        cid = id_map.get(oid, oid)
        buckets[cid].append(n)

    merged_nodes = []
    conflict_report = []

    for cid, group in buckets.items():
        # vote type
        types = [str(g.get("type", "Unknown")) for g in group]
        type_counts = Counter(types)
        chosen_type = type_counts.most_common(1)[0][0]

        # Build no-prefix label candidates
        raw_labels = [str(g.get("label", g["id"])) for g in group]
        no_prefix_labels = []
        for lab, g in zip(raw_labels, group):
            base = label_no_prefix(lab)
            if not base:
                base = label_no_prefix(str(g["id"]))
            no_prefix_labels.append(base)

        # Choose canonical label by most common normalized form, then shortest pretty label
        norm_no_prefix = [normalize_for_merge(x) for x in no_prefix_labels]
        best_norm = Counter(norm_no_prefix).most_common(1)[0][0]
        best_candidates = [lab for lab, nlab in zip(no_prefix_labels, norm_no_prefix) if nlab == best_norm]
        chosen_label = min(best_candidates, key=len) if best_candidates else (label_no_prefix(cid) or cid)

        # Aliases
        aliases = sorted({str(g["id"]) for g in group})

        # Keep both raw alias labels and no-prefix alias labels for traceability
        alias_labels_raw = sorted({str(g.get("label", g["id"])) for g in group})
        alias_labels_no_prefix = sorted({label_no_prefix(x) for x in alias_labels_raw if label_no_prefix(x)})

        node_out = {
            "id": cid,
            "type": chosen_type,
            "label": chosen_label,  # <-- no prefix
            "aliases": aliases,
            "alias_labels_raw": alias_labels_raw,
            "alias_labels": alias_labels_no_prefix,
        }
        merged_nodes.append(node_out)

        if len(type_counts) > 1:
            conflict_report.append({
                "id": cid,
                "type_distribution": dict(type_counts),
                "aliases": aliases,
            })

    merged_nodes.sort(key=lambda x: x["id"])
    report = {
        "num_nodes_in": len(nodes),
        "num_nodes_out": len(merged_nodes),
        "type_conflicts": conflict_report,
        "num_type_conflicts": len(conflict_report),
    }
    return merged_nodes, report


# ---------------------------
# 3) Remap + merge edges (union evidence/papers/etc)
# ---------------------------
def merge_edges(edges: List[Dict[str, Any]], id_map: Dict[str, str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    merged = {}

    def add_set_field(dst: Dict[str, Any], key: str, values):
        if key not in dst:
            dst[key] = set()
        for v in ensure_list(values):
            if v is None:
                continue
            sv = str(v).strip()
            if not sv or sv.lower() == "null":
                continue
            dst[key].add(sv)

    for e in edges:
        s0 = str(e["source"])
        t0 = str(e["target"])
        s = id_map.get(s0, s0)
        t = id_map.get(t0, t0)
        rel = str(e.get("relation", "related_to"))

        k = (s, rel, t)
        if k not in merged:
            merged[k] = {
                "source": s,
                "target": t,
                "relation": rel,
                "papers": set(),
                "evidence": set(),
                "certainty": set(),
                "confidence": set(),
            }

        add_set_field(merged[k], "papers", e.get("papers", []))
        add_set_field(merged[k], "evidence", e.get("evidence", []))
        add_set_field(merged[k], "certainty", e.get("certainty", []))
        add_set_field(merged[k], "confidence", e.get("confidence", []))

    out = []
    for obj in merged.values():
        obj["papers"] = sorted(list(obj["papers"]))
        obj["evidence"] = sorted(list(obj["evidence"]))
        obj["certainty"] = sorted(list(obj["certainty"]))
        obj["confidence"] = sorted(list(obj["confidence"]))
        out.append(obj)

    out.sort(key=lambda x: (x["source"], x["relation"], x["target"]))
    report = {"num_edges_in": len(edges), "num_edges_out": len(out)}
    return out, report


# ---------------------------
# 4) Remove isolated nodes
# ---------------------------
def filter_isolated_nodes(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    deg = {n["id"]: 0 for n in nodes}
    for e in edges:
        s = str(e["source"])
        t = str(e["target"])
        if s in deg:
            deg[s] += 1
        if t in deg:
            deg[t] += 1

    kept = [n for n in nodes if deg.get(n["id"], 0) > 0]
    removed = [n for n in nodes if deg.get(n["id"], 0) == 0]

    report = {
        "num_nodes_before": len(nodes),
        "num_nodes_after": len(kept),
        "num_isolated_removed": len(removed),
    }
    return kept, report


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_graph", required=True, help="Input graph JSON with {nodes, edges}")
    ap.add_argument("--out_graph", default="kg_out/kg_postprocessed.json", help="Output postprocessed graph JSON")
    ap.add_argument("--out_map", default="data/step3_kg_operation_reports/kg_id_map.json", help="Output node id mapping (old->new)")
    ap.add_argument("--out_report", default="data/step3_kg_operation_reports/kg_postprocess_report.json", help="Output report JSON")

    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model")
    ap.add_argument("--threshold", type=float, default=0.8, help="Cosine similarity threshold for merging")
    ap.add_argument("--knn_k", type=int, default=25, help="KNN neighborhood size for candidate merges")
    ap.add_argument("--same_type_only", action="store_true", help="Only merge nodes with the same type")

    args = ap.parse_args()

    graph = load_graph(args.in_graph)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    # 1) synonym/near-duplicate merge mapping (based on no-prefix labels)
    id_map, synonym_report = cluster_nodes_by_semantics(
        nodes=nodes,
        model_name=args.model,
        similarity_threshold=args.threshold,
        knn_k=args.knn_k,
        same_type_only=args.same_type_only,
    )

    # 2) merge nodes + resolve type conflicts (output label is no-prefix)
    merged_nodes, type_report = merge_nodes(nodes, id_map)

    # 3) remap + merge edges (union evidence/papers/etc)
    merged_edges, edge_report = merge_edges(edges, id_map)

    # 4) remove isolated nodes
    final_nodes, iso_report = filter_isolated_nodes(merged_nodes, merged_edges)

    # Safety: filter edges referencing removed nodes
    keep_ids = {n["id"] for n in final_nodes}
    final_edges = [e for e in merged_edges if e["source"] in keep_ids and e["target"] in keep_ids]

    out_graph = {"nodes": final_nodes, "edges": final_edges}
    save_json(out_graph, args.out_graph)
    save_json(id_map, args.out_map)

    report = {
        "input": {"nodes": len(nodes), "edges": len(edges)},
        "synonym_merge": synonym_report,
        "type_merge": type_report,
        "edge_merge": edge_report,
        "isolation_filter": iso_report,
        "final": {"nodes": len(final_nodes), "edges": len(final_edges)},
        "notes": [
            "Merging basis uses NO-PREFIX labels (strip leading Type_ then normalize).",
            "Output node.label is also NO-PREFIX (human-facing).",
            "Adjust --threshold to control aggressiveness; use --same_type_only first to reduce false merges.",
        ],
    }
    save_json(report, args.out_report)

    print("Done.")
    print(f"- out_graph:   {args.out_graph}")
    print(f"- out_map:     {args.out_map}")
    print(f"- out_report:  {args.out_report}")


if __name__ == "__main__":
    main()
