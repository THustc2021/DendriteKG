#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Elsevier Full-Text Retrieval XML extractor.

This script is designed for Elsevier's Full-Text Retrieval API XML responses
(often starting with <full-text-retrieval-response ...>).

It extracts common metadata (title, authors, DOI, journal, etc.) plus a clean
plain-text version of the body (section titles + paragraphs).

Examples:
  python elsevier_xml_extractor.py input.xml -o output.json
  python elsevier_xml_extractor.py /path/to/dir --glob "*.xml" -o out_dir

Notes:
- Elsevier XML uses many namespaces; this script is namespace-aware.
- The "full_text" is best-effort plain text; figures/tables/math are reduced
  to their textual content.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import xml.etree.ElementTree as ET


@dataclass
class ExtractedArticle:
    source_file: str
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    doi: Optional[str] = None
    pii: Optional[str] = None
    eid: Optional[str] = None
    journal: Optional[str] = None
    cover_date: Optional[str] = None
    publisher: Optional[str] = None
    copyright: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    abstract: Optional[str] = None
    full_text: str = ""


def _strip_ws(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s


def _localname(tag: str) -> str:
    # {namespace}tag -> tag
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _gather_namespaces(xml_path: str) -> Dict[str, str]:
    """Collect namespaces from the XML file (prefix -> uri)."""
    ns: Dict[str, str] = {}
    for event, elem in ET.iterparse(xml_path, events=("start-ns",)):
        prefix, uri = elem
        if prefix is None:
            prefix = ""
        if prefix not in ns:
            ns[prefix] = uri
    return ns


def _find_text(root: ET.Element, xpath: str, ns: Dict[str, str]) -> Optional[str]:
    el = root.find(xpath, ns)
    if el is None:
        return None
    txt = "".join(el.itertext())
    txt = _strip_ws(txt)
    return txt or None


def _find_all_text(root: ET.Element, xpath: str, ns: Dict[str, str]) -> List[str]:
    out: List[str] = []
    for el in root.findall(xpath, ns):
        txt = _strip_ws("".join(el.itertext()))
        if txt:
            out.append(txt)
    return out


def _iter_section_blocks(root: ET.Element, ns: Dict[str, str]) -> Iterable[Tuple[str, str]]:
    """Yield (kind, text) where kind in {"h", "p"} for headings/paragraphs."""

    # The body commonly lives under: originalText/xocs:doc/.../ce:sections
    # We keep it flexible: search any ce:section-title and ce:para under originalText.
    original = root.find(".//{*}originalText")
    search_root = original if original is not None else root

    # Traverse in document order and emit titles/paras.
    # ElementTree doesn't provide full XPath axes; we do a manual walk.
    for el in search_root.iter():
        ln = _localname(el.tag)
        if ln == "section-title":
            t = _strip_ws("".join(el.itertext()))
            if t:
                yield ("h", t)
        elif ln == "para":
            t = _strip_ws("".join(el.itertext()))
            if t:
                yield ("p", t)


def extract_elsevier_fulltext_xml(xml_path: str) -> ExtractedArticle:
    ns = _gather_namespaces(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Core metadata is usually in <coredata> with dc:/prism:
    # We use wildcard namespace matches where possible, but keep prefixes if present.
    # Determine likely prefixes from the file.
    # If these prefixes aren't present, the wildcard find will still work for many tags.

    # Title
    title = _find_text(root, ".//{*}coredata/{*}title", ns)

    # Authors: dc:creator repeats
    authors = _find_all_text(root, ".//{*}coredata/{*}creator", ns)

    doi = _find_text(root, ".//{*}coredata/{*}doi", ns) or _find_text(root, ".//{*}coredata/{*}identifier", ns)
    # If identifier is like "doi:...", normalize
    if doi and doi.lower().startswith("doi:"):
        doi = doi.split(":", 1)[1].strip()

    pii = _find_text(root, ".//{*}coredata/{*}pii", ns)
    eid = _find_text(root, ".//{*}coredata/{*}eid", ns) or _find_text(root, ".//{*}scopus-eid", ns)

    journal = _find_text(root, ".//{*}coredata/{*}publicationName", ns) or _find_text(root, ".//{*}srctitle", ns)
    cover_date = _find_text(root, ".//{*}coredata/{*}coverDate", ns)
    publisher = _find_text(root, ".//{*}coredata/{*}publisher", ns)
    copyright_txt = _find_text(root, ".//{*}coredata/{*}copyright", ns)

    # Keywords: dcterms:subject repeats
    keywords = _find_all_text(root, ".//{*}coredata/{*}subject", ns)

    # Abstract/description: dc:description
    abstract = _find_text(root, ".//{*}coredata/{*}description", ns)

    # Full text body
    blocks: List[str] = []
    for kind, text in _iter_section_blocks(root, ns):
        if kind == "h":
            # Make headings stand out
            blocks.append(f"\n## {text}\n")
        else:
            blocks.append(text)

    # Join and lightly clean
    full_text = "\n".join(blocks)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

    return ExtractedArticle(
        source_file=os.path.abspath(xml_path),
        title=title,
        authors=authors,
        doi=doi,
        pii=pii,
        eid=eid,
        journal=journal,
        cover_date=cover_date,
        publisher=publisher,
        copyright=copyright_txt,
        keywords=keywords,
        abstract=abstract,
        full_text=full_text,
    )


def _write_json(data: object, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Extract metadata + full text from Elsevier full-text retrieval XML.")
    p.add_argument("input", help="Input XML file, or a directory if --glob is used")
    p.add_argument("-o", "--output", required=True, help="Output path: .json file, or directory if batch")
    p.add_argument("--glob", default=None, help="If input is a directory, glob pattern (e.g. '*.xml')")
    p.add_argument("--also-txt", action="store_true", help="Also write a .txt with the full_text")

    args = p.parse_args(argv)

    in_path = args.input
    out_path = args.output

    inputs: List[str] = []
    batch = False

    if os.path.isdir(in_path):
        if not args.glob:
            p.error("When input is a directory, you must provide --glob, e.g. --glob '*.xml'")
        batch = True
        inputs = sorted(glob.glob(os.path.join(in_path, args.glob)))
        if not inputs:
            raise FileNotFoundError(f"No files matched {args.glob!r} under {in_path!r}")
        os.makedirs(out_path, exist_ok=True)
    else:
        inputs = [in_path]

    results: List[dict] = []
    for xml_file in inputs:
        article = extract_elsevier_fulltext_xml(xml_file)
        article_dict = asdict(article)
        results.append(article_dict)

        if batch:
            base = os.path.splitext(os.path.basename(xml_file))[0]
            json_out = os.path.join(out_path, f"{base}.json")
            _write_json(article_dict, json_out)
            if args.also_txt:
                txt_out = os.path.join(out_path, f"{base}.txt")
                with open(txt_out, "w", encoding="utf-8") as f:
                    f.write(article.full_text)
        else:
            _write_json(article_dict, out_path)
            if args.also_txt:
                txt_out = os.path.splitext(out_path)[0] + ".txt"
                with open(txt_out, "w", encoding="utf-8") as f:
                    f.write(article.full_text)

    # If batch, also emit an index file for convenience
    if batch:
        index_out = os.path.join(out_path, "_index.json")
        _write_json(results, index_out)

    return 0
