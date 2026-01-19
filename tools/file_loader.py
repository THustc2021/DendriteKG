import re

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from lxml import etree

def extract_elsevier_fulltext_xml(xml_path: str) -> Dict[str, Any]:
    """
    Extract "valuable" content from Elsevier Full-Text Retrieval XML.

    Returns a dict with:
      - metadata: title/doi/pii/publication/coverDate/authors/keywords/abstract
      - sections: list of {path, title, paragraphs}
      - figures: list of {label, caption, ref}
      - tables: list of {label, caption, ref}
      - full_text: concatenated text (sectioned)
    """

    parser = etree.XMLParser(recover=True, huge_tree=True, remove_comments=True)
    tree = etree.parse(xml_path, parser)
    root = tree.getroot()

    def norm_ws(s: str) -> str:
        s = re.sub(r"\s+", " ", s or "").strip()
        return s

    def text_of_first(xpath_expr: str) -> Optional[str]:
        nodes = root.xpath(xpath_expr)
        if not nodes:
            return None
        # node can be element or attribute string
        if isinstance(nodes[0], etree._Element):
            return norm_ws(nodes[0].xpath("string()"))
        return norm_ws(str(nodes[0]))

    # --------- metadata (coredata) ----------
    metadata: Dict[str, Any] = {}
    metadata["title"] = text_of_first('//*[local-name()="coredata"]/*[local-name()="title"][1]')
    metadata["doi"] = text_of_first('//*[local-name()="coredata"]/*[local-name()="doi"][1]')
    # Some Elsevier XML stores DOI in dc:identifier like "doi:..."
    if not metadata.get("doi"):
        dc_identifier = text_of_first('//*[local-name()="coredata"]/*[local-name()="identifier"][1]')
        if dc_identifier and "doi:" in dc_identifier.lower():
            metadata["doi"] = dc_identifier.split("doi:")[-1].strip()

    metadata["pii"] = text_of_first('//*[local-name()="coredata"]/*[local-name()="pii"][1]')
    metadata["publicationName"] = text_of_first(
        '//*[local-name()="coredata"]/*[local-name()="publicationName"][1]'
    )
    metadata["coverDate"] = text_of_first('//*[local-name()="coredata"]/*[local-name()="coverDate"][1]')

    # authors: dc:creator in coredata OR ce:author in originalText
    creators = root.xpath('//*[local-name()="coredata"]/*[local-name()="creator"]')
    metadata["authors"] = [norm_ws(c.xpath("string()")) for c in creators if norm_ws(c.xpath("string()"))] or None

    # keywords: sometimes in coredata subject, sometimes in originalText
    subjects = root.xpath('//*[local-name()="coredata"]/*[local-name()="subject"]')
    keywords = [norm_ws(s.xpath("string()")) for s in subjects if norm_ws(s.xpath("string()"))]
    metadata["keywords"] = keywords or None

    # abstract/description: commonly in dc:description
    metadata["abstract"] = text_of_first('//*[local-name()="coredata"]/*[local-name()="description"][1]')

    # --------- body sections ----------
    # Elsevier fulltext usually has ce:section with section-title and para
    section_nodes = root.xpath('//*[local-name()="section"]')

    def extract_paragraph_texts(section_el: etree._Element) -> List[str]:
        # Collect para/list-item etc. within this section, excluding nested sections
        # Approach: iterate over direct descendants until a nested section appears.
        paras: List[str] = []
        for child in section_el:
            lname = etree.QName(child).localname
            if lname == "section":
                # stop: nested section handled separately
                continue
            if lname in {"para", "simple-para"}:
                t = norm_ws(child.xpath("string()"))
                if t:
                    paras.append(t)
            # Some documents store content inside lists
            if lname in {"list", "ordered-list", "unordered-list"}:
                items = child.xpath('.//*[local-name()="list-item"]')
                for it in items:
                    t = norm_ws(it.xpath("string()"))
                    if t:
                        paras.append(t)
        return paras

    def section_title(section_el: etree._Element) -> Optional[str]:
        t = section_el.xpath('.//*[local-name()="section-title"][1]')
        if t:
            return norm_ws(t[0].xpath("string()"))
        return None

    def build_section_path(section_el: etree._Element) -> str:
        # Build hierarchical path based on ancestor sections' titles
        titles: List[str] = []
        for anc in section_el.xpath("ancestor-or-self::*[local-name()='section']"):
            st = section_title(anc)
            if st:
                titles.append(st)
        return " > ".join(titles) if titles else ""

    sections: List[Dict[str, Any]] = []

    if section_nodes:
        # Only keep "top-level" sections; nested ones will be included as separate entries anyway,
        # but path will preserve hierarchy.
        for sec in section_nodes:
            title = section_title(sec)
            paras = extract_paragraph_texts(sec)
            if not title and not paras:
                continue
            sections.append(
                {
                    "path": build_section_path(sec),
                    "title": title,
                    "paragraphs": paras,
                }
            )
    else:
        # Fallback: take all paragraphs in doc order
        paras = root.xpath('//*[local-name()="para" or local-name()="simple-para"]')
        fallback_paras = [norm_ws(p.xpath("string()")) for p in paras if norm_ws(p.xpath("string()"))]
        if fallback_paras:
            sections.append({"path": "", "title": None, "paragraphs": fallback_paras})

    # --------- figures & tables (captions) ----------
    def extract_caption(parent_xpath: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        nodes = root.xpath(parent_xpath)
        for n in nodes:
            # try common fields: label, caption, id/ref
            label = None
            ref = n.get("id") or n.get("ref") or n.get("xml:id")
            lab = n.xpath('.//*[local-name()="label"][1]')
            if lab:
                label = norm_ws(lab[0].xpath("string()"))

            cap = n.xpath('.//*[local-name()="caption"][1]')
            caption = norm_ws(cap[0].xpath("string()")) if cap else None

            # Some store title inside caption
            if not caption:
                cap2 = n.xpath('.//*[local-name()="figure-caption" or local-name()="table-caption"][1]')
                caption = norm_ws(cap2[0].xpath("string()")) if cap2 else None

            if label or caption:
                out.append({"label": label, "caption": caption, "ref": ref})
        return out

    figures = extract_caption('//*[local-name()="figure"]')
    tables = extract_caption('//*[local-name()="table" or local-name()="table-wrap"]')

    # --------- build full_text ----------
    full_text_parts: List[str] = []
    if metadata.get("title"):
        full_text_parts.append(f"TITLE: {metadata['title']}")
    if metadata.get("abstract"):
        full_text_parts.append(f"ABSTRACT: {metadata['abstract']}")

    for sec in sections:
        hdr = sec.get("path") or sec.get("title")
        if hdr:
            full_text_parts.append(f"\n## {hdr}")
        for p in sec.get("paragraphs", []):
            full_text_parts.append(p)

    full_text = "\n".join([p for p in full_text_parts if p])

    return {
        "metadata": metadata,
        "sections": sections,
        "figures": figures,
        "tables": tables,
        "full_text": full_text,
    }