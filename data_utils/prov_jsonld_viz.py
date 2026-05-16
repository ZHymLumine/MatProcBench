#!/usr/bin/env python3
"""
prov_jsonld_viz.py — Local PROV-JSONLD visualizer
Reads MatPROV.jsonl or a standalone PROV-JSONLD file,
renders provenance graphs to SVG/PDF, and optionally exports JSON-LD.

Usage:
  # Visualize all entries in MatPROV.jsonl
  python3 prov_jsonld_viz.py data/raw_data/MatPROV.jsonl -o output_viz -f svg

  # Single entry by index
  python3 prov_jsonld_viz.py data/raw_data/MatPROV.jsonl -o output_viz -f pdf --index 0

  # A standalone PROV-JSONLD file (single graph or list)
  python3 prov_jsonld_viz.py my_graph.json -o output_viz -f svg --jsonld-out
"""

import json
import argparse
import sys
from pathlib import Path
import html
import graphviz

# ── Node visual styles (mirrors the Cytoscape.js styles in the original) ──────
NODE_STYLES = {
    "Material": {
        "shape": "box",
        "style": "rounded,filled",
        "fillcolor": "#8ECAE6",
        "fontcolor": "#000000",
        "color": "#8ECAE6",   # no visible border (same as fill)
    },
    "Tool": {
        "shape": "diamond",
        "style": "filled",
        "fillcolor": "#F4A361",
        "fontcolor": "#000000",
        "color": "#F4A361",
    },
    "Agent": {
        "shape": "hexagon",
        "style": "filled",
        "fillcolor": "#9C27B0",
        "fontcolor": "#ffffff",
        "color": "#9C27B0",
    },
    "Activity": {
        "shape": "ellipse",
        "style": "filled",
        "fillcolor": "#91BF6D",
        "fontcolor": "#000000",
        "color": "#91BF6D",
    },
    "Entity": {
        "shape": "box",
        "style": "rounded,filled",
        "fillcolor": "#CFE2FF",
        "fontcolor": "#000000",
        "color": "#CFE2FF",
    },
}

META_NODE_STYLE = {
    "shape": "box",
    "style": "rounded,filled",
    "fillcolor": "#DDDDDD",
    "fontcolor": "#333333",
    "color": "#DDDDDD",     # no visible border
    "fontsize": "10",
}


def _get_label(item: dict) -> str:
    labels = item.get("label", [])
    if labels and isinstance(labels, list):
        return labels[0].get("@value", item.get("@id", ""))
    if isinstance(labels, str):
        return labels
    return item.get("@id", "")


def _get_node_type(item: dict) -> str:
    t = item.get("@type", "")
    if t == "Entity":
        subtype_list = item.get("type", [])
        if subtype_list and isinstance(subtype_list, list):
            subtype = subtype_list[0].get("@value", "")
        else:
            subtype = str(subtype_list) if subtype_list else ""
        if subtype == "tool":
            return "Tool"
        if subtype == "material":
            return "Material"
    return t


def _collect_meta(item: dict) -> list[tuple[str, str]]:
    """Return list of (attr_name, value) for matprov: attributes with non-null values."""
    result = []
    for key, val in item.items():
        if not key.startswith("matprov:"):
            continue
        if isinstance(val, list) and val:
            v = val[0].get("@value", "")
        elif isinstance(val, str):
            v = val
        else:
            continue
        if v:
            result.append((key.replace("matprov:", ""), v))
    return result


def build_dot_graph(prov_jsonld: dict, graph_label: str = "") -> graphviz.Digraph:
    """Convert a single PROV-JSONLD @graph to a Graphviz Digraph."""
    dot = graphviz.Digraph(comment=graph_label)
    dot.attr(
        rankdir="TB",
        label=html.escape(graph_label),
        fontsize="13",
        fontname="Arial",
        splines="ortho",
        nodesep="0.4",
        ranksep="0.6",
    )
    dot.attr("node", fontname="Arial", fontsize="12", margin="0.15,0.08")
    dot.attr("edge", fontname="Arial", fontsize="10", color="#000000",
             arrowsize="0.8")

    items = prov_jsonld.get("@graph", [])
    relation_types = {"Usage", "Generation", "Association", "Derivation",
                      "WasGeneratedBy", "Used", "WasAssociatedWith",
                      "WasAttributedTo", "WasDerivedFrom"}

    for item in items:
        node_id = item.get("@id")
        if not node_id:
            continue
        node_type = _get_node_type(item)
        if node_type in relation_types:
            continue

        label = _get_label(item)
        meta = _collect_meta(item)
        style = NODE_STYLES.get(node_type, NODE_STYLES["Entity"]).copy()
        dot.node(node_id, label=label, **style)

        # Metadata as a separate grey node connected by a dotted edge (matches web version)
        if meta:
            meta_id = f"{node_id}__meta"
            meta_label = "\n".join(f"{k}: {v}" for k, v in meta)
            dot.node(meta_id, label=meta_label, **META_NODE_STYLE)
            dot.edge(node_id, meta_id,
                     style="dotted", color="#B0BEC5", arrowhead="none",
                     penwidth="1")

    # Render edges from relation nodes
    for item in items:
        t = item.get("@type", "")
        if t == "Usage":
            src = item.get("entity") or item.get("prov:entity")
            tgt = item.get("activity") or item.get("prov:activity")
            if src and tgt:
                dot.edge(src, tgt)
        elif t == "Generation":
            src = item.get("activity") or item.get("prov:activity")
            tgt = item.get("entity") or item.get("prov:entity")
            if src and tgt:
                dot.edge(src, tgt)
        elif t == "Association":
            src = item.get("agent") or item.get("prov:agent")
            tgt = item.get("activity") or item.get("prov:activity")
            if src and tgt:
                dot.edge(src, tgt, label="assoc", style="dashed")
        elif t == "Derivation":
            src = item.get("usedEntity") or item.get("prov:usedEntity")
            tgt = item.get("generatedEntity") or item.get("prov:generatedEntity")
            if src and tgt:
                dot.edge(src, tgt, label="derived", style="dashed")

    return dot


def load_entries(path: Path) -> list[dict]:
    """Load PROV-JSONLD entries.
    Supports:
      - .jsonl  → each line is { doi, label, prov_jsonld: {...} }
      - .json   → either a single PROV-JSONLD dict with @graph,
                  or a list of such dicts,
                  or a list of { label, @graph } objects (original viewer format)
    """
    entries = []
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prov = obj.get("prov_jsonld", obj)
                label = obj.get("label", obj.get("doi", ""))
                entries.append({"label": label, "prov_jsonld": prov})
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if "@graph" in item:
                    entries.append({
                        "label": item.get("label", ""),
                        "prov_jsonld": item,
                    })
                else:
                    entries.append({"label": item.get("label", ""), "prov_jsonld": item})
        elif isinstance(data, dict):
            if "@graph" in data:
                entries.append({"label": data.get("label", path.stem), "prov_jsonld": data})
            else:
                # Possibly a { label, prov_jsonld } wrapper
                prov = data.get("prov_jsonld", data)
                entries.append({"label": data.get("label", path.stem), "prov_jsonld": prov})

    return entries


def _safe_filename(s: str, max_len: int = 60) -> str:
    safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in s)
    return safe.strip().replace(" ", "_")[:max_len]


def visualize(
    entries: list[dict],
    output_dir: Path,
    fmt: str = "svg",
    jsonld_out: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(entries)

    for idx, entry in enumerate(entries):
        label = entry.get("label", f"entry_{idx}")
        prov = entry["prov_jsonld"]
        stem = f"{idx:04d}_{_safe_filename(label)}" if label else f"entry_{idx:04d}"

        # Build and render graph
        dot = build_dot_graph(prov, graph_label=label)
        out_path = output_dir / stem

        try:
            dot.render(filename=str(out_path), format=fmt, cleanup=True)
            rendered = f"{out_path}.{fmt}"
            print(f"[{idx+1}/{total}] {rendered}")
        except graphviz.backend.execute.CalledProcessError as e:
            print(f"[{idx+1}/{total}] ERROR rendering '{label}': {e}", file=sys.stderr)

        # Optionally save JSON-LD
        if jsonld_out:
            jld_path = output_dir / f"{stem}.jsonld"
            with open(jld_path, "w", encoding="utf-8") as f:
                json.dump(prov, f, ensure_ascii=False, indent=2)
            print(f"           → {jld_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PROV-JSONLD local visualizer — outputs SVG/PDF vector graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input file: MatPROV.jsonl, or a standalone .json / .jsonld",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("prov_viz_output"),
        help="Output directory (default: prov_viz_output/)",
    )
    parser.add_argument(
        "-f", "--format",
        default="svg",
        choices=["svg", "pdf", "png", "eps"],
        help="Output format (default: svg)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Only process this entry index (0-based) — useful for quick preview",
    )
    parser.add_argument(
        "--jsonld-out",
        action="store_true",
        help="Also save each graph as a .jsonld file in the output directory",
    )
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Error: '{args.input}' not found")

    print(f"Loading entries from {args.input} ...")
    entries = load_entries(args.input)
    print(f"Found {len(entries)} entries.")

    if args.index is not None:
        if args.index < 0 or args.index >= len(entries):
            sys.exit(f"Error: --index {args.index} out of range (0–{len(entries)-1})")
        entries = [entries[args.index]]

    visualize(entries, args.output, fmt=args.format, jsonld_out=args.jsonld_out)
    print(f"\nDone. Output saved to: {args.output}/")


if __name__ == "__main__":
    main()
