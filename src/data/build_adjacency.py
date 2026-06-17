"""Script to regenerate wa_adjacency.json from the US Cencus county adjacency data.
    Source: pip install county-adjacency
    python -m src.data.build_adjacency
"""
import importlib.util
import json
from pathlib import Path

STATE_ABBR = "WA"
STATE_FULL = "Washington"
OUT = Path("src/data/wa_adjacency.json")

def load_census_table():
    """
    Return package bundled dict.
    """
    try:
        from county_adjacency import supported_areas, get_neighboring_areas
        table = {a: list(get_neighboring_areas(a)) for a in supported_areas()}
        if table:
            return table
    except Exception:
        pass

    spec = importlib.util.find_spec("county_adjacency") # locate
    if spec is None or spec.origin is None:
        raise ImportError("Run: pip install county-adjacency")
    merged = {}
    for f in sorted(Path(spec.origin).parent.glob("data*.py")):
        s = importlib.util.spec_from_file_location(f.stem, f)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        data = getattr(m, "united_states_adjacency_data", {})
        for name, rec in data.items():
            merged[name] = rec["adjacent"] if isinstance(rec, dict) else rec
        if not merged:
            raise RuntimeError(
                    "no data found"
                    "Try: pip install --upgrade county-adjacency"
                )
            return merged

def _is_wa(name):
    return name.endswith(f", {STATE_ABBR}") or name.endswith(f", {STATE_FULL}")

def _base(name):                         # King County, WA...Washington -> "King"
    return (name.replace(f", {STATE_FULL}", "")
                .replace(f", {STATE_ABBR}", "")
                .replace(" County", "").strip())

def build(table):
    # seed every WA county
    counties = {_base(n) for n in table if _is_wa(n)}
    # undirected edge if either county lists the other
    edges = set()
    for name, neighbors in table.items():
        if not _is_wa(name):
            continue
        for nb in neighbors:
            if _is_wa(nb) and nb != name:
                edges.add(frozenset((_base(name), _base(nb))))

    adj = {c: [] for c in counties}
    for e in edges:
        a, b = tuple(e)
        adj[a].append(b)
        adj[b].append(a)
    return {c: sorted(adj[c]) for c in sorted(adj)}

if __name__ == "__main__":
    table = load_census_table()
    if not any(_is_wa(n) for n in table):
        print(f"Found {len(table)} areas but none matched"
              f"', {STATE_ABBR}' or ', {STATE_FULL}'. Sample of actual names:")
        for n in list(table)[:12]:
            print("   ", repr(n))
        raise SystemExit("format above")

    adj = build(table)
    OUT.write_text(json.dumps(adj, indent=1))
    edges = sum(len(v) for v in adj.values()) // 2
    print(f"wrote {OUT}: {len(adj)} counties, {edges} edges")





