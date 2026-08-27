"""Turn Python source files into code records via static AST analysis.

Reads .py files from every directory under src/ (whatever put them there
- see fetch_official_example_code.py), plus any .py files dropped
directly in src/ itself without a subdirectory around them, and never
touches the network. Each top-level function/class becomes one record,
the retrieval-sized unit, plus one module-context record per file holding
the imports, module-level constants, and __main__ glue.

Every record carries the target library's APIs it references, resolved
statically against the API records parse_api.py produced: references that
don't resolve land in unknown_refs instead of being silently trusted.
Which library is the target comes from config, so retargeting is a config
edit.

Python only - the stdlib ast module parses Python and nothing else. Other
languages need their own parser (libclang, tree-sitter, ...) in a sibling
parse_<language>_code.py.

ast only reads the code, never executes it - which is what makes it safe
to run over downloaded third-party sources. The tradeoff is that
dynamically constructed references (getattr(lib, name), aliasing through
a variable) are invisible, so apis_used is a conservative lower bound,
not an exhaustive list.
"""

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "config"))
from config import (
    EXAMPLES_MANIFEST_NAME,
    api_jsonl_path,
    code_sources,
    raw_text_dir,
    src_dir,
    parsed_module_name,
    parsed_module_version,
)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_module_aliases(tree, module_name):
    """Map local names to the module path they refer to, e.g.
    `import pycolmap` -> {"pycolmap": "pycolmap"},
    `from pycolmap import logging` -> {"logging": "pycolmap.logging"}."""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == module_name:
                    if alias.asname:
                        aliases[alias.asname] = alias.name
                    else:
                        aliases[module_name] = module_name
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == module_name:
                for alias in node.names:
                    aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def attribute_chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def collect_api_refs(node, aliases, known_names):
    """Resolve every library reference in this subtree to the longest name
    that actually exists in the API records."""
    used, unknown = set(), set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Attribute):
            continue
        chain = attribute_chain(sub)
        if not chain or chain[0] not in aliases:
            continue
        qualified = aliases[chain[0]].split(".") + chain[1:]
        for end in range(len(qualified), 1, -1):
            candidate = ".".join(qualified[:end])
            if candidate in known_names:
                used.add(candidate)
                break
        else:
            unknown.add(".".join(qualified))

    # Nested prefixes of a longer match are noise (pycolmap.Camera when
    # pycolmap.Camera.create was matched), keep only the most specific.
    used = {u for u in used if not any(other != u and other.startswith(u + ".") for other in used)}
    return sorted(used), sorted(unknown)


def segment_start_line(node):
    if getattr(node, "decorator_list", None):
        return node.decorator_list[0].lineno
    return node.lineno


def build_file_records(filename, source_text, module_name, known_names, name_prefix, provenance=None):
    tree = ast.parse(source_text)
    aliases = collect_module_aliases(tree, module_name)
    lines = source_text.splitlines()
    provenance = provenance or {}

    def make_record(name, code, refs_node, start, end):
        used, unknown = collect_api_refs(refs_node, aliases, known_names)
        # A segment that never touches the library has nothing to offer a
        # RAG built around it - argument parsers, plotting helpers, plain
        # dataclasses. They are findable by nothing useful and would only
        # dilute retrieval, so they are left out rather than stored with an
        # empty apis_used. The cost is that a helper supporting a workflow
        # goes with them; the surrounding functions that do call the
        # library still describe that workflow.
        if not used and not unknown:
            return None
        record = {"name": name, "kind": "example"}
        if provenance.get("url"):
            record["source"] = f"{provenance['url']}#L{start}-L{end}"
        if provenance.get("license"):
            record["license"] = provenance["license"]
        record["apis_used"] = used
        record["code"] = code
        if unknown:
            record["unknown_refs"] = unknown
        return record

    records = []
    segments = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    covered = set()
    for node in segments:
        start, end = segment_start_line(node), node.end_lineno
        # Claimed either way: a skipped segment's lines still belong to it,
        # not to the module-context record, which would otherwise absorb
        # whole unrelated functions into what should be imports and glue.
        covered.update(range(start, end + 1))
        code = "\n".join(lines[start - 1:end])
        record = make_record(f"{name_prefix}/{filename}::{node.name}", code, node, start, end)
        if record:
            records.append(record)

    # The module-context record stands for the file as a whole, so its
    # apis_used is scanned over the entire tree, not just the leftover
    # lines its code field holds. Scanning only those lines yields nothing
    # useful: imports and an `if __name__` call site contain no attribute
    # accesses to resolve, so every such record came out with an empty
    # apis_used while the file plainly used the library throughout.
    context_code = "\n".join(
        line for i, line in enumerate(lines, 1) if i not in covered
    ).strip()
    if context_code:
        record = make_record(f"{name_prefix}/{filename}", context_code, tree, 1, len(lines))
        if record:
            records.append(record)
    return records


def load_manifest(source_dir):
    manifest_path = source_dir / EXAMPLES_MANIFEST_NAME
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def parse_source_dir(source_dir, module_name, known_names, name_prefix, recursive=True):
    manifest = load_manifest(source_dir)
    urls = manifest.get("files", {})
    # One licence for the whole directory when every file came from the
    # same repository, per-file when they did not.
    licenses, shared_license = manifest.get("licenses", {}), manifest.get("license")

    records = []
    # Recursive by default: sources drawn from several repositories are
    # nested under a directory per repository, because basenames collide
    # across projects. src/ itself passes recursive=False - it already
    # recurses into every subdirectory as that subdirectory's own source,
    # so reading it recursively here too would parse every file twice.
    finder = source_dir.rglob if recursive else source_dir.glob
    for path in sorted(finder("*.py")):
        key = str(path.relative_to(source_dir))
        provenance = {"url": urls.get(key), "license": licenses.get(key, shared_license)}
        file_records = build_file_records(
            key, path.read_text(encoding="utf-8"), module_name, known_names, name_prefix, provenance
        )
        records.extend(file_records)
        print(f"  {key}: {len(file_records)} records")
    return records


def main():
    version = parsed_module_version
    known_names = {r["name"] for r in read_jsonl(api_jsonl_path())}

    sources = code_sources()
    if not sources:
        print(f"No source directories under {src_dir()} - nothing to parse")
        return
    raw_text_dir().mkdir(parents=True, exist_ok=True)

    for source in sources:
        source_dir = source["src_dir"]
        print(f"{source_dir.name}:")
        records = parse_source_dir(
            source_dir, parsed_module_name, known_names, source["name_prefix"], source.get("recursive", True)
        )
        write_jsonl(records, source["jsonl"])
        unknown_count = sum(1 for r in records if r.get("unknown_refs"))
        print(f"Wrote {len(records)} records to {source['jsonl']}")
        if unknown_count:
            print(f"WARNING: {unknown_count} records reference names that don't resolve against {version}")


if __name__ == "__main__":
    main()
