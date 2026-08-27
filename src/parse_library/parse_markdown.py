"""Turn the Markdown documentation under src/ into records, one per section.

Reads every .md file beneath each directory in src/, plus any .md files
dropped directly in src/ itself, and writes one records file per source
into raw_text/, named <directory>_docs.jsonl (src_docs.jsonl for the
loose ones) - so documentation you gathered yourself only has to be
dropped under src/, in a directory or not, exactly like example code.

**A section is one record.** A heading, and the text between it and the
next heading of any level. Which means a heading with subsections keeps
only its own opening prose, and its children keep theirs: content lands in
exactly one record instead of parent records repeating everything beneath
them. Sections holding nothing but subheadings produce no record.

Parent sections are kept rather than skipped, because in practice their
opening prose is the most useful text in the file - what the tool is, how
to get started, what the output means. Only the deepest sections carry the
detail; the shallow ones carry the orientation.

The body lands in a "section_text" field rather than "doc". A docstring
read off a library object and the prose under a heading in a documentation
file are not the same thing, and chunk recipes select records by which
fields they have - so one name for both would have made any recipe naming
it apply to both kinds at once.

**Each record's heading path is stored with it**, since a section title
often means nothing alone. "Average bandwidth" is not a question anyone
asks. "Vela Performance Estimations > Estimated memory bandwidth > Average
bandwidth" is what they meant, and is what gets embedded.

Fenced code blocks are pulled out into a "code" field as well as being
left in place, so a section documenting a command is retrievable as the
runnable thing it shows.

Nothing here is specific to any project: sections, headings and fences are
Markdown, so retargeting is the two names in config. What the pipeline
cannot know is whether a heading is worth indexing, which is why every
section with a body becomes a record and the judgement stays with whoever
chose the files.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "config"))
from config import code_sources, raw_text_dir, src_dir

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
SPDX_RE = re.compile(r"SPDX-License-Identifier:\s*(.+?)\s*$", re.M)
COMMENT_BLOCK_RE = re.compile(r"<!--.*?-->", re.S)


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_license(text):
    """Projects state their licence in an SPDX header comment, which is
    both the most reliable place to find it and something no reader of the
    prose needs - so it is read once here and stripped from the body."""
    match = SPDX_RE.search(text)
    return match.group(1) if match else None


def split_headings(lines):
    """(level, title, line_number) for every heading, ignoring anything
    inside a fenced block.

    Fence tracking is the whole point: shell examples are full of comments
    starting with #, and a "# Install dependencies" inside one would
    otherwise open a section that swallows the rest of the file."""
    headings = []
    fence = None
    for number, line in enumerate(lines, 1):
        marker = FENCE_RE.match(line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token[0]
            elif token[0] == fence:
                fence = None
            continue
        if fence is not None:
            continue
        match = HEADING_RE.match(line)
        if match:
            headings.append((len(match.group(1)), match.group(2), number))
    return headings


def heading_paths(headings):
    """The chain of enclosing titles for each heading, by keeping a stack
    of open headings and dropping any at or below the current level."""
    paths = []
    stack = []
    for level, title, _ in headings:
        while stack and stack[-1][0] >= level:
            stack.pop()
        paths.append([open_title for _, open_title in stack] + [title])
        stack.append((level, title))
    return paths


def extract_code_blocks(body_lines):
    """The contents of each fenced block, joined. The fence markers and
    language tag are dropped - what is wanted is the runnable text."""
    blocks = []
    current = None
    fence = None
    for line in body_lines:
        marker = FENCE_RE.match(line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence, current = token[0], []
            elif token[0] == fence:
                if current:
                    blocks.append("\n".join(current))
                fence, current = None, None
            continue
        if current is not None:
            current.append(line)
    # An unclosed fence still holds real content; keeping it is better than
    # discarding the block because the file forgot a marker.
    if current:
        blocks.append("\n".join(current))
    return "\n\n".join(blocks) or None


def blank_leading_comments(text):
    """Empty out HTML comments before the first heading - the licence
    header, which every reader already knows and no one would search for.

    Blanked rather than deleted, keeping one newline per line removed, so
    every line number still refers to the same line of the file on disk.
    Deleting the block shifts everything after it, and since the only use
    of these numbers is following a record back to what it came from, a
    silently shifted one is worse than none.

    Comments further down are left alone: they sit among the prose and may
    be the only thing explaining it."""
    first_heading = HEADING_RE.search(text)
    cut = first_heading.start() if first_heading else len(text)
    head = COMMENT_BLOCK_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text[:cut])
    return head + text[cut:]


def parse_markdown_file(path, name_prefix, relative_name):
    raw = path.read_text(encoding="utf-8")
    license_name = extract_license(raw)
    lines = blank_leading_comments(raw).splitlines()

    headings = split_headings(lines)
    if not headings:
        return []
    paths = heading_paths(headings)

    records = []
    stem = Path(relative_name).stem
    for index, (level, title, line_number) in enumerate(headings):
        end = headings[index + 1][2] - 1 if index + 1 < len(headings) else len(lines)
        body_lines = lines[line_number:end]
        body = "\n".join(body_lines).strip()
        if not body:
            continue

        record = {
            "name": f"{name_prefix}/{stem}::{' > '.join(paths[index])}",
            "kind": "doc_section",
            "heading_path": paths[index],
            "heading_level": level,
            "section_text": body,
            "source": f"{relative_name}#L{line_number}-L{end}",
        }
        code = extract_code_blocks(body_lines)
        if code:
            record["code"] = code
        if license_name:
            record["license"] = license_name
        records.append(record)
    return records


def parse_source_dir(source_dir, name_prefix, recursive=True):
    records = []
    finder = source_dir.rglob if recursive else source_dir.glob
    for path in sorted(finder("*.md")):
        relative_name = path.relative_to(source_dir).as_posix()
        file_records = parse_markdown_file(path, name_prefix, relative_name)
        records.extend(file_records)
        print(f"  {relative_name}: {len(file_records)} sections")
    return records


def main():
    sources = code_sources()
    if not sources:
        print(f"No source directories under {src_dir()} - nothing to parse")
        return
    raw_text_dir().mkdir(parents=True, exist_ok=True)

    total = 0
    for source in sources:
        source_dir = source["src_dir"]
        recursive = source.get("recursive", True)
        finder = source_dir.rglob if recursive else source_dir.glob
        if not any(finder("*.md")):
            continue
        print(f"{source_dir.name}:")
        records = parse_source_dir(source_dir, source["name_prefix"], recursive)
        write_jsonl(records, source["docs_jsonl"])
        print(f"Wrote {len(records)} records to {source['docs_jsonl']}")
        total += len(records)

    if not total:
        print(f"No .md files under {src_dir()} - nothing to parse")


if __name__ == "__main__":
    main()
