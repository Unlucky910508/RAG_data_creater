"""Split each enriched API record into one or more embedding chunks.

Each record gets indexed under multiple text representations (its
explanation, its signature) so a conceptual query ("how do I set the
camera model") and a precise one ("does this take a min_num_trials
argument") can each match a chunk suited to that query style. Chunks only
carry record_id + chunk_type + embedded_text - the full record already
lives in the API jsonl and is looked up by record_id after a chunk
matches, so nothing is duplicated here.

Which fields make up which chunk_type is a data-only recipe
(CHUNK_FIELDS in config/config.py), not code - this step reads each
chunk_type's embedding_fields and renders them through the shared
vocabulary in src/common/record_fields.py. The same recipe's
return_fields are used at the other end of the pipeline by search.py,
which is why that vocabulary is shared rather than local.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "config"))
from config import (
    CHUNK_FIELDS,
    CHUNK_MAX_CHARS,
    CHUNK_SPLIT_PATTERNS,
    chunked_text_dir,
    chunks_jsonl_path_for,
    raw_text_dir,
    record_jsonl_paths,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from record_fields import build_text, has_required_fields


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
TABLE_RULE_RE = re.compile(r"^\|[\s\-:|]+\|\s*$", re.M)
BLANK_RUN_RE = re.compile(r"\n{3,}")

CLEANERS = {}


def clean_markdown(text):
    """Markup that carries nothing a reader of the text would say aloud.

    Link targets go and the link text stays: in a reference table the
    targets are in-document anchors, repeated near-identically on every
    row, and they drown out the names the row is about. Table rules go for
    the same reason. Only the embedded text is treated this way - the
    answer keeps the markdown it was written in."""
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = TABLE_RULE_RE.sub("", text)
    return BLANK_RUN_RE.sub("\n\n", text).strip()


CLEANERS["markdown"] = clean_markdown


def split_points(lines):
    """Every line start that a split may fall on, with how strong a
    boundary it is - lower sorts first, and the weakest matches any line.

    Boundaries are enumerated rather than chosen so that whatever picks
    between them, here or later, is choosing from positions the text
    itself offers. Nothing can propose a split the structure does not
    already permit."""
    ranked = {}
    for index, line in enumerate(lines):
        for rank, pattern in enumerate(CHUNK_SPLIT_PATTERNS):
            if re.search(pattern, line):
                ranked[index] = rank
                break
    return ranked


def merge_orphans(pieces, limit):
    """Fold a piece too small to mean anything into the one before it.

    Preferring the strongest boundary in reach will happily cut just
    before a closing fence or bracket, leaving it alone as the next piece
    - vectors for ")" and "```", which match nothing and cost the same as
    any other. Merging is backwards so the fragment rejoins what it
    belonged to, and adjacent pieces are concatenated, so the text is
    still exactly what it was.

    A merge can carry a piece past the limit by less than a tenth of it.
    That is the smaller error: a slightly long piece embeds fine, while an
    orphan is a vector that should never have existed."""
    minimum = max(1, limit // 10)
    merged = []
    for piece in pieces:
        if merged and len(piece.strip()) < minimum:
            merged[-1] += piece
        else:
            merged.append(piece)
    # The first piece has nothing behind it, so it merges forwards
    # instead. This is the record's name left on its own when what follows
    # is one unsplittable line - the name belongs with the thing it names.
    while len(merged) > 1 and len(merged[0].strip()) < minimum:
        merged[0] += merged.pop(1)
    return merged


def split_text(text, limit):
    """The text as pieces no longer than the limit, cut only at line
    starts.

    Each cut is the strongest boundary available within reach, and among
    equals the furthest - so a piece runs as long as it may and still ends
    somewhere the text meant to end. Pieces are verbatim and in order:
    joined back together they are the original, which is what makes this
    safe to do to text nobody will read again before it is embedded."""
    if not limit or len(text) <= limit:
        return [text]

    lines = text.splitlines(keepends=True)
    ranked = split_points(lines)
    starts = [0]
    for index, line in enumerate(lines):
        starts.append(starts[-1] + len(line))

    pieces = []
    begin = 0
    while begin < len(lines):
        budget = starts[begin] + limit
        reachable = [i for i in range(begin + 1, len(lines) + 1) if starts[i] <= budget]
        if not reachable:
            # A single line longer than the limit: take it whole rather
            # than cut inside it.
            end = begin + 1
        else:
            best = min(ranked.get(i, len(CHUNK_SPLIT_PATTERNS)) for i in reachable)
            end = max(i for i in reachable if ranked.get(i, len(CHUNK_SPLIT_PATTERNS)) == best)
        pieces.append("".join(lines[begin:end]))
        begin = end

    pieces = merge_orphans(pieces, limit)

    # The guarantee this rests on, checked rather than assumed: the pieces
    # are the text, in order, with nothing added, dropped or altered.
    # Whatever chooses the boundaries - these patterns now, something with
    # more judgement later - cannot quietly rewrite what it divides.
    if "".join(pieces) != text:
        raise AssertionError(f"split lost content: {len(''.join(pieces))} chars from {len(text)}")

    # Edges trimmed only now, after that holds: leading and trailing
    # whitespace is nothing to embed, and dropping it cannot move content
    # between pieces.
    return [piece for piece in (piece.strip() for piece in pieces) if piece]


def build_chunks(record):
    """Both texts for a chunk: the one it is found by, and the one it is
    answered with.

    The answer is rendered here rather than at query time so that it can
    be stored in the vector database, which then holds everything a
    search needs. That is what lets the serving side be handed over as a
    directory and a store, with no records file beside it.

    The cost is that changing a return_fields recipe means running this
    and load_vectordb.py again, where assembling at query time would have
    taken effect on the next question."""
    chunks = []
    for chunk_type, spec in CHUNK_FIELDS.items():
        if not has_required_fields(record, spec["required"]):
            continue
        text = build_text(record, spec["embedding_fields"])
        if not text:
            continue
        cleaner = CLEANERS.get(spec.get("embedding_cleanup"))
        if cleaner:
            text = cleaner(text)
        # Falls back to the embedded text so a recipe without a return of
        # its own still answers with something. Built before the split and
        # shared by every piece: a piece is a way of finding the record,
        # not a part of the answer.
        return_text = build_text(record, spec.get("return_fields") or spec["embedding_fields"]) or text
        for part, piece in enumerate(split_text(text, CHUNK_MAX_CHARS)):
            chunks.append({
                "record_id": record["name"],
                "chunk_type": chunk_type,
                "part": part,
                "embedded_text": piece,
                "return_text": return_text,
            })
    return chunks


def main():

    record_paths = record_jsonl_paths()
    if not record_paths:
        print(f"No records under {raw_text_dir()} - nothing to chunk")
        return
    chunked_text_dir().mkdir(parents=True, exist_ok=True)

    for record_path in record_paths:
        records = read_jsonl(record_path)
        chunks = []
        for record in records:
            chunks.extend(build_chunks(record))
        output_path = chunks_jsonl_path_for(record_path)
        write_jsonl(chunks, output_path)
        print(f"Wrote {len(chunks)} chunks from {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
