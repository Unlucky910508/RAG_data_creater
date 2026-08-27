"""Drop the explanation field from selected records so the next
parse_explanations.py run regenerates them.

Explanations are generated from whatever fields a record had at the time.
When an enricher later adds or changes a field, the explanations written
before that were produced from a context the model never saw the new
field in, and they go stale - but only for the records that actually
gained something. Deleting the whole jsonl and starting over would
regenerate every record, most of which are unchanged, at a real cost in
LLM calls.

So this clears explanations selectively: name the fields whose arrival
invalidates an explanation, and only records carrying one of them are
reset. parse_explanations.py skips records that already have an
explanation, so it then regenerates exactly the cleared ones.

    # after parse_signatures.py started emitting a "doc" field
    invalidate_explanations.py data/pycolmap_4.1.0/raw_text/api.jsonl --when-field doc

    # see what would happen first
    invalidate_explanations.py data/pycolmap_4.1.0/raw_text/api.jsonl --when-field doc --dry-run

    # after a prompt change, where every record's output is stale
    invalidate_explanations.py data/pycolmap_4.1.0/raw_text/api.jsonl --all
"""

import argparse
import json
import sys
from pathlib import Path

EXPLANATION_FIELD = "explanation"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def should_invalidate(record, trigger_fields):
    """True if the record carries any of the named fields. No fields named
    means every record with an explanation qualifies (--all)."""
    if trigger_fields is None:
        return True
    return any(record.get(field) for field in trigger_fields)


def invalidate(records, trigger_fields):
    """Strip explanations in place, returning the names that lost one."""
    cleared = []
    for record in records:
        if record.get(EXPLANATION_FIELD) and should_invalidate(record, trigger_fields):
            del record[EXPLANATION_FIELD]
            cleared.append(record["name"])
    return cleared


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("jsonl_path", type=Path, help="record jsonl to rewrite in place")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--when-field",
        nargs="+",
        metavar="FIELD",
        help="clear records carrying any of these fields, e.g. --when-field doc",
    )
    selector.add_argument("--all", action="store_true", help="clear every explanation")
    parser.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.jsonl_path.exists():
        sys.exit(f"No such file: {args.jsonl_path}")

    records = read_jsonl(args.jsonl_path)
    explained = sum(1 for r in records if r.get(EXPLANATION_FIELD))
    cleared = invalidate(records, None if args.all else args.when_field)

    for name in cleared[:10]:
        print(f"  {name}")
    if len(cleared) > 10:
        print(f"  ... and {len(cleared) - 10} more")

    verb = "would clear" if args.dry_run else "cleared"
    print(f"{verb} {len(cleared)} of {explained} explanations in {len(records)} records")

    if args.dry_run:
        print("Dry run - file untouched.")
        return
    if not cleared:
        print("Nothing to do - file untouched.")
        return

    write_jsonl(records, args.jsonl_path)
    print(f"Rewrote {args.jsonl_path}. Run parse_explanations.py to regenerate.")


if __name__ == "__main__":
    main()
