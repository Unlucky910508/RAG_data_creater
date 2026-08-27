from pathlib import Path

# Re-exported rather than kept here: which model servers to talk to
# depends on the machine, while the rest of this file describes the
# dataset. Importers still get everything from config, so nothing
# downstream needs to know about the split.
from AI_server_config import (  # noqa: F401
    API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
    VERIFY_SSL,
)

parsed_module_name = "vela"
# Stated here rather than read from the installed library, because almost
# nothing in the pipeline introspects it - the version is what names the
# dataset, so most steps need the string and not the module. parse_api.py
# and parse_signatures.py do import it, and check this matches what they
# found: a mismatch means writing one version's API into another
# version's file, which nothing downstream could tell had happened.
parsed_module_version = "3.7.0"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Not tracked by git, like data/: which parts of a library to index is a
# local judgement, and the filtering step is optional - absent a file it
# says so and stops rather than failing the pipeline.
FILTER_DIR = Path(__file__).resolve().parent.parent / "filter"

# Where the official example scripts live. Left as None, each is worked
# out from the package itself - the repository and licence from its PyPI
# metadata, the directory by looking through the repository for a
# conventionally named one holding .py files - so pointing the pipeline at
# another library is still only a change of parsed_module_name.
#
# Set any of them to take that decision by hand instead. Do that when the
# fetch reports it could not work something out: it stops and says which
# of these to fill in rather than guessing, since a wrong guess would
# quietly fill the dataset with another project's code.
#
# What resolution finds for pycolmap, as a reference for the shape:
#   EXAMPLES_GITHUB_REPO = "colmap/colmap"
#   EXAMPLES_PATH_IN_REPO = "python/examples"
#   EXAMPLES_LICENSE = "BSD-3-Clause"
EXAMPLES_GITHUB_REPO = None
EXAMPLES_PATH_IN_REPO = None
EXAMPLES_LICENSE = None
# Directory names a project might keep its examples under, tried in the
# repository tree when EXAMPLES_PATH_IN_REPO is None.
EXAMPLES_DIR_CONVENTIONS = (
    "examples",
    "example",
    "samples",
    "sample",
    "demos",
    "demo",
    "tutorials",
    "tutorial",
    "cookbook",
    "recipes",
)
# Downloaded .py files land here, with a _manifest.json recording where
# each came from. parse_python_code.py reads this directory; it never
# downloads.
EXAMPLES_MANIFEST_NAME = "_manifest.json"


# Community code that uses this library, for fetch_community_code.py.
# Trust here is not taken from stars or reputation: those are only coarse
# prefilters to keep the candidate set small. The decisive test is the static
# one the pipeline already does - resolving every reference against the
# API records of the installed version.
#
# Optional. Without it GitHub allows 60 requests an hour, which covers a
# run or so; with it, 5000. Needs no scopes at all - everything read here
# is public - so a token restricted to public repositories is enough.
GITHUB_TOKEN_FILE = Path(__file__).resolve().parent / "github_token.txt"


def load_github_token():
    if not GITHUB_TOKEN_FILE.exists():
        return None
    return GITHUB_TOKEN_FILE.read_text(encoding="utf-8").strip() or None


COMMUNITY_LICENSE_ALLOWLIST = {
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "Apache-2.0",
    "ISC",
    "0BSD",
    "Unlicense",
}
# Anything not on the list above is turned away, including GitHub's
# NOASSERTION - which means a licence file exists but could not be matched
# to a known one, so its terms are unknown until somebody reads them.
#
# These were once collected for review instead. Dropping them is the safer
# trade: code that reaches the dataset gets quoted to an agent and can end
# up shaping what it writes, and obligations do not stop applying because
# a file was split into functions first. The corpus is easier to widen -
# raise COMMUNITY_SEARCH_PAGES, lower COMMUNITY_MIN_STARS - than to audit
# after the fact.
#
# The cost is real: colmap's own COPYING.txt reads NOASSERTION because its
# BSD text carries a preamble about dependencies, so the upstream
# project's benchmark code is turned away too. Set EXAMPLES_* or add an
# allowlist entry if you have read a licence and want it accepted.
COMMUNITY_MIN_STARS = 200
COMMUNITY_MAX_AGE_DAYS = 730
# Pages of grep.app results to walk. Raising this is the cheap way to make
# up for a strict licence policy: search wider rather than accept less
# certain code.
COMMUNITY_SEARCH_PAGES = 20
# Measured per function, because that is the unit parse_python_code.py
# turns into a record: the question is whether a file contains at least
# one function that would make a good one. Counting distinct APIs across a
# whole file instead scores a 2100-line grab-bag touching seven APIs the
# same as a 150-line function using eleven, and only the second is worth
# quoting. Three distinct APIs in one function is enough to be showing
# composition rather than a single call.
COMMUNITY_MIN_APIS_PER_FUNCTION = 3
# Any reference that does not resolve against the installed version's API
# records is evidence the file targets a different version.
COMMUNITY_MAX_UNKNOWN_REFS = 0


# Everything generated for one library at one version lives under a single
# directory, split by what the contents are rather than by which script
# wrote them:
#
#   <module>_<version>/
#     src/              .py files, one directory per source
#       official/       from fetch_official_example_code.py
#       community/      from fetch_community_code.py
#       <yours>/        anything you put there
#     raw_text/         one jsonl of records per source, plus api.jsonl
#     chunked_text/     the same, split into embedding chunks
#     chroma/           the vector store
#
# The steps between those directories work by looking at what is there,
# not by consulting a list here, so a directory you create under src/ is
# parsed and a jsonl you write into raw_text/ is chunked without anything
# being registered first.


def library_dir():
    return DATA_DIR / f"{parsed_module_name}_{parsed_module_version}"


def src_dir():
    return library_dir() / "src"


def raw_text_dir():
    return library_dir() / "raw_text"


def chunked_text_dir():
    return library_dir() / "chunked_text"


def official_src_dir():
    return src_dir() / "official"


def community_src_dir():
    return src_dir() / "community"


def community_candidates_path():
    """What the community fetch kept and what it turned away. Deliberately
    outside raw_text/, which holds records: this is a report about a fetch,
    and anything in raw_text/ would be chunked and indexed."""
    return library_dir() / "community_candidates.jsonl"


def api_filter_path(mode):
    """The prefix list for one filtering policy. Named after the policy -
    exclude.py, keep.py - so asking for a policy is enough to say which
    file to read."""
    return FILTER_DIR / f"{parsed_module_name}_{parsed_module_version}" / f"{mode}.py"


def api_jsonl_path():
    return raw_text_dir() / "api.jsonl"


def code_sources():
    """Every directory under src/, each becoming one records file named
    after it, plus src/ itself for files dropped there directly rather
    than under a subdirectory - so a file doesn't need a folder built
    around it just to be picked up. The directory name (or "src" for
    that last one) is also the prefix its records carry, so a hit says
    where the code came from.

    Subdirectories are read recursively (community/ nests one directory
    per repository), but src/ itself is only read one level deep - it
    already recurses into every subdirectory as its own source, so
    reading it recursively too would parse every file under src/ twice.
    A subdirectory literally named "src" would collide with this jsonl
    name; nothing stops that, it just is not worth guarding against.

    Code and documentation from the same directory are kept in separate
    files - parse_python_code.py writes "jsonl", parse_markdown.py writes
    "docs_jsonl" - because one directory can hold both, and each step
    rewrites its own file whole. Sharing one would mean whichever ran
    second erased the other's work."""
    root = src_dir()
    if not root.exists():
        return []
    sources = [
        {
            "src_dir": directory,
            "jsonl": raw_text_dir() / f"{directory.name}.jsonl",
            "docs_jsonl": raw_text_dir() / f"{directory.name}_docs.jsonl",
            "name_prefix": directory.name,
            "recursive": True,
        }
        for directory in sorted(root.iterdir())
        if directory.is_dir()
    ]
    sources.append({
        "src_dir": root,
        "jsonl": raw_text_dir() / f"{root.name}.jsonl",
        "docs_jsonl": raw_text_dir() / f"{root.name}_docs.jsonl",
        "name_prefix": root.name,
        "recursive": False,
    })
    return sources


def record_jsonl_paths():
    """Every records file. Read from disk rather than assembled from a
    list, so a jsonl written there by hand is picked up like any other."""
    root = raw_text_dir()
    return sorted(root.glob("*.jsonl")) if root.exists() else []


def chunks_jsonl_path_for(record_jsonl_path):
    """The chunk file for a records file: same name, _chunks appended,
    under chunked_text/."""
    return chunked_text_dir() / f"{record_jsonl_path.stem}_chunks.jsonl"


def chunk_jsonl_paths():
    """Every chunk file, again by looking rather than by listing."""
    root = chunked_text_dir()
    return sorted(root.glob("*.jsonl")) if root.exists() else []


def chroma_dir():
    """One store per library and version, alongside the data it indexes,
    so a rebuild is a directory to delete."""
    return library_dir() / "chroma"


# BAAI/bge-m3 (like most embedding models) is trained/evaluated for cosine
# similarity, not Chroma's default squared-L2 distance. Only affects a
# collection at creation time - set here so load_vectordb.py and
# search.py can never create it with mismatched metrics.
CHROMA_DISTANCE_METRIC = "cosine"


def chroma_collection_name():
    """The collection inside that database. Now that the directory is
    already per-library-and-version this is belt and braces, but naming it
    the same way costs nothing and keeps a store readable if one is ever
    pointed at by hand."""
    return f"{parsed_module_name}_{parsed_module_version}"


# Ceiling on results per search. Each hit carries a full record's worth of
# text, so a caller asking for many of them floods the agent's context
# with loosely-related matches. Requests above this are clamped, not
# rejected - an over-eager caller still gets an answer.
MAX_TOP_K = 5


# Recipe per chunk_type. All three lists name fields from
# FIELD_EXTRACTORS in src/common/record_fields.py; recombining existing
# fields is a config edit, only a genuinely new field needs code.
#
#   embedding_fields - concatenated into the text that gets embedded and
#                      matched against queries (parse_chunks.py)
#   required         - fields a record must actually have for this
#                      chunk_type to exist at all, so e.g. a class with no
#                      signatures gets no "signature" chunk just because
#                      it has a name
#   return_fields    - concatenated into the text handed back to the
#                      caller on a hit (search.py). Assembled at query
#                      time, so editing this takes effect immediately
#                      without re-embedding anything.
#
# Splitting the two lists means a chunk can be *found* by one kind of text
# and *answered* with another - e.g. match on a prose explanation but hand
# back the source code.
# Longest embedding text a chunk may carry. Over this, a chunk is split
# into several - all keeping the same record_id and the same full
# return_text, so the record is found through whichever piece matches and
# still answered with the whole of it. search.py already collapses
# multiple chunks of one record into a single hit.
#
# The limit is not the model's context window, which nothing here comes
# near. It is that one vector averages everything it was given: a section
# listing fifty operators, or a two-hundred-line function, produces a
# vector sharply about none of them. Splitting is preferred to summarising
# because the detail is what gets asked about - operator names, config
# keys, the order of API calls - and a summary short enough to help is
# short enough to have dropped it.
#
# Set to None to store every chunk whole.
CHUNK_MAX_CHARS = 2000
# Boundaries a split may fall on, strongest first; the last one is why a
# split can always be made. Line starts throughout, so no piece ever cuts
# mid-line.
CHUNK_SPLIT_PATTERNS = (
    r"^\s*$",            # blank line - a paragraph or block ended
    r"^\s*```|^\s*~~~",  # a fenced block opens or closes
    r"^#{1,6}\s",         # a heading
    r"^\s*\|",           # a table row
    r"^\s*[-*+]\s",      # a list item
    r"^\S",              # an unindented line - a top-level statement
    r"",                 # any line at all
)

CHUNK_FIELDS = {
    "explanation": {
        "embedding_fields": ["name", "doc", "explanation"],
        "required": ["explanation"],
        "return_fields": ["name", "kind", "signatures", "parameter_names", "doc", "explanation"],
    },
    "signature": {
        "embedding_fields": ["name", "signatures", "parameter_names"],
        "required": ["signatures"],
        "return_fields": ["name", "kind", "signatures", "parameter_names", "doc", "explanation"],
    },
    "example_workflow": {
        "embedding_fields": ["name", "apis_used"],
        # apis_used as well as code, because it is the whole of what this
        # chunk matches on: a record without it embeds its name and nothing
        # else, which matches everything weakly and answers nothing. Code
        # from documentation is exactly that case - a CLI invocation calls
        # no API - and it is still reachable through the "example" chunk,
        # which embeds the code itself.
        "required": ["code", "apis_used"],
        "return_fields": ["name", "source", "apis_used"],
    },
    "example": {
        "embedding_fields": ["name", "apis_used", "code"],
        "required": ["code"],
        "return_fields": ["name", "source", "code"],
    },
    # Prose from the documentation (parse_markdown.py). Required on "doc"
    # rather than "explanation" because the body already is a written
    # explanation - one a human wrote about their own software - so asking
    # an LLM to restate it would only add a way for it to be wrong.
    # heading_path is embedded because a section title is often meaningless
    # alone: "Average bandwidth" is a question nobody asks, while
    # "Vela Performance Estimations > Estimated memory bandwidth >
    # Average bandwidth" is what they meant.
    "doc_section": {
        "embedding_fields": ["name", "heading_path", "section_text"],
        # Strips markdown link syntax from the embedded text only, keeping
        # the link text and dropping the target. In a reference table the
        # targets are in-document anchors repeated on every row - 80% of
        # the longest section here, and near-identical strings at that, so
        # they crowd out the operator names that are the reason anyone
        # would look. The answer keeps the original markdown, links and
        # all, since a reader follows them.
        #
        # Named per recipe rather than applied everywhere because it is
        # only safe on markdown: in Python, `handlers[key](arg)` matches
        # the same pattern and would be gutted.
        "embedding_cleanup": "markdown",
        # section_text, not doc: a docstring read off a library object and
        # the prose under a heading in a documentation file are different
        # things, and recipes match on which fields a record has rather
        # than on what kind it is. Naming both "doc" was enough to make
        # this recipe apply to every API record that had one.
        "required": ["section_text"],
        "return_fields": ["name", "heading_path", "section_text", "source"],
    },
}


