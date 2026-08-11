"""Fetches an English Wikipedia plot summary for every movie in the catalog.

WHY PLOT SUMMARIES AND NOT DIALOGUE TRANSCRIPTS
    The agent has to answer "no deaths" and "not scary". In animation those
    events are almost always visual, not spoken: nobody says "Mufasa is dead"
    on screen, and a subtitle transcript therefore carries no usable signal.
    The Wikipedia plot section narrates them explicitly -- for The Lion King
    it reads "...Scar betrays him by throwing him into the stampede to his
    death". Narration beats dialogue for this specific question.

    Coverage and licensing also favour it. English Wikipedia has a detailed
    plot section for essentially every Disney/Pixar theatrical feature, while
    screenplay corpora skew to live-action and subtitle corpora need
    credentials. Wikipedia text is CC BY-SA, so it is safe to store and quote
    from a public repository; scraped subtitles and screenplays are not.

HOW MOVIES ARE MATCHED
    Through Wikidata, on IMDb id (property P345), never on title. Disney's
    catalog is full of live-action remakes sharing an exact title with their
    animated original -- The Lion King, The Jungle Book, Aladdin, Cinderella,
    Dumbo, Pinocchio, Beauty and the Beast, The Little Mermaid, Mulan, Lady
    and the Tramp. Title matching would silently attach the wrong film's plot
    and make the agent cite a real scene from the wrong movie. Rows with no
    IMDb id are reported as unmatched rather than guessed at.

USAGE
    Run from the repo root, after fetch_tmdb_catalog.py:

        python -m data_preprocessing.fetch_wikipedia_plots
        python -m data_preprocessing.fetch_wikipedia_plots --limit 20

OUTPUT
    data_preprocessing/data_ready/wikipedia_plots.csv -- gitignored and
    regenerable. No credentials and no LLM budget are needed.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import unquote

import requests

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# Wikipedia asks for a descriptive User-Agent identifying the caller.
USER_AGENT = "MoviBot/0.1 (course project; contact via github.com/alanarazi7/movibot)"

DEFAULT_CATALOG = Path("data_preprocessing") / "data_ready" / "tmdb_catalog.csv"
DEFAULT_OUT_DIR = Path("data_preprocessing") / "data_ready"

# Wikidata tolerates large VALUES blocks.
SPARQL_BATCH = 150

# Wikipedia refuses to batch WHOLE-article extracts: passing several titles
# makes it answer "exlimit was too large for a whole article extracts request,
# lowered to 1" and silently return text for one page only. Batching is
# available for intro-only extracts, which are useless here -- the plot
# section sits well below the intro. So: one article per request.
EXTRACT_BATCH = 1

# Headings used for the story on film articles, in preference order. The last
# four cover Disney's package films -- Fantasia, Fantasia 2000, Saludos Amigos,
# Melody Time, Make Mine Music -- which are anthologies of shorts and so have
# no single "Plot" section, listing their segments instead.
PLOT_SECTION_NAMES = (
    "plot",
    "plot summary",
    "synopsis",
    "story",
    "premise",
    "program",
    "film segments",
    "segments",
    "vignettes",
)

OUTPUT_COLUMNS = [
    "movie_id", "imdb_id", "title", "release_year",
    "wikipedia_title", "wikipedia_url", "plot_words", "plot_text",
]

# Whole-article extracts cannot be batched, so this runs one serial request per
# film. Wikipedia starts returning 429 at ~0.1s spacing; 0.5s is polite and
# still finishes the full catalog in well under ten minutes.
_REQUEST_PAUSE_SECONDS = 0.5
_MAX_ATTEMPTS = 6


class WikiError(RuntimeError):
    pass


# ---------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------

def chunked(items: Sequence[Any], size: int) -> Iterator[list]:
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    for start in range(0, len(items), size):
        yield list(items[start:start + size])


def article_title_from_url(url: str) -> str:
    return unquote(url.rsplit("/", 1)[-1]).replace("_", " ")


def parse_sparql_articles(payload: Any) -> dict[str, str]:
    """Maps IMDb id -> English Wikipedia article title, first binding wins."""
    if not isinstance(payload, dict):
        return {}

    bindings = (payload.get("results") or {}).get("bindings") or []
    out: dict[str, str] = {}
    for row in bindings:
        imdb = ((row.get("imdb") or {}).get("value") or "").strip()
        article = ((row.get("article") or {}).get("value") or "").strip()
        if imdb and article and imdb not in out:
            out[imdb] = article_title_from_url(article)
    return out


def extract_plot(extract_text: str | None) -> str | None:
    """Pulls the plot section out of a plain-text MediaWiki extract.

    Only `== Heading ==` terminates the section; `=== Subheading ===` is part
    of it, because long films split their plot into acts.
    """
    if not extract_text:
        return None

    for name in PLOT_SECTION_NAMES:
        # `\n== Name ==\n` ... up to the next level-2 heading or end of text.
        pattern = re.compile(
            r"\n==[ \t]*" + re.escape(name) + r"[ \t]*==[ \t]*\n(.*?)(?=\n==[ \t]*[^=]|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(extract_text)
        if match:
            body = match.group(1).strip()
            if body:
                return body
    return None


# ---------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------

def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return session


def _get(session: requests.Session, url: str, params: dict[str, Any], accept: str) -> Any:
    last = _MAX_ATTEMPTS - 1

    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = session.get(
                url, params=params, timeout=90, headers={"Accept": accept}
            )
        except requests.RequestException as exc:
            if attempt == last:
                raise WikiError(f"Network error calling {url}: {exc}") from exc
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 429 or response.status_code >= 500:
            if attempt == last:
                raise WikiError(f"{url} kept returning {response.status_code}")
            # Honour Retry-After when the server sends it, back off otherwise.
            try:
                wait = int(response.headers.get("Retry-After", ""))
            except ValueError:
                wait = 0
            time.sleep(max(wait, 2 ** attempt + 1))
            continue
        if not response.ok:
            raise WikiError(f"{url} returned {response.status_code}")

        time.sleep(_REQUEST_PAUSE_SECONDS)
        return response.json()

    raise WikiError(f"Gave up calling {url}")


def resolve_articles(session: requests.Session, imdb_ids: Sequence[str]) -> dict[str, str]:
    """IMDb id -> Wikipedia article title, via Wikidata. Exact, never fuzzy."""
    resolved: dict[str, str] = {}

    for batch in chunked(imdb_ids, SPARQL_BATCH):
        values = " ".join(f'"{i}"' for i in batch)
        query = (
            "SELECT ?imdb ?article WHERE { "
            f"VALUES ?imdb {{ {values} }} "
            "?film wdt:P345 ?imdb . "
            "?article schema:about ?film ; "
            "schema:isPartOf <https://en.wikipedia.org/> . }"
        )
        payload = _get(
            session,
            WIKIDATA_SPARQL_URL,
            {"query": query, "format": "json"},
            "application/sparql-results+json",
        )
        resolved.update(parse_sparql_articles(payload))

    return resolved


def fetch_extracts(session: requests.Session, titles: Sequence[str]) -> dict[str, str]:
    """Article title -> plain-text extract, batched."""
    extracts: dict[str, str] = {}

    for batch in chunked(titles, EXTRACT_BATCH):
        payload = _get(
            session,
            WIKIPEDIA_API_URL,
            {
                "action": "query",
                "prop": "extracts",
                "explaintext": "1",
                "redirects": "1",
                "format": "json",
                "formatversion": "2",
                "titles": "|".join(batch),
            },
            "application/json",
        )
        for page in (payload.get("query") or {}).get("pages") or []:
            title = page.get("title")
            text = page.get("extract")
            if title and text:
                extracts[title] = text

    return extracts


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N catalog rows (cheap test run).",
    )
    parser.add_argument(
        "--min-words", type=int, default=50,
        help="Reject plot sections shorter than this (default: 50).",
    )
    return parser.parse_args()


def load_catalog(path: Path, limit: int | None) -> list[dict[str, str]]:
    if not path.exists():
        raise WikiError(
            f"Catalog not found: {path}. Run fetch_tmdb_catalog.py first."
        )
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit is not None else rows


def main() -> int:
    args = parse_args()

    try:
        rows = load_catalog(args.catalog, args.limit)
        print(f"1/4 Loaded {len(rows):,} catalog rows from {args.catalog}")

        with_id = [r for r in rows if (r.get("imdb_id") or "").strip()]
        print(f"2/4 Resolving {len(with_id):,} IMDb ids through Wikidata...")

        session = build_session()
        articles = resolve_articles(session, [r["imdb_id"].strip() for r in with_id])
        print(f"    {len(articles):,} resolved to a Wikipedia article")

        print("3/4 Fetching article extracts...")
        extracts = fetch_extracts(session, sorted(set(articles.values())))
        print(f"    {len(extracts):,} extracts retrieved")

    except WikiError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("4/4 Extracting plot sections...")
    out_rows: list[dict[str, Any]] = []
    no_imdb = len(rows) - len(with_id)
    unresolved = 0
    no_plot = 0
    too_short = 0

    for row in with_id:
        imdb_id = row["imdb_id"].strip()
        article = articles.get(imdb_id)
        if not article:
            unresolved += 1
            continue

        plot = extract_plot(extracts.get(article))
        if plot is None:
            no_plot += 1
            continue

        words = len(plot.split())
        if words < args.min_words:
            too_short += 1
            continue

        out_rows.append({
            "movie_id": row["id"],
            "imdb_id": imdb_id,
            "title": row["title"],
            "release_year": row["release_year"],
            "wikipedia_title": article,
            "wikipedia_url": "https://en.wikipedia.org/wiki/"
                             + article.replace(" ", "_"),
            "plot_words": words,
            "plot_text": plot,
        })

    out_rows.sort(key=lambda r: int(r["movie_id"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "wikipedia_plots.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    total = len(rows)
    print("\n=== WIKIPEDIA PLOT FETCH COMPLETE ===")
    print(f"Catalog rows:      {total:,}")
    print(f"No IMDb id:        {no_imdb:,} (not guessed by title -- reported, not matched)")
    print(f"No Wikidata link:  {unresolved:,}")
    print(f"No plot section:   {no_plot:,}")
    print(f"Plot too short:    {too_short:,} (under {args.min_words} words)")
    print(f"Written:           {len(out_rows):,} -> {out_path}")
    if out_rows:
        words = sorted(r["plot_words"] for r in out_rows)
        print(f"Coverage:          {len(out_rows) / total:.1%} of the catalog")
        print(f"Plot length:       median {words[len(words) // 2]:,} words, "
              f"min {words[0]:,}, max {words[-1]:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
