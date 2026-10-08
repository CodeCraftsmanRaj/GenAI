
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser
import re
import time

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.spit.ac.in/"
ALLOWED_HOST = "www.spit.ac.in"
OUTPUT_DIR = Path("knowledge_base/website_pages")
MAX_PAGES = 100
DELAY_SECONDS = 1.0

SEED_URLS = [
    "https://www.spit.ac.in/",
    "https://www.spit.ac.in/about/",
    "https://www.spit.ac.in/admissions/",
    "https://www.spit.ac.in/curriculum-2/",
    "https://www.spit.ac.in/academic-calendar/",
    "https://www.spit.ac.in/news-events/",
    "https://www.spit.ac.in/category/notice/",
    "https://www.spit.ac.in/deans/",
]

HEADERS = {
    "User-Agent": (
        "SPIT-RAG-StudentResearchBot/1.0 "
        "(educational website indexing)"
    )
}

SKIP_PATH_PARTS = (
    "/wp-admin/",
    "/wp-json/",
    "/feed/",
    "/author/",
    "/tag/",
    "/search/",
)

SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".zip", ".mp4", ".mp3", ".css", ".js", ".xml",
)


def normalize_url(url):
    url, _ = urldefrag(url)
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return None

    if parsed.hostname != ALLOWED_HOST:
        return None

    if parsed.username or parsed.password:
        return None

    path = parsed.path or "/"

    if any(part in path.lower() for part in SKIP_PATH_PARTS):
        return None

    if path.lower().endswith(SKIP_EXTENSIONS):
        return None

    # Remove query strings to avoid duplicate tracking URLs.
    return parsed._replace(
        scheme="https",
        netloc=ALLOWED_HOST,
        query="",
        fragment="",
    ).geturl()


def load_robots():
    robots_url = urljoin(BASE_URL, "robots.txt")
    parser = RobotFileParser()
    parser.set_url(robots_url)

    try:
        response = requests.get(
            robots_url,
            headers=HEADERS,
            timeout=20,
        )
        response.raise_for_status()
        parser.parse(response.text.splitlines())
        print(f"Loaded crawling rules from {robots_url}")
        return parser
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not read robots.txt. Check site access before crawling."
        ) from exc


def extract_page(html, url):
    soup = BeautifulSoup(html, "html.parser")

    for element in soup.select(
        "script, style, noscript, nav, footer, header, "
        "form, iframe, svg"
    ):
        element.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else url

    content = (
        soup.select_one("main")
        or soup.select_one("article")
        or soup.select_one(".entry-content")
        or soup.select_one("#content")
        or soup.body
    )

    if content is None:
        return None, []

    # Preserve headings to make the extracted text easier to retrieve.
    lines = []
    for element in content.find_all(
        ["h1", "h2", "h3", "h4", "p", "li", "td", "th"]
    ):
        text = element.get_text(" ", strip=True)
        if not text:
            continue

        if element.name == "h1":
            lines.append(f"# {text}")
        elif element.name == "h2":
            lines.append(f"## {text}")
        elif element.name in ("h3", "h4"):
            lines.append(f"### {text}")
        elif element.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    clean_text = "\n\n".join(lines)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()

    links = []
    for anchor in soup.find_all("a", href=True):
        absolute = normalize_url(urljoin(url, anchor["href"]))
        if absolute:
            links.append(absolute)

    markdown = (
        f"# {title}\n\n"
        f"Source URL: {url}\n\n"
        f"Retrieved for the SPIT RAG knowledge base.\n\n"
        f"{clean_text}\n"
    )

    return markdown, links


def safe_filename(url):
    path = urlparse(url).path.strip("/")
    if not path:
        return "home.md"

    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", path)
    return f"{slug[:150]}.md"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    robots = load_robots()
    session = requests.Session()
    session.headers.update(HEADERS)

    queue = deque(SEED_URLS)
    visited = set()
    saved = 0
    failures = []

    while queue and saved < MAX_PAGES:
        url = normalize_url(queue.popleft())

        if not url or url in visited:
            continue

        visited.add(url)

        if not robots.can_fetch(HEADERS["User-Agent"], url):
            print(f"ROBOTS DISALLOW: {url}")
            continue

        try:
            response = session.get(
                url,
                timeout=30,
                allow_redirects=True,
            )
            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if "text/html" not in content_type:
                continue

            final_url = normalize_url(response.url)
            if not final_url:
                continue

            markdown, links = extract_page(
                response.text, final_url
            )

            if not markdown or len(markdown) < 150:
                print(f"SKIP (little readable text): {final_url}")
                continue

            destination = OUTPUT_DIR / safe_filename(final_url)

            # Avoid silently overwriting two pages with the same slug.
            if destination.exists():
                destination = OUTPUT_DIR / (
                    destination.stem
                    + "_"
                    + str(abs(hash(final_url)) % 1000000)
                    + ".md"
                )

            destination.write_text(
                markdown,
                encoding="utf-8",
            )

            saved += 1
            print(f"[{saved}/{MAX_PAGES}] {final_url}")

            for link in links:
                if link not in visited:
                    queue.append(link)

        except requests.RequestException as exc:
            failures.append({"url": url, "error": str(exc)})
            print(f"FAILED: {url}: {exc}")

        time.sleep(DELAY_SECONDS)

    report = {
        "base_url": BASE_URL,
        "pages_saved": saved,
        "urls_attempted": len(visited),
        "failures": failures,
        "output_directory": str(OUTPUT_DIR),
    }

    (OUTPUT_DIR / "_crawl_report.json").write_text(
        __import__("json").dumps(
            report, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print("\nCrawl complete.")
    print(f"Pages saved: {saved}")
    print(f"Failures: {len(failures)}")
    print(f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()