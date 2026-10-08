
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser
import hashlib
import json
import re
import time

import requests
from bs4 import BeautifulSoup


BASE_URLS = [
    "https://www.spit.ac.in/",
    "https://www.spit.ac.in/admissions/",
    "https://www.spit.ac.in/academics/",
    "https://www.spit.ac.in/curriculum-2/",
    "https://www.spit.ac.in/academic-calendar/",
]

KB_ROOT = Path("knowledge_base")
CRAWLED_PAGES = KB_ROOT / "website_pages"
PDF_DIR = KB_ROOT / "institutional_pdfs" / "downloaded"

MAX_HTML_PAGES = 200
MAX_PDFS = 250
MAX_PDF_BYTES = 100 * 1024 * 1024
REQUEST_DELAY = 1.0

USER_AGENT = "SPIT-RAG-StudentResearchBot/1.0"
HEADERS = {"User-Agent": USER_AGENT}

ALLOWED_HOSTS = {"spit.ac.in", "www.spit.ac.in"}
SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".zip", ".mp4", ".mp3", ".css", ".js",
)

session = requests.Session()
session.headers.update(HEADERS)
robots_cache = {}


def normalize_url(url):
    """Keep URLs on the official SPIT domain only."""
    if not url:
        return None

    url, _ = urldefrag(url)
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return None

    if parsed.hostname not in ALLOWED_HOSTS:
        return None

    if parsed.username or parsed.password:
        return None

    if parsed.path.lower().endswith(SKIP_EXTENSIONS):
        return None

    # Use HTTPS and remove fragments; retain query strings for PDF links.
    return parsed._replace(scheme="https", fragment="").geturl()


def is_pdf_url(url):
    return urlparse(url).path.lower().endswith(".pdf")


def can_fetch(url):
    host = urlparse(url).hostname

    if host not in robots_cache:
        robots_url = f"https://{host}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)

        try:
            response = session.get(robots_url, timeout=20)

            if response.status_code == 404:
                parser.parse([])
            else:
                response.raise_for_status()
                parser.parse(response.text.splitlines())

            robots_cache[host] = parser

        except requests.RequestException as exc:
            # Fail closed if the crawling rules cannot be checked.
            print(f"Cannot verify robots.txt for {host}: {exc}")
            robots_cache[host] = None

    parser = robots_cache[host]
    return parser is not None and parser.can_fetch(USER_AGENT, url)


def initial_urls():
    """Reuse source URLs from the pages already crawled."""
    urls = set(BASE_URLS)
    pattern = re.compile(r"^Source URL:\s*(https?://\S+)", re.MULTILINE)

    if CRAWLED_PAGES.exists():
        for file in CRAWLED_PAGES.rglob("*.md"):
            try:
                text = file.read_text(encoding="utf-8")
            except OSError:
                continue

            match = pattern.search(text)
            if match:
                normalized = normalize_url(match.group(1))
                if normalized and not is_pdf_url(normalized):
                    urls.add(normalized)

    return sorted(urls)


def pdf_filename(url):
    """Use a readable filename and avoid collisions."""
    name = Path(urlparse(url).path).name or "document.pdf"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)

    if not name.lower().endswith(".pdf"):
        name += ".pdf"

    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    return f"{Path(name).stem}_{digest}.pdf"


def download_pdf(url):
    if not can_fetch(url):
        print(f"ROBOTS DISALLOW: {url}")
        return None

    try:
        with session.get(
            url,
            timeout=45,
            stream=True,
            allow_redirects=True,
        ) as response:
            response.raise_for_status()

            final_url = normalize_url(response.url)
            if not final_url:
                print(f"SKIP external redirect: {response.url}")
                return None

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if (
                "pdf" not in content_type
                and not is_pdf_url(final_url)
            ):
                print(f"SKIP non-PDF response: {final_url}")
                return None

            declared_size = int(
                response.headers.get("Content-Length", "0") or 0
            )
            if declared_size > MAX_PDF_BYTES:
                print(f"SKIP oversized PDF: {final_url}")
                return None

            data = bytearray()
            for block in response.iter_content(chunk_size=64 * 1024):
                if not block:
                    continue
                data.extend(block)

                if len(data) > MAX_PDF_BYTES:
                    print(f"SKIP oversized download: {final_url}")
                    return None

            if not data.startswith(b"%PDF-"):
                print(f"SKIP invalid PDF content: {final_url}")
                return None

            destination = PDF_DIR / pdf_filename(final_url)
            destination.write_bytes(data)

            print(f"DOWNLOADED: {destination.name}")
            return {
                "url": final_url,
                "file": str(destination),
                "bytes": len(data),
                "content_type": content_type,
            }

    except requests.RequestException as exc:
        print(f"PDF DOWNLOAD FAILED: {url}: {exc}")
        return None


def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    queue = deque(initial_urls())
    visited = set()
    pdf_urls = set()
    downloaded = []
    failures = []

    print(f"Starting from {len(queue)} SPIT page URLs.")

    while queue and len(visited) < MAX_HTML_PAGES:
        url = normalize_url(queue.popleft())

        if not url or url in visited or is_pdf_url(url):
            continue

        visited.add(url)

        if not can_fetch(url):
            print(f"ROBOTS DISALLOW: {url}")
            continue

        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()

            final_url = normalize_url(response.url)
            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if not final_url or "text/html" not in content_type:
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            for anchor in soup.find_all("a", href=True):
                link = normalize_url(urljoin(final_url, anchor["href"]))
                if not link:
                    continue

                if is_pdf_url(link):
                    pdf_urls.add(link)
                elif link not in visited:
                    queue.append(link)

            print(
                f"PAGE {len(visited)}/{MAX_HTML_PAGES}: {final_url}"
            )

        except requests.RequestException as exc:
            failures.append({"url": url, "error": str(exc)})
            print(f"PAGE FAILED: {url}: {exc}")

        time.sleep(REQUEST_DELAY)

    print(f"\nDiscovered {len(pdf_urls)} unique PDF URLs.")

    for number, url in enumerate(sorted(pdf_urls), start=1):
        if len(downloaded) >= MAX_PDFS:
            print("Reached the configured PDF limit.")
            break

        result = download_pdf(url)
        if result:
            downloaded.append(result)

        if number < len(pdf_urls):
            time.sleep(REQUEST_DELAY)

    report = {
        "domain": "spit.ac.in",
        "html_pages_visited": len(visited),
        "pdf_urls_discovered": len(pdf_urls),
        "pdfs_downloaded": len(downloaded),
        "download_directory": str(PDF_DIR),
        "downloads": downloaded,
        "page_failures": failures,
    }

    report_path = PDF_DIR / "_download_manifest.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nPDF download complete.")
    print(f"Pages visited: {len(visited)}")
    print(f"PDFs downloaded: {len(downloaded)}")
    print(f"Manifest: {report_path.resolve()}")


if __name__ == "__main__":
    main()