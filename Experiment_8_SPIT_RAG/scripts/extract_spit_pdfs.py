
from pathlib import Path
import json
import traceback

from pypdf import PdfReader


PDF_DIR = Path("knowledge_base/institutional_pdfs/downloaded")
OUTPUT_DIR = Path("results/pdf_extracted")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_DIR.rglob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found under {PDF_DIR.resolve()}")
        print("Run: uv run python scripts/download_spit_pdfs.py")
        return

    report = {
        "pdf_count": len(pdf_files),
        "successful": 0,
        "failed": 0,
        "empty_text": 0,
        "documents": [],
    }

    for number, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{number}/{len(pdf_files)}] {pdf_path.name}")

        item = {
            "pdf": str(pdf_path),
            "pages": 0,
            "characters_extracted": 0,
            "status": "pending",
        }

        try:
            reader = PdfReader(str(pdf_path), strict=False)

            if reader.is_encrypted:
                try:
                    result = reader.decrypt("")
                    if result == 0:
                        raise ValueError("PDF requires a password")
                except Exception as exc:
                    raise ValueError(
                        f"Cannot decrypt PDF: {exc}"
                    ) from exc

            metadata = reader.metadata
            title = (
                str(metadata.title).strip()
                if metadata and metadata.title
                else pdf_path.stem
            )
            author = (
                str(metadata.author).strip()
                if metadata and metadata.author
                else "Unknown"
            )

            output_parts = [
                f"# {title}",
                f"Source PDF: {pdf_path.as_posix()}",
                f"Author: {author}",
                f"Page count: {len(reader.pages)}",
                "",
            ]

            extracted_characters = 0

            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as exc:
                    text = ""
                    print(
                        f"  Warning: page {page_number} "
                        f"could not be extracted: {exc}"
                    )

                text = text.strip()
                extracted_characters += len(text)

                output_parts.extend([
                    f"## Page {page_number}",
                    text if text else (
                        "[No selectable text extracted from this page.]"
                    ),
                    "",
                ])

            output_path = OUTPUT_DIR / f"{pdf_path.stem}.md"
            output_path.write_text(
                "\n".join(output_parts),
                encoding="utf-8",
            )

            item.update({
                "title": title,
                "author": author,
                "pages": len(reader.pages),
                "characters_extracted": extracted_characters,
                "output": str(output_path),
                "status": (
                    "success" if extracted_characters > 0
                    else "empty_text_requires_ocr_check"
                ),
            })

            if extracted_characters > 0:
                report["successful"] += 1
                print(
                    f"  Extracted {extracted_characters:,} characters "
                    f"from {len(reader.pages)} pages."
                )
            else:
                report["empty_text"] += 1
                print(
                    "  No text extracted. This PDF may be scanned "
                    "and require OCR."
                )

        except Exception as exc:
            item.update({
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            report["failed"] += 1
            print(f"  FAILED: {exc}")

        report["documents"].append(item)

    report_path = OUTPUT_DIR / "_extraction_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nExtraction complete.")
    print(f"PDFs found: {report['pdf_count']}")
    print(f"Successfully extracted: {report['successful']}")
    print(f"Empty text / OCR candidates: {report['empty_text']}")
    print(f"Failed: {report['failed']}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Report: {report_path.resolve()}")


if __name__ == "__main__":
    main()