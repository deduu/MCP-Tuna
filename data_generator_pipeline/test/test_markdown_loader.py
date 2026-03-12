from pathlib import Path

from data_generator_pipeline.loaders import MarkdownPageLoader


def test_markdown_loader_splits_h2_sections(tmp_path: Path):
    doc = tmp_path / "sample.md"
    doc.write_text(
        "---\n"
        "title: Sample\n"
        "---\n\n"
        "# Sample Title\n\n"
        "Intro paragraph.\n\n"
        "## First Section\n\n"
        "Alpha content.\n\n"
        "## Second Section\n\n"
        "Beta content.\n",
        encoding="utf-8",
    )

    _, pages = MarkdownPageLoader().load(str(doc))

    assert len(pages) == 2
    assert all(page["markdown"].startswith("---") for page in pages)
    assert "## First Section" in pages[0]["markdown"]
    assert "## Second Section" in pages[1]["markdown"]


def test_ksmi_markdown_docs_now_split_into_multiple_chunks():
    loader = MarkdownPageLoader()

    _, commercial_pages = loader.load("data/KSMI/CommercialReporting.md")
    _, technical_pages = loader.load("data/KSMI/TechnicalCodification.md")

    assert len(commercial_pages) > 1
    assert len(technical_pages) > 1
