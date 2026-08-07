from edgar_parse import chunk_filing, split_into_sections


_SAMPLE_10K_HTML = """
<html><body>
<p>TABLE OF CONTENTS</p>
<p>Item 1. Business</p>
<p>Item 1A. Risk Factors</p>
<p>Item 7. Management's Discussion and Analysis</p>

<p>Item 1. Business</p>
<p>We design, manufacture, and sell widgets and related services worldwide.</p>

<p>Item 1A. Risk Factors</p>
<p>Our results of operations could be materially harmed by supply chain disruption.</p>

<p>Item 7. Management's Discussion and Analysis</p>
<p>Net sales increased year over year, driven by higher unit volumes.</p>
<table>
<tr><td>Net sales</td><td>$1,234</td><td>$1,100</td></tr>
<tr><td>Gross margin</td><td>45.2%</td><td>43.9%</td></tr>
</table>
<p>The increase was primarily attributable to services revenue.</p>
</body></html>
"""


def test_splits_on_item_boundaries_not_token_count():
    sections = split_into_sections(_SAMPLE_10K_HTML)
    items = [s.item for s in sections]
    assert "Item 1" in items
    assert "Item 1A" in items
    assert "Item 7" in items
    # each Item's body only contains that Item's own narrative, not the next one's
    item_1 = next(s for s in sections if s.item == "Item 1")
    assert "widgets" in item_1.text
    assert "supply chain" not in item_1.text


def test_table_of_contents_duplicate_headings_collapsed_to_real_section():
    sections = split_into_sections(_SAMPLE_10K_HTML)
    # exactly one "Item 1A" section should exist (TOC duplicate must not split it)
    assert sum(1 for s in sections if s.item == "Item 1A") == 1


def test_numeric_claim_inside_table_survives_verbatim_into_a_chunk():
    chunks = chunk_filing(_SAMPLE_10K_HTML)
    item_7_chunks = [c for c in chunks if c.item == "Item 7"]
    assert item_7_chunks
    combined = "\n".join(c.text for c in item_7_chunks)
    assert "$1,234" in combined
    assert "45.2%" in combined


def test_no_item_headings_falls_back_to_full_document():
    html = "<html><body><p>No item markers here, just prose about the company.</p></body></html>"
    sections = split_into_sections(html)
    assert len(sections) == 1
    assert sections[0].item == "FULL_DOCUMENT"


def test_chunks_never_cross_an_item_boundary():
    chunks = chunk_filing(_SAMPLE_10K_HTML, chunk_words=6, overlap=1)
    for c in chunks:
        # with a tiny chunk_words window, Item 1's chunks must never contain
        # text belonging to Item 1A or Item 7's narrative
        if c.item == "Item 1":
            assert "supply chain" not in c.text
            assert "Net sales" not in c.text
