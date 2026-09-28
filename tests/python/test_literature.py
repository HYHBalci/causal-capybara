import io
import json
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'sidecar'), str(ROOT / 'engines/python')]
from capy_sidecar import literature, reports


def test_sources_are_unique_and_have_safe_links():
    rows = literature.records()
    assert len(rows) >= 15
    assert len({row['id'] for row in rows}) == len(rows)
    for row in rows:
        assert literature.safe_url(row['url'])
        assert len(row['citation']) > 60
        assert literature.resolve(row['citation'])['resolved']


def test_resolution_preserves_locator_and_handles_unknowns():
    source = literature.resolve('Hernán & Robins (2020), ch. 13')
    assert source['url'] == 'https://miguelhernan.org/whatifbook'
    assert 'ch. 13' in source['citation']
    assert not literature.resolve('An unknown citation')['resolved']
    assert not literature.safe_url('javascript:alert(1)')


def test_report_reference_links_survive_text_html_latex_and_word():
    source = literature.resolve("Callaway & Sant'Anna (2021)")
    blocks = [{'type': 'reference', 'text': source['citation'], 'url': source['url']}]
    assert '](' + source['url'] + ')' in reports._render_markdown(blocks)
    assert 'href="' + source['url'] + '"' in reports._render_html(blocks, title='References')
    assert '\\url{' + source['url'] + '}' in reports._render_latex(blocks, title='References')
    import docx
    from docx.shared import Pt
    blob = reports._docx_bytes(blocks, title='References', ctx=None, docx=docx, Pt=Pt)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        links = ElementTree.fromstring(archive.read('word/_rels/document.xml.rels'))
        assert any(link.get('Target') == source['url'] and link.get('TargetMode') == 'External' for link in links)
        assert b'w:hyperlink' in archive.read('word/document.xml')
