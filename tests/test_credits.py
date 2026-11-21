"""The credits screen's text: the README's data attributions, verbatim, and every layer's licence."""

from pathlib import Path

from contravia.credits import SECTIONS

README = (Path(__file__).resolve().parent.parent / "README.md").read_text()


def test_the_credits_quote_each_data_attribution_of_the_readme():
    data = README.split("## Data", 1)[1].split("\n## ", 1)[0]
    attributions = [line.split(": ", 1)[1] for line in data.splitlines() if line.startswith("- ")]
    lines = [line for _, ls in SECTIONS for line in ls]
    assert len(attributions) == 5
    for a in attributions:
        assert a in lines, a


def test_the_credits_name_every_layer_licence():
    text = "\n".join(line for _, ls in SECTIONS for line in ls)
    for name in ("GPLv3", "CC BY-NC-SA 4.0", "ODbL 1.0", "SIL Open Font License 1.1", "MIT", "Licencia de Mods"):
        assert name in text, name
