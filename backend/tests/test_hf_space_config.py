"""
Tests for the Hugging Face Space configuration: the YAML front matter in
README.md that the Spaces builder reads (missing/invalid front matter puts
the Space in CONFIG_ERROR), plus the files it points at.
"""
import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
README = os.path.join(REPO_ROOT, 'README.md')

# Colors the Spaces config reference accepts for colorFrom/colorTo.
VALID_COLORS = {'red', 'yellow', 'green', 'blue', 'indigo', 'purple', 'pink', 'gray'}
REQUIRED_KEYS = {'title', 'emoji', 'colorFrom', 'colorTo', 'sdk', 'app_file'}


def parse_front_matter(text):
    """Minimal `key: value` front-matter parser (avoids a PyYAML dependency)."""
    if not text.startswith('---\n'):
        return None
    end = text.find('\n---', 3)
    if end == -1:
        return None
    block = text[4:end]
    out = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or ':' not in line:
            continue
        key, _, value = line.partition(':')
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


@pytest.fixture(scope='module')
def front_matter():
    with open(README, encoding='utf-8') as fh:
        fm = parse_front_matter(fh.read())
    assert fm is not None, 'README.md is missing YAML front matter'
    return fm


class TestSpaceConfig:
    def test_readme_starts_with_front_matter(self):
        with open(README, encoding='utf-8') as fh:
            assert fh.read().startswith('---\n')

    def test_required_keys_present(self, front_matter):
        missing = REQUIRED_KEYS - set(front_matter)
        assert not missing, f'missing Space config keys: {sorted(missing)}'

    def test_sdk_is_gradio(self, front_matter):
        assert front_matter['sdk'] == 'gradio'

    def test_colors_are_valid(self, front_matter):
        assert front_matter['colorFrom'] in VALID_COLORS
        assert front_matter['colorTo'] in VALID_COLORS

    def test_app_file_exists(self, front_matter):
        assert os.path.exists(os.path.join(REPO_ROOT, front_matter['app_file']))

    def test_short_description_within_limit(self, front_matter):
        # Spaces truncates the card subtitle past 60 characters.
        assert len(front_matter.get('short_description', '')) <= 60

    def test_sdk_version_matches_requirements(self, front_matter):
        with open(os.path.join(REPO_ROOT, 'requirements.txt'), encoding='utf-8') as fh:
            pinned = [l.strip() for l in fh if l.strip().startswith('gradio==')]
        assert pinned, 'requirements.txt does not pin gradio'
        assert front_matter['sdk_version'] == pinned[0].split('==', 1)[1]
