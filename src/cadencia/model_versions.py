"""Hash-verified model releases, available from a checkout or installed wheel."""
import hashlib
import json
from pathlib import Path
import re

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT = _SOURCE_ROOT if (_SOURCE_ROOT / 'pyproject.toml').is_file() else Path(__file__).resolve().parents[1]
DEFAULT_MODEL_VERSION = 'v1'
SCHEMA = 'cadencia.model_release.v1'


def is_model_version(value):
    return isinstance(value, str) and re.fullmatch(r'v\d+(?:\.\d+\.\d+)?', value) is not None


def resolve_model_version(version=DEFAULT_MODEL_VERSION):
    if not is_model_version(version):
        raise ValueError('Model version must use vMAJOR or vMAJOR.MINOR.PATCH')
    registry = ROOT / 'model_releases' / f'{version}.json'
    if not registry.is_file():
        raise FileNotFoundError(f'Model release is not installed: {version}')
    manifest = json.loads(registry.read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('version') != version:
        raise ValueError('Invalid model release manifest')
    if manifest.get('status', 'released') != 'released':
        raise ValueError(f'Model release {version} is not trained/published yet')
    for name, artifact in manifest['artifacts'].items():
        path = (ROOT / artifact['path']).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError('Release artifact escapes repository')
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
            raise ValueError(f'Model release checksum mismatch: {name}')
    return ROOT / manifest['artifacts']['decoder']['path']
