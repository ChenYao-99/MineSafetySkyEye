"""Verify frozen local model artifacts and their inference protocols before loading."""
import hashlib
import json
from pathlib import Path


class ModelArtifactError(RuntimeError):
    pass


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ModelRegistry:
    def __init__(self, manifest_path, models_root):
        self.path = Path(manifest_path)
        self.root = Path(models_root).resolve()
        self._verified = {}

    def verify(self, key, configured_path, protocol):
        try:
            content = self.path.read_bytes()
            manifest = json.loads(content)
            entry = manifest['artifacts'][key]
            path = (self.root / entry['path']).resolve()
            if manifest['schema_version'] != 1 or not path.is_relative_to(self.root):
                raise ValueError('invalid registry schema/path')
            if path != Path(configured_path).resolve() or entry['protocol'] != protocol:
                raise ValueError('configured model or input protocol differs from frozen registry')
            stat = path.stat()
            manifest_sha = hashlib.sha256(content).hexdigest()
            signature = (manifest_sha, str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if self._verified.get(key) != signature:
                if file_sha256(path) != entry['sha256']:
                    raise ValueError('weights hash mismatch')
                self._verified[key] = signature
            return {'registry_schema_version': 1, 'registry_sha256': manifest_sha,
                    'weights_sha256': entry['sha256'], 'protocol': dict(protocol),
                    'validation_status': entry['validation_status']}
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ModelArtifactError(f'模型注册校验失败（{key}），未加载或替代该模型') from exc
