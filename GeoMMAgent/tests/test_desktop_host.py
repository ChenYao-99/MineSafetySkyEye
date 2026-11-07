"""Desktop integration must preserve the local web launcher and service identity."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('kat_desktop_host', ROOT / 'desktop/app.py')
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


class DesktopHostTests(unittest.TestCase):
    def test_workbench_is_injected_only_into_the_exact_local_desktop_page(self):
        window = Mock()
        window.get_current_url.return_value = 'https://example.com/'
        self.assertIsNone(host.install_workbench(window, 'http://127.0.0.1:5100/'))
        window.load_css.assert_not_called()
        window.evaluate_js.assert_not_called()
        window.get_current_url.return_value = 'http://127.0.0.1:5100/'
        window.evaluate_js.return_value = {'ready': True, 'version': 'test'}
        self.assertTrue(host.install_workbench(window, 'http://127.0.0.1:5100/')['ready'])
        window.evaluate_js.return_value = {'ready': False}
        with self.assertRaises(RuntimeError):
            host.install_workbench(window, 'http://127.0.0.1:5100/')
        window.evaluate_js.return_value = {'ready': False, 'pending': True}
        self.assertIsNone(host.install_workbench(window, 'http://127.0.0.1:5100/'))

    def test_original_launcher_is_reused_without_opening_a_browser(self):
        command = host.startup_command(host.options([]))
        self.assertIn(str(ROOT / 'start_demo.ps1'), command)
        self.assertIn('-NoBrowser', command)
        self.assertNotIn('-LocalFactsOnly', command)
        self.assertNotIn('stop_demo.ps1', ' '.join(command))

    def test_explicit_offline_mode_and_ports_are_forwarded(self):
        args = host.options(['--local-facts-only', '--backend-port', '5120', '--detector-port', '5121'])
        command = host.startup_command(args)
        self.assertIn('-LocalFactsOnly', command)
        self.assertIn('5120', command)
        self.assertIn('5121', command)

    def test_runtime_mismatch_never_reports_ready(self):
        args = host.options([])
        with self.assertRaises(RuntimeError):
            host.verify_health({'service': 'other-app'}, args)
        value = {'service': 'kat-backend', 'runtime_root': str(ROOT),
                 'backend_runtime_sha256': 'stale', 'detector_online': True,
                 'detector_runtime_compatible': True}
        with patch('runtime_identity.runtime_fingerprint', return_value='current'):
            with self.assertRaises(RuntimeError):
                host.verify_health(value, args)
            value['backend_runtime_sha256'] = 'current'
            host.verify_health(value, args)
            with self.assertRaises(RuntimeError):
                host.verify_health(value, host.options(['--local-facts-only']))

    def test_background_service_launch_does_not_wait_for_inherited_pipe_EOF(self):
        with patch.object(Path, 'open') as opened, \
             patch.object(host.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            host.run_launcher(host.options([]))
        self.assertIs(run.call_args.kwargs['stdout'], opened.return_value.__enter__.return_value)
        self.assertNotEqual(run.call_args.kwargs['stdout'], host.subprocess.PIPE)
        with patch.object(Path, 'open'), \
             patch.object(host.subprocess, 'run', return_value=SimpleNamespace(returncode=1)):
            with self.assertRaises(RuntimeError):
                host.run_launcher(host.options([]))

    def test_splash_escapes_error_text_and_explains_shared_service_lifetime(self):
        page = host.splash('<script>bad()</script>', failed=True)
        self.assertNotIn('<script>bad()', page)
        self.assertIn('&lt;script&gt;', page)
        self.assertIn('网页版仍可使用', page)

    def test_desktop_paths_are_separate_from_web_frontend_and_on_D(self):
        self.assertEqual(host.CACHE.drive.upper(), 'D:')
        self.assertEqual(host.EXPORTS.drive.upper(), 'D:')

    def test_downloads_reserve_unique_names_without_overwriting(self):
        host.CACHE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=host.CACHE) as temporary:
            folder = Path(temporary)
            original = folder / 'report.json'
            original.write_bytes(b'keep-existing-report')
            first = host.reserve_download(folder, 'report.json')
            second = host.reserve_download(folder, 'report.json')
            self.assertEqual(original.read_bytes(), b'keep-existing-report')
            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, folder)
            self.assertEqual(second.parent, folder)


if __name__ == '__main__':
    unittest.main(verbosity=2)
