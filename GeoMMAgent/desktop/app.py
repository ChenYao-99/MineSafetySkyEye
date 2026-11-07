"""Independent Windows desktop host for the existing, unchanged local web UI."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import html
import json
import logging
import os
import re
from pathlib import Path
import subprocess
import sys
import threading
from urllib.request import ProxyHandler, build_opener
from urllib.parse import urlsplit
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_runtime import configure_runtime_paths

TITLE = '矿安天眼 · 影像分析工作台'
CACHE = ROOT / 'cache' / 'desktop'
EXPORTS = ROOT / 'data' / 'desktop_exports'
SCENES = (
    ('影像检测', 'scene-mine'), ('变化分析', 'scene-compare'),
    ('智能体问答', 'scene-qa'), ('案例归档', 'scene-archive'),
    ('安全知识库', 'scene-kb'), ('模型评测', 'scene-bench'),
)


def port_number(value):
    number = int(value)
    if not 1024 <= number <= 65535:
        raise argparse.ArgumentTypeError('端口须在1024到65535之间')
    return number


def options(argv=None):
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument('--backend-port', type=port_number, default=5100)
    parser.add_argument('--detector-port', type=port_number, default=5101)
    parser.add_argument('--agent-python', default='D:/Miniconda3/envs/GeoMMA/python.exe')
    parser.add_argument('--detector-python', default='D:/Miniconda3/envs/lmhld/python.exe')
    parser.add_argument('--detector-dir', default='D:/LMHLD')
    parser.add_argument('--local-facts-only', action='store_true')
    parser.add_argument('--debug-port', type=port_number)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    if args.backend_port == args.detector_port or args.debug_port in (args.backend_port, args.detector_port):
        parser.error('界面、检测和可选调试端口必须不同')
    return args


def startup_command(args):
    command = [
        'powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', str(ROOT / 'start_demo.ps1'), '-NoBrowser', '-TimeoutSeconds', '150',
        '-AgentPython', str(Path(args.agent_python)),
        '-DetectorPython', str(Path(args.detector_python)),
        '-DetectorDirectory', str(Path(args.detector_dir)),
        '-BackendPort', str(args.backend_port), '-DetectorPort', str(args.detector_port),
    ]
    if args.local_facts_only:
        command.append('-LocalFactsOnly')
    return command


def read_health(url):
    # Loopback requests must not go through a user's HTTP proxy.
    with build_opener(ProxyHandler({})).open(url + 'api/health', timeout=6) as response:
        return json.load(response)


def run_launcher(args):
    # Background Windows service children may inherit a pipe handle. Reading PIPE
    # to EOF would then wait for the long-running services, even after PS exited.
    output = CACHE / f'launcher_{args.backend_port}.log'
    with output.open('a', encoding='utf-8') as stream:
        result = subprocess.run(startup_command(args), cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            timeout=210, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError(f'本地服务启动检查未通过，请查看{output}和cache/launcher。')


def verify_health(value, args):
    from runtime_identity import runtime_fingerprint
    if (value.get('service') != 'kat-backend'
            or Path(value.get('runtime_root', '')).resolve() != ROOT
            or value.get('backend_runtime_sha256') != runtime_fingerprint(ROOT, 'backend')
            or not value.get('detector_online')
            or value.get('detector_runtime_compatible') is not True):
        raise RuntimeError('本地服务来源或版本不匹配，请使用原有停止工具结束旧服务后重试')
    if args.local_facts_only and value.get('reporting_mode') != 'local_facts_only':
        raise RuntimeError('当前服务不是请求的本地事实模式')


def splash(message, failed=False):
    title = '启动未完成' if failed else '正在准备工作区'
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
    <style>body{{margin:0;background:#0b1628;color:#dbe9f6;font:16px "Microsoft YaHei",sans-serif;
    display:grid;min-height:100vh;place-items:center}}main{{max-width:650px;padding:52px}}
    .brand{{color:#59d8cf;letter-spacing:5px;font-size:14px}}h1{{font-size:38px;font-weight:600}}
    p{{line-height:1.9;white-space:pre-wrap}}small{{color:#98abc3}}</style>
    <main><div class="brand">矿安天眼 / DESKTOP</div><h1>{title}</h1>
    <p>{html.escape(message)}</p><small>桌面与网页版共享本地服务。关闭桌面窗口后，网页版仍可使用。</small></main></html>'''


def show_message(title, message):
    ctypes.windll.user32.MessageBoxW(None, str(message), str(title), 0x40)


def install_workbench(window, url):
    """Compose only the trusted desktop document; the served HTML stays intact."""
    if window.get_current_url() != url:
        return None
    css = (ROOT / 'desktop/workbench.css').read_text(encoding='utf-8')
    source = (ROOT / 'desktop/workbench.js').read_text(encoding='utf-8').rstrip().removesuffix(';')
    # Apply CSS and DOM composition in one renderer call. Separate asynchronous
    # calls could briefly expose an unstyled layout after a page reload.
    script = ("(()=>{if(location.href!==" + json.dumps(url) + ")return {ready:false,pending:true};"
              "let s=document.getElementById('katDesktopWorkbenchStyle');"
              "if(!s){s=document.createElement('style');s.id='katDesktopWorkbenchStyle';document.head.appendChild(s)}"
              "s.textContent=" + json.dumps(css, ensure_ascii=False) + ";const result=" + source +
              ";if(result?.ready && getComputedStyle(document.getElementById('katWorkbench')).display!=='grid')"
              "throw new Error('工作台样式未完整载入');return result;})()")
    result = window.evaluate_js(script)
    if isinstance(result, dict) and result.get('pending'):
        return None
    if not isinstance(result, dict) or result.get('ready') is not True:
        raise RuntimeError('工作台布局未能完整载入')
    return result


def reserve_download(directory, filename):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', Path(filename).name).strip(' .') or 'report'
    source = Path(name)
    for index in range(10000):
        suffix = '' if index == 0 else f' ({index})'
        target = directory / f'{source.stem[:160]}{suffix}{source.suffix[:16]}'
        try:
            descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        os.close(descriptor)
        return target
    raise OSError('导出目录中存在过多同名文件，请整理后重试')


def acquire_instance(port):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    identity = hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]
    handle = kernel.CreateMutexW(None, False, f'Local\\KATDesktop_{identity}_{port}')
    if not handle:
        raise OSError(ctypes.get_last_error(), '无法创建桌面实例锁')
    if ctypes.get_last_error() == 183:
        kernel.CloseHandle(handle)
        user = ctypes.windll.user32
        user.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        user.FindWindowW.restype = ctypes.c_void_p
        user.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        user.SetForegroundWindow.argtypes = [ctypes.c_void_p]
        user.IsIconic.argtypes = [ctypes.c_void_p]
        window = user.FindWindowW(None, TITLE)
        if window:
            user.ShowWindow(window, 9 if user.IsIconic(window) else 5)
            user.SetForegroundWindow(window)
        return None
    return handle


def main(argv=None):
    args = options(argv)
    if os.name != 'nt':
        raise RuntimeError('本桌面原型目前针对Windows和WebView2')
    configure_runtime_paths()
    for directory in (CACHE, EXPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    if args.check_only:
        import importlib.metadata
        print(json.dumps({'desktop_version': importlib.metadata.version('pywebview'),
                          'url': f'http://127.0.0.1:{args.backend_port}/',
                          'web_frontend_modified': False, 'starts_services': False,
                          'exports': str(EXPORTS)}, ensure_ascii=False))
        return 0

    log = (CACHE / 'desktop.log').open('a', encoding='utf-8', buffering=1)
    sys.stdout = sys.stderr = log
    logging.basicConfig(stream=log, level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', force=True)
    handle = acquire_instance(args.backend_port)
    if handle is None:
        return 0
    try:
        import webview
        from webview.menu import Menu, MenuAction, MenuSeparator

        url = f'http://127.0.0.1:{args.backend_port}/'
        profile = CACHE / f'webview2_{args.backend_port}'
        profile.mkdir(exist_ok=True)
        state_path = CACHE / f'instance_{args.backend_port}.json'
        prefs_path = CACHE / 'window.json'
        size = {'width': 1380, 'height': 900}
        try:
            saved = json.loads(prefs_path.read_text(encoding='utf-8'))
            size = {'width': max(1000, min(1920, int(saved['width']))),
                    'height': max(720, min(1200, int(saved['height'])))}
        except (OSError, ValueError, TypeError, KeyError):
            pass
        closed = threading.Event()
        frontend_ready = threading.Event()
        record_lock = threading.Lock()

        def record(status):
            value = {'status': status, 'pid': os.getpid(), 'url': url,
                     'profile': str(profile), 'exports': str(EXPORTS),
                     'services_continue_after_window_close': True,
                     'local_facts_only_requested': args.local_facts_only,
                     'workbench_ready': frontend_ready.is_set(),
                     'debug_port': args.debug_port}
            with record_lock:
                if closed.is_set(): value['status'] = 'closed'
                temporary = state_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
                temporary.replace(state_path)

        def switch_scene(scene):
            if not closed.is_set() and window.get_current_url() == url:
                window.evaluate_js(f'document.querySelector(\'[data-target="{scene}"]\').click()')

        def about():
            show_message(TITLE, '独立桌面窗口，复用原有网页功能与本地服务。\n\n'
                         f'网页版：{url}\n默认导出目录：{EXPORTS}\n'
                         f'日志：{CACHE / "desktop.log"}\n\n'
                         '关闭窗口不会停止本地服务；服务可用原有stop_demo.bat停止。')

        menus = [Menu('工作区', [MenuAction(title, lambda key=scene: switch_scene(key))
                               for title, scene in SCENES]),
                 Menu('工具', [MenuAction('在浏览器中打开', lambda: webbrowser.open(url)),
                               MenuAction('打开导出目录', lambda: os.startfile(EXPORTS)),
                               MenuAction('打开桌面日志目录', lambda: os.startfile(CACHE)),
                               MenuSeparator(), MenuAction('关闭桌面窗口', lambda: window.destroy())]),
                 Menu('帮助', [MenuAction('关于与运行说明', about)])]
        window = webview.create_window(TITLE, html=splash('正在检查并启动本地分析服务，请稍候。\n'
                    '首次启动需要核对运行环境和模型文件。'),
                    width=size['width'], height=size['height'], min_size=(1000, 720),
                    background_color='#0B1628', text_select=True, zoomable=True, maximized=True,
                    menu=menus)
        download_hooked = False
        downloads = {}

        def notify(message):
            def display():
                if closed.is_set():
                    return
                script = '''(()=>{let n=document.getElementById('katDesktopNotice');
                if(!n){n=document.createElement('div');n.id='katDesktopNotice';document.body.appendChild(n)}
                n.style.cssText='position:fixed;right:24px;top:24px;z-index:99999;max-width:560px;padding:16px 22px;background:#07365b;color:white;border:1px solid #46bcb4;border-radius:10px;box-shadow:0 8px 28px #0004;font:14px Microsoft YaHei;white-space:pre-wrap';
                n.textContent=MESSAGE;setTimeout(()=>n.remove(),9000)})()'''.replace('MESSAGE', json.dumps(message, ensure_ascii=False))
                try:
                    window.run_js(script)
                except Exception:
                    logging.exception('Desktop notification failed')
            threading.Thread(target=display, daemon=True).start()

        def on_loaded():
            nonlocal download_hooked
            if closed.is_set():
                return
            try:
                result = install_workbench(window, url)
                if result:
                    frontend_ready.set()
                    record('ready')
                    logging.info('Desktop workbench ready: %s', result.get('version'))
            except Exception:
                logging.exception('Workbench layout failed; original browser entry preserved')
                notify('工作台布局未完整载入。可从“工具”打开网页版继续使用。')
            if download_hooked:
                return
            # Pinned pywebview 6.2.1 exposes its native WebView2 control. Downloads
            # use D: without modal dialogs, and never overwrite existing files.
            from System import Action

            def configure_downloads():
                nonlocal download_hooked
                core = window.native.webview.CoreWebView2
                if download_hooked or core is None:
                    return

                def save_download(sender, event):
                    try:
                        destination = reserve_download(EXPORTS, str(event.ResultFilePath))
                    except OSError:
                        event.Cancel = True
                        logging.exception('Cannot create desktop export')
                        notify('无法保存导出文件，请检查D盘空间及导出目录权限。')
                        return
                    event.ResultFilePath = str(destination)
                    event.Handled = True
                    operation = event.DownloadOperation

                    def finished(sender, state_event):
                        state = str(operation.State)
                        if state not in ('Completed', 'Interrupted'):
                            return
                        if state == 'Completed':
                            logging.info('Download completed: %s', destination.name)
                            notify(f'已保存：{destination.name}\n目录：{EXPORTS}\n可从“工具 → 打开导出目录”查看。')
                        else:
                            logging.warning('Download interrupted: %s', destination.name)
                            notify('导出未完成，请重试；未将部分文件标记为成功。')
                        operation.StateChanged -= finished
                        downloads.pop(str(destination), None)

                    downloads[str(destination)] = (operation, finished)
                    operation.StateChanged += finished

                def download_permission(sender, event):
                    from Microsoft.Web.WebView2.Core import CoreWebView2PermissionState
                    caller, expected = urlsplit(str(event.Uri)), urlsplit(url)
                    if (str(event.PermissionKind) == 'MultipleAutomaticDownloads'
                            and (caller.scheme, caller.hostname, caller.port)
                            == (expected.scheme, expected.hostname, expected.port)):
                        event.State = CoreWebView2PermissionState.Allow
                        event.Handled = True

                core.DownloadStarting -= window.native.browser.on_download_starting
                core.DownloadStarting += save_download
                core.PermissionRequested += download_permission
                core.Profile.DefaultDownloadFolderPath = str(EXPORTS)
                window._desktop_download_handler = save_download  # keep delegate alive
                window._desktop_permission_handler = download_permission
                download_hooked = True

            window.native.Invoke(Action(configure_downloads))

        def on_resize(width, height):
            if width >= 1000 and height >= 720:
                size.update(width=width, height=height)

        def on_closed():
            closed.set()
            prefs_path.write_text(json.dumps(size), encoding='utf-8')
            record('closed')
            logging.info('Desktop window closed; shared web services left running')

        def start_services():
            try:
                run_launcher(args)
                verify_health(read_health(url), args)
                if not closed.is_set():
                    window.load_url(url)
                    record('ready')
                    logging.info('Desktop ready: %s', url)
            except Exception as error:
                logging.exception('Desktop startup failed')
                if not closed.is_set():
                    window.load_html(splash(str(error) + '\n\n原有网页版入口保留，关闭本窗口后可检查日志。', failed=True))
                    record('failed')

        window.events.loaded += on_loaded
        window.events.resized += on_resize
        window.events.closed += on_closed
        webview.settings['ALLOW_DOWNLOADS'] = True
        webview.settings['ALLOW_FILE_URLS'] = False
        webview.settings['OPEN_DEVTOOLS_IN_DEBUG'] = False
        webview.settings['REMOTE_DEBUGGING_PORT'] = args.debug_port
        record('starting')
        logging.info('Starting desktop process %s, target %s', os.getpid(), url)
        webview.start(start_services, gui='edgechromium', storage_path=str(profile), private_mode=False,
                      localization={'global.ok': '确定', 'global.cancel': '取消',
                                    'global.saveFile': '保存文件',
                                    'windows.fileFilter.allFiles': '所有文件'})
        return 0
    except Exception as error:
        logging.exception('Desktop host failed')
        show_message(TITLE, f'桌面窗口启动失败：{error}\n\n日志：{CACHE / "desktop.log"}\n原有网页版入口保留。')
        return 1
    finally:
        ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(handle))


if __name__ == '__main__':
    raise SystemExit(main())
