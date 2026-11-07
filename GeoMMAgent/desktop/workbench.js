/* Desktop-only composition. Existing controls, IDs and business handlers are retained. */
(() => {
  'use strict';
  const VERSION = '2026.10.workbench.1';
  if (window.KATWorkbench?.version === VERSION) return {ready: true, version: VERSION};
  const originalNodes = new Map([...document.querySelectorAll('[id]')].map(node => [node.id, node]));
  const $ = id => document.getElementById(id) || originalNodes.get(id);
  const originalWrap = document.querySelector('.wrap');
  const originalHeader = document.querySelector('body > header');
  const navigation = $('sceneNav');
  if (!originalWrap || !navigation || !$('mineDrop')) return {ready: false};
  const moves = [];
  const move = (node, parent) => {
    if (!node) throw new Error('工作台所需控件缺失');
    moves.push([node, node.parentNode, node.nextSibling]); parent.appendChild(node); return node;
  };
  const element = (tag, className, text) => {
    const node = document.createElement(tag); if (className) node.className = className;
    if (text) node.textContent = text; return node;
  };
  const paths = {
    image: '<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="8" r="1.5"/><path d="m3 17 6-6 4 4 3-3 5 5"/>',
    compare: '<rect x="3" y="4" width="7" height="16" rx="2"/><rect x="14" y="4" width="7" height="16" rx="2"/><path d="M11 9h2m-2 6h2"/>',
    agents: '<path d="M8 8h8v8H8zM12 3v5m0 8v5M3 12h5m8 0h5"/><circle cx="12" cy="3" r="1"/><circle cx="12" cy="21" r="1"/><circle cx="3" cy="12" r="1"/><circle cx="21" cy="12" r="1"/>',
    folder: '<path d="M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v10H3z"/><path d="M3 10h18"/>',
    book: '<path d="M4 4h6a2 2 0 0 1 2 2v15a3 3 0 0 0-3-2H4zm16 0h-6a2 2 0 0 0-2 2v15a3 3 0 0 1 3-2h5z"/>',
    chart: '<path d="M4 3v17h17M8 15l4-5 4 2 5-7"/>',
    upload: '<path d="M12 16V4m-5 5 5-5 5 5M4 15v5h16v-5"/>',
    help: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 1 1 4 2c-1 .7-1.5 1.2-1.5 2.5M12 17h.01"/>',
    arrow: '<path d="M7 17 17 7M7 7h10v10"/>',
    eye: '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    layers: '<path d="m12 3 10 6-10 6L2 9zm-9 11 9 5 9-5M3 18l9 5 9-5"/>',
    clear: '<path d="m5 6 13 13M18 6 5 19"/>',
    pin: '<path d="M20 10c0 6-8 11-8 11S4 16 4 10a8 8 0 1 1 16 0z"/><circle cx="12" cy="10" r="2.5"/>',
  };
  const icon = name => {
    const node = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    node.setAttribute('viewBox', '0 0 24 24'); node.setAttribute('fill', 'none');
    node.setAttribute('stroke', 'currentColor'); node.setAttribute('stroke-width', '1.7');
    node.setAttribute('stroke-linecap', 'round'); node.setAttribute('stroke-linejoin', 'round');
    node.setAttribute('aria-hidden', 'true'); node.innerHTML = paths[name] || paths.image; return node;
  };
  const button = (label, action, name, className = '') => {
    const node = element('button', `desk-button ${className}`); node.type = 'button';
    node.setAttribute('aria-label', label); node.title = label;
    if (name) node.append(icon(name)); node.append(element('span', '', label));
    node.addEventListener('click', action); return node;
  };
  const scenes = {
    'scene-mine': ['影像检测', '导入矿区影像，圈定范围，查看检测结果与复核报告。', 'image'],
    'scene-compare': ['变化分析', '对比同一范围的两期影像，检查模型候选的变化。', 'compare'],
    'scene-qa': ['智能体问答', '围绕一张影像提问，查看实际执行角色与分析依据。', 'agents'],
    'scene-archive': ['案例归档', '按矿区整理分析记录，回看影像并保存复核意见。', 'folder'],
    'scene-kb': ['安全知识库', '查找规范、处置措施与公开案例，核对资料出处。', 'book'],
    'scene-bench': ['模型评测', '在LMHLD测试图块上查看区域模型与多尺度结果。', 'chart'],
  };

  try {
    const root = element('div', 'desk-shell'); root.id = 'katWorkbench';
    const sidebar = element('aside', 'desk-sidebar');
    const brand = element('div', 'desk-brand');
    const mark = element('span', 'desk-brand-mark'); mark.append(icon('eye'));
    const brandText = element('div'); brandText.append(element('strong', '', '矿安天眼'), element('small', '', '矿区影像分析工作台'));
    brand.append(mark, brandText); sidebar.append(brand);
    sidebar.append(element('div', 'desk-nav-heading', '分析与复核')); move(navigation, sidebar);
    const pills = [...navigation.querySelectorAll('.nav-pill')];
    Object.entries(scenes).forEach(([id, [title, , symbol]], index) => {
      const pill = pills.find(node => node.dataset.target === id);
      if (!pill) throw new Error('工作区导航缺失');
      if (index === 4) navigation.append(element('div', 'desk-nav-heading', '研究与资料'));
      pill.replaceChildren(icon(symbol), element('span', 'desk-nav-label', title));
      pill.setAttribute('role', 'button'); pill.tabIndex = 0; navigation.append(pill);
    });
    const sidebarFoot = element('div', 'desk-sidebar-foot');
    sidebarFoot.append(element('span', 'desk-live-dot'), element('div', '', '本地工作空间'));
    sidebar.append(sidebarFoot, element('div', 'desk-sidebar-caption', '影像 · 证据 · 复核'));

    const workspace = element('main', 'desk-workspace');
    const topbar = element('div', 'desk-topbar');
    const heading = element('div', 'desk-heading');
    const titleLine = element('div', 'desk-title-line');
    const title = element('h1', 'desk-page-title'); const badges = element('div', 'desk-scene-badges');
    titleLine.append(title, badges); const subtitle = element('p', 'desk-page-subtitle'); heading.append(titleLine, subtitle);
    topbar.append(heading);
    const utilities = element('div', 'desk-utilities');
    move(originalHeader.querySelector('.status-dot'), utilities);
    utilities.append(button('网页版', () => window.open(location.origin + '/', '_blank'), 'arrow', 'desk-web-link'));
    const help = button('操作指引', () => helpDialog.showModal(), 'help', 'desk-help'); utilities.append(help); topbar.append(utilities);
    const context = element('div', 'desk-context'); context.append(icon('folder')); move($('activeCaseLabel'), context);
    const stage = element('div', 'desk-stage');
    Object.keys(scenes).forEach(id => {
      const scene = $(id); const badge = scene.querySelector('.scene-badge');
      if (badge) { badge.dataset.scene = id; move(badge, badges); }
      move(scene, stage); scene.classList.add('desk-scene');
    });
    workspace.append(topbar, context, stage);
    const footer = element('div', 'desk-statusbar');
    footer.append(element('span', '', '影像结果用于辅助筛查与人工复核'), element('span', '', '本地归档 · 导出保存至 D 盘'));
    workspace.append(footer); root.append(sidebar, workspace);
    document.body.append(root);
    const helpDialog = element('dialog', 'desk-help-dialog');
    helpDialog.innerHTML = '<div class="desk-help-kicker">开始一次分析</div><h2>从影像到可复核的结果</h2><ol><li><b>导入影像</b><p>选择或拖入矿区影像。GeoTIFF会读取原始波段与地理标签。</p></li><li><b>设置并检测</b><p>可在原图拖动圈定范围，再点击“开始检测”。通常保留默认模型即可。</p></li><li><b>查看结果与报告</b><p>切换原图和候选叠加，查看量测与分级依据，再生成复核报告。</p></li><li><b>保存与交接</b><p>在案例归档中补充意见；导出文件保存在D盘导出目录。</p></li></ol>';
    helpDialog.append(button('知道了', () => helpDialog.close(), null, 'desk-primary'));
    root.append(helpDialog);

    let current = 'scene-mine';
    function syncNavigation() {
      pills.forEach(pill => { const active = pill.dataset.target === current; pill.classList.toggle('active', active); pill.setAttribute('aria-current', active ? 'page' : 'false'); });
      navigation.scrollLeft = 0;
    }
    function selectScene(id) {
      if (!scenes[id]) return;
      current = id; root.dataset.scene = id;
      Object.keys(scenes).forEach(key => { $(key).hidden = key !== id; $(key).style.display = key === id ? 'block' : 'none'; });
      badges.querySelectorAll('[data-scene]').forEach(node => { node.hidden = node.dataset.scene !== id; });
      title.textContent = scenes[id][0]; subtitle.textContent = scenes[id][1]; syncNavigation();
      if (id === 'scene-archive') loadArchiveCases();
      requestAnimationFrame(() => { fitCanvas(); syncNavigation(); });
    }
    navigation.addEventListener('click', event => {
      if (!window.KATWorkbench) return;
      const pill = event.target.closest('.nav-pill'); if (!pill) return;
      event.preventDefault(); event.stopImmediatePropagation(); selectScene(pill.dataset.target);
    }, true);
    navigation.addEventListener('keydown', event => {
      if (!window.KATWorkbench) return;
      if ((event.key === 'Enter' || event.key === ' ') && event.target.matches('.nav-pill')) { event.preventDefault(); selectScene(event.target.dataset.target); }
    });
    window.addEventListener('resize', () => requestAnimationFrame(syncNavigation));
    window.addEventListener('scroll', () => requestAnimationFrame(syncNavigation), {passive: true});

    function panel(label, subtitleText) {
      const node = element('section', 'desk-panel'); const head = element('div', 'desk-panel-head');
      head.append(element('h2', '', label)); if (subtitleText) head.append(element('span', 'desk-muted', subtitleText));
      node.append(head); return node;
    }
    function inspector(labels) {
      const node = element('aside', 'desk-inspector'); const tabs = element('div', 'desk-tabs'); tabs.setAttribute('role', 'tablist');
      const panes = labels.map((label, index) => {
        const pane = element('div', 'desk-tab-panel'); pane.setAttribute('role', 'tabpanel'); pane.hidden = index !== 0;
        const tab = button(label, () => select(index)); tab.setAttribute('role', 'tab'); tab.setAttribute('aria-selected', String(index === 0));
        tabs.append(tab); node.append(pane); return pane;
      });
      node.prepend(tabs);
      function select(index) {
        panes.forEach((pane, i) => { pane.hidden = i !== index; });
        [...tabs.children].forEach((tab, i) => tab.setAttribute('aria-selected', String(i === index)));
        node.dataset.tab = String(index);
      }
      return {node, panes, select};
    }
    function empty(label, description, symbol = 'layers') {
      const node = element('div', 'desk-empty'); node.append(icon(symbol), element('h3', '', label), element('p', '', description)); return node;
    }
    function makeAdvanced(titleText) {
      const detail = element('details', 'desk-advanced'); detail.append(element('summary', '', titleText)); return detail;
    }
    const mine = $('scene-mine').querySelector('.scene-body');
    const originalColumns = mine.querySelector(':scope > .cols');
    const settings = originalColumns.children[1];
    const bandNote = mine.querySelector(':scope > .note');
    const roi = $('roiWrap'); const roiColumns = roi.querySelector('.cols'); const roiExplanation = roiColumns.children[1];
    const layout = element('div', 'desk-analysis-layout');
    const center = element('div', 'desk-analysis-main');
    const workflow = element('div', 'desk-flow');
    ['导入影像', '范围可选', '运行检测', '复核与导出'].forEach((text, i) => { const step = element('div', 'desk-flow-step'); step.append(element('span', '', String(i + 1)), element('b', '', text)); workflow.append(step); });
    center.append(workflow);
    const viewer = panel('影像视图'); viewer.classList.add('desk-viewer');
    const tools = element('div', 'desk-viewer-tools');
    const originalButton = button('原始影像', () => setImageMode('original'), 'eye'); originalButton.id = 'deskOriginalView';
    const overlayButton = button('检测叠加', () => setImageMode('overlay'), 'layers'); overlayButton.id = 'deskOverlayView'; overlayButton.disabled = true;
    const imageModes = element('div', 'desk-view-modes'); imageModes.append(originalButton, overlayButton);
    const chooseImage = button('选择影像', () => $('mineFile').click(), 'upload');
    tools.append(imageModes, chooseImage);
    viewer.querySelector('.desk-panel-head').append(tools);
    const canvasArea = element('div', 'desk-canvas-area'); canvasArea.id = 'deskCanvasArea';
    move($('mineDrop'), canvasArea); move(roi, canvasArea);
    const emptyUploadIcon = $('mineDrop').querySelector('.di'); emptyUploadIcon.replaceChildren(icon('upload'));
    $('mineDrop').querySelector('.dt').textContent = '拖入矿区影像，开始分析';
    $('mineDrop').querySelector('.d-empty').append(element('span', 'desk-choose-file', '选择影像'));
    $('mineFile').setAttribute('aria-label', '选择矿区影像，支持JPG、PNG和GeoTIFF');
    const overlay = element('img', 'desk-overlay-image'); overlay.alt = '同次检测候选叠加'; overlay.hidden = true; canvasArea.append(overlay);
    const waiting = element('div', 'desk-image-loading', '正在读取影像…'); waiting.hidden = true; canvasArea.append(waiting);
    const canvasFooter = element('div', 'desk-canvas-footer');
    const rangeText = element('span', 'desk-range-text', '导入影像后，可直接在图上拖动圈定范围');
    canvasFooter.append(rangeText, button('清除范围', () => { clearRoi(); setImageMode('original'); }, 'clear'));
    const metadata = element('div', 'desk-image-metadata');
    viewer.append(canvasArea, canvasFooter, metadata); center.append(viewer);
    const right = inspector(['分析设置', '检测结果', '复核报告']); right.node.id = 'deskMineInspector';
    const [settingsPane, resultsPane, reportPane] = right.panes;
    settingsPane.append(element('div', 'desk-inspector-kicker', '选择分析方式'), element('h2', 'desk-inspector-title', '检测参数'));
    const fields = [...settings.querySelectorAll(':scope > .field')]; move(fields[0], settingsPane);
    const advanced = makeAdvanced('影像与尺度设置');
    fields.slice(1).forEach(field => move(field, advanced)); move(bandNote, advanced); settingsPane.append(advanced);
    const rangeHelp = makeAdvanced('监测范围与量测说明'); move(roiExplanation, rangeHelp); settingsPane.append(rangeHelp);
    settingsPane.append(element('div', 'desk-guidance', '建议流程：导入影像 → 在原图圈定关注范围（可选）→ 开始检测。'));
    const resultEmpty = empty('尚无检测结果', '完成检测后，在这里查看候选占比、对象量测与筛查依据。');
    resultsPane.append(resultEmpty); move($('uploadResult'), resultsPane);
    const reportEmpty = empty('等待生成复核报告', '先完成检测，再从检测结果中生成报告。协作过程与知识出处会保留。', 'agents');
    reportPane.append(reportEmpty); move($('rp-upload'), reportPane);
    const action = element('div', 'desk-primary-action'); move($('uploadBtn'), action);
    action.append(element('span', '', '结果用于辅助筛查，需结合原图复核')); right.node.append(action);
    layout.append(center, right.node); mine.append(layout); originalColumns.hidden = true;
    originalColumns.classList.add('desk-original-columns');
    let imageMode = 'original', overlaySource = '', priorStats = null, priorFile = null;
    function setImageMode(mode) {
      imageMode = mode === 'overlay' && overlaySource ? 'overlay' : 'original';
      canvasArea.dataset.view = imageMode; overlay.hidden = imageMode !== 'overlay';
      originalButton.classList.toggle('active', imageMode === 'original'); overlayButton.classList.toggle('active', imageMode === 'overlay');
      originalButton.setAttribute('aria-pressed', String(imageMode === 'original'));
      overlayButton.setAttribute('aria-pressed', String(imageMode === 'overlay'));
      updateRangeHint();
      if (imageMode === 'original') requestAnimationFrame(fitCanvas);
    }
    function fitCanvas() {
      if (!window.KATWorkbench || current !== 'scene-mine' || !roiImg || imageMode !== 'original') return;
      const canvas = $('roiCanvas'), bounds = canvasArea.getBoundingClientRect();
      const scale = Math.min((bounds.width - 40) / roiImg.width, (bounds.height - 40) / roiImg.height);
      if (!Number.isFinite(scale) || scale <= 0) return;
      const width = Math.max(1, Math.round(roiImg.width * scale)), height = Math.max(1, Math.round(roiImg.height * scale));
      if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; drawRoi(); }
      if (canvas.style.width !== `${width}px`) canvas.style.width = `${width}px`;
      if (canvas.style.height !== `${height}px`) canvas.style.height = `${height}px`;
    }
    canvasArea.addEventListener('dragover', event => { event.preventDefault(); canvasArea.classList.add('dragover'); });
    canvasArea.addEventListener('dragleave', () => canvasArea.classList.remove('dragover'));
    canvasArea.addEventListener('drop', event => {
      if (event.target.closest('#mineDrop')) return;
      event.preventDefault(); canvasArea.classList.remove('dragover');
      if (event.dataTransfer?.files[0]) handleMineFile(event.dataTransfer.files[0]);
    });
    $('uploadBtn').addEventListener('click', () => { if (mineFileObj) right.select(1); });

    // Comparison: inputs/controls remain together, output is a separate canvas card.
    const comparison = $('scene-compare').querySelector('.scene-body');
    const compareColumns = comparison.querySelector(':scope > .cols');
    const compareOptions = comparison.querySelector(':scope > .note');
    const compareLayout = element('div', 'desk-analysis-layout desk-compare-layout');
    const compareMain = element('div', 'desk-compare-main');
    const pairPanel = panel('两期影像', '先选择T1，再选择T2'); move(compareColumns, pairPanel);
    const changePanel = panel('变化结果', '仅在共同有效范围内比较');
    const changeEmpty = empty('准备对比两期影像', '选择同一区域的两期影像，检查位置对应方式，再运行变化分析。', 'compare');
    changePanel.append(changeEmpty); move($('cmpResult'), changePanel); compareMain.append(pairPanel, changePanel);
    const compareRight = inspector(['分析设置', '复核报告']);
    compareRight.panes[0].append(element('h2', 'desk-inspector-title', '对比参数')); move(compareOptions, compareRight.panes[0]);
    move($('cmpReportBtn'), compareRight.panes[0]); compareRight.panes[0].append(element('div', 'desk-guidance', '采集日期与位置对应均影响变化解释。模型候选的新增或消退，需要结合原图和作业记录复核。'));
    const compareReportEmpty = empty('等待变化趋势报告', '完成变化分析后，可以生成同次报告并导出交接。', 'agents');
    compareRight.panes[1].append(compareReportEmpty); move($('rp-compare'), compareRight.panes[1]);
    const compareAction = element('div', 'desk-primary-action'); move($('cmpBtn'), compareAction); compareRight.node.append(compareAction);
    compareLayout.append(compareMain, compareRight.node); comparison.append(compareLayout);
    compareMain.classList.add('desk-scroll');

    // Other tasks retain their full controls with the same desktop visual system.
    ['scene-bench', 'scene-qa', 'scene-kb', 'scene-archive'].forEach(id => $(id).querySelector('.scene-body').classList.add('desk-document-page'));
    $('scene-qa').querySelector('.cols').classList.add('desk-qa-grid');
    $('qaDrop').querySelector('.di').replaceChildren(icon('image'));
    $('qaPanel').querySelector('.empty .ei').replaceChildren(icon('agents'));
    document.querySelectorAll('#regionChips .region-chip').forEach(chip => {
      chip.textContent = chip.textContent.replace(/^[\p{Regional_Indicator}\s]+/u, '');
    });
    const kbPrompt = element('div', 'desk-kb-prompt'); kbPrompt.append(element('h2', '', '查找一条可核对的依据'), element('p', '', '输入主题或问题，查看相关知识片段与来源。'));
    $('scene-kb').querySelector('.scene-body').prepend(kbPrompt);

    const archiveLayout = $('scene-archive').querySelector('.archive-layout');
    const archiveWorkspace = archiveLayout.querySelector('.archive-workspace');
    const archiveTabs = inspector(['记录与影像', '分析报告', '人工复核']);
    archiveTabs.node.classList.add('desk-archive-tabs');
    move(archiveWorkspace, archiveTabs.panes[0]);
    const archivedReportEmpty = empty('选择一次历史分析', '在案例中打开一条分析记录，查看同次报告与来源。', 'folder');
    archiveTabs.panes[1].append(archivedReportEmpty); move($('rp-archive'), archiveTabs.panes[1]);
    const reviewEmpty = empty('等待选择复核记录', '打开历史分析后，可以补记核查位置并保存人工意见。', 'pin');
    archiveTabs.panes[2].append(reviewEmpty); move($('archiveReviewPanel'), archiveTabs.panes[2]);
    archiveLayout.append(archiveTabs.node);
    let priorCase = null, priorArchivedReport = null;
    function syncArchive() {
      const selected = archiveReport?.report_id;
      archivedReportEmpty.hidden = Boolean(selected); reviewEmpty.hidden = Boolean(selected);
      if (activeCaseId !== priorCase) {
        priorCase = activeCaseId;
        if (activeCaseId) archiveWorkspace.querySelector('.archive-editor').open = false;
        archiveTabs.select(0);
      }
      if (selected !== priorArchivedReport) { priorArchivedReport = selected; archiveTabs.select(selected ? 2 : 0); }
    }
    const archiveWatch = new MutationObserver(() => queueMicrotask(syncArchive));
    archiveWatch.observe($('archiveReports'), {childList: true, subtree: true});
    archiveWatch.observe($('archiveReviewPanel'), {attributes: true, attributeFilter: ['style']});

    function organizeDetectionResult(stats) {
      const info = $('uploadResult').querySelector('.rc-info');
      if (!info || info.dataset.workbenchOrganized) return;
      info.dataset.workbenchOrganized = 'true';
      info.prepend(element('div', 'desk-result-scope', stats.scope === 'roi' ? '当前圈定监测范围' : '整幅影像范围'));
      for (const node of info.childNodes) if (node.nodeType === Node.TEXT_NODE && node.textContent.includes('风险区域')) node.textContent = node.textContent.replace('风险区域', '候选区域');
      const notes = [...info.querySelectorAll(':scope > .note')];
      const disclosure = makeAdvanced('影像条件与适用范围'); disclosure.classList.add('desk-result-notes');
      notes.forEach(note => disclosure.append(note)); info.append(disclosure);
      const foot = $('uploadResult').querySelector('.rc-foot');
      if (foot) {
        const label = foot.querySelector('.btn-text'); if (label) label.textContent = '生成复核报告';
        info.insertBefore(foot, info.querySelector(':scope > details'));
      }
    }

    const stripEmoji = node => {
      if (!node) return;
      node.querySelectorAll('.btn-text,.pr-label,.kb-title,.step-title').forEach(label => {
        for (const child of label.childNodes) if (child.nodeType === Node.TEXT_NODE) {
          const next = child.textContent.replace(/^[\s\p{Extended_Pictographic}\uFE0F\u200D]+/u, '');
          if (next !== child.textContent) child.textContent = next;
        }
      });
    };
    let queued = false;
    function updateRangeHint() {
      rangeText.textContent = imageMode === 'overlay' ? '红色为模型候选；切换原始影像可调整监测范围' : roiFrac ? `已圈定图幅的 ${((roiFrac.x1 - roiFrac.x0) * (roiFrac.y1 - roiFrac.y0) * 100).toFixed(0)}% · 统计按范围内计算` : roiImg ? '在原图上拖动圈定范围；未圈定时按整幅统计' : '导入影像后，可直接在图上拖动圈定范围';
    }
    function sync() {
      queued = false;
      const loaded = Boolean(roiImg && mineFileObj), stats = lastGeomStats;
      if (stats) organizeDetectionResult(stats);
      canvasArea.classList.toggle('has-image', loaded); waiting.hidden = !mineFileObj || loaded;
      updateRangeHint();
      const file = mineFileObj;
      const chooseLabel = chooseImage.querySelector('span');
      const chooseText = file ? '更换影像' : '选择影像';
      if (chooseLabel.textContent !== chooseText) { chooseLabel.textContent = chooseText; chooseImage.setAttribute('aria-label', chooseText); chooseImage.title = chooseText; }
      const geo = lastMeta?.georeference;
      const fileMeta = file ? [file.name, `${(file.size / 1024 / 1024).toFixed(2)} MB`, geo ? `${geo.width} × ${geo.height} px` : '', currentGsd() ? `${currentGsd()} 米/像素` : '米制尺度待确认'].filter(Boolean) : ['支持 JPG / PNG / GeoTIFF', '原始影像与检测量测保持关联'];
      const metaText = fileMeta.join('   ·   '); if (metadata.textContent !== metaText) metadata.textContent = metaText;
      if (!$('uploadBtn').classList.contains('running')) $('uploadBtn').disabled = !file;
      if (!$('uploadBtn').classList.contains('running')) {
        const label = $('uploadBtn').querySelector('.btn-text');
        const caption = stats ? '重新检测' : '开始检测';
        if (label.textContent !== caption) label.textContent = caption;
      }
      action.classList.toggle('desk-repeat', Boolean(stats));
      resultEmpty.hidden = Boolean($('uploadResult').textContent.trim());
      const resultImage = $('uploadResult').querySelector('.rc-main > img');
      const nextOverlay = stats && resultImage ? resultImage.src : '';
      if (nextOverlay !== overlaySource) {
        overlaySource = nextOverlay;
        if (overlaySource) { overlay.src = overlaySource; right.select(1); setImageMode('overlay'); }
        else { overlay.removeAttribute('src'); setImageMode('original'); }
      }
      overlayButton.disabled = !overlaySource;
      if (priorStats && !stats && !$('uploadBtn').classList.contains('running')) right.select(0);
      if (file !== priorFile) { priorFile = file; setImageMode('original'); }
      priorStats = stats;
      const hasReport = $('rp-upload').classList.contains('show'); reportEmpty.hidden = hasReport;
      const compareHasReport = $('rp-compare').classList.contains('show'); compareReportEmpty.hidden = compareHasReport;
      changeEmpty.hidden = Boolean($('cmpResult').textContent.trim());
      [...workflow.children].forEach((step, index) => { step.classList.toggle('done', index === 0 ? loaded : index === 1 ? Boolean(roiFrac) : index === 2 ? Boolean(stats) : Boolean(reportParams.upload?.report_id)); });
      stripEmoji(root); fitCanvas();
    }
    function queueSync() { if (!queued) { queued = true; queueMicrotask(sync); } }
    const watch = new MutationObserver(queueSync);
    ['mineDrop', 'mineThumb', 'mineName', 'roiWrap', 'uploadResult', 'rp-upload', 'cmpResult', 'rp-compare', 'sampleGrid', 'qaPanel', 'kbResults'].forEach(id => watch.observe($(id), {childList: true, subtree: true, attributes: true, characterData: true}));
    const reportObserver = new MutationObserver(() => { if ($('rp-upload').classList.contains('show')) right.select(2); queueSync(); });
    reportObserver.observe($('rp-upload'), {attributes: true, attributeFilter: ['class']});
    const compareObserver = new MutationObserver(() => { if ($('rp-compare').classList.contains('show')) compareRight.select(1); queueSync(); });
    compareObserver.observe($('rp-compare'), {attributes: true, attributeFilter: ['class']});
    new ResizeObserver(() => requestAnimationFrame(fitCanvas)).observe(canvasArea);
    ['mineFile', 'uploadModel', 'uploadGrid', 'gsdInput', 'uploadBands'].forEach(id => { $(id).addEventListener('change', queueSync); $(id).addEventListener('input', queueSync); });
    originalHeader.hidden = true; originalWrap.hidden = true;
    document.body.classList.add('kat-workbench');
    window.KATWorkbench = {version: VERSION, selectScene, refresh: queueSync, currentScene: () => current, setImageMode};
    selectScene(current); sync(); syncArchive();
    return {ready: true, version: VERSION, scenes: Object.keys(scenes).length};
  } catch (error) {
    for (const [node, parent, next] of moves.reverse()) if (parent) parent.insertBefore(node, next?.parentNode === parent ? next : null);
    document.getElementById('katWorkbench')?.remove();
    originalHeader.hidden = false; originalWrap.hidden = false; document.body.classList.remove('kat-workbench');
    throw error;
  }
})();
