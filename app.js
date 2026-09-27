(() => {
  'use strict';
  const data = window.STORYENGINE_DATA;
  if (!data) return;
  const $ = id => document.getElementById(id);
  const all = selector => [...document.querySelectorAll(selector)];
  const clock = seconds => `${Math.floor(Math.max(0, seconds) / 60)}:${String(Math.floor(Math.max(0, seconds) % 60)).padStart(2, '0')}`;
  const escapeHTML = value => String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
  const press = (selector, chosen, attribute) => all(selector).forEach(button => button.setAttribute('aria-pressed', String(button.dataset[attribute] === chosen)));

  // Local anchors work both at a domain root and under a GitHub Pages project path.
  const navToggle = $('nav-toggle');
  const navLinks = $('nav-links');
  function closeNav() {
    navToggle?.setAttribute('aria-expanded', 'false');
    navToggle?.setAttribute('aria-label', 'Open navigation');
    navLinks?.classList.remove('is-open');
  }
  navToggle?.addEventListener('click', () => {
    const opened = navToggle.getAttribute('aria-expanded') !== 'true';
    navToggle.setAttribute('aria-expanded', String(opened));
    navToggle.setAttribute('aria-label', opened ? 'Close navigation' : 'Open navigation');
    navLinks?.classList.toggle('is-open', opened);
  });
  navLinks?.querySelectorAll('a').forEach(link => link.addEventListener('click', closeNav));
  document.addEventListener('keydown', event => { if (event.key === 'Escape') closeNav(); });

  const showcase = $('showcase-video');
  let currentStory = data.showcases[0].id;
  function showStory(id) {
    const story = data.showcases.find(item => item.id === id);
    if (!story || !showcase) return;
    const wasPlaying = !showcase.paused;
    showcase.pause();
    showcase.poster = story.poster;
    showcase.src = story.video;
    showcase.preload = 'none';
    showcase.setAttribute('aria-label', `${story.title}, ${story.backbone}, ${story.duration} seconds`);
    $('showcase-title').textContent = story.title;
    $('showcase-description').textContent = story.description;
    currentStory = id;
    const index = data.showcases.findIndex(item => item.id === id);
    if ($('showcase-counter')) $('showcase-counter').textContent = `${String(index + 1).padStart(2, '0')} / ${String(data.showcases.length).padStart(2, '0')}`;
    if ($('showcase-focus')) $('showcase-focus').textContent = `Watch for ${story.focus || 'continuity across shots'}.`;
    press('[data-showcase]', id, 'showcase');
    if (wasPlaying) showcase.play().catch(() => {});
  }
  all('[data-showcase]').forEach(button => button.addEventListener('click', () => showStory(button.dataset.showcase)));
  function nextStory(offset) {
    const index = data.showcases.findIndex(item => item.id === currentStory);
    showStory(data.showcases[(index + offset + data.showcases.length) % data.showcases.length].id);
  }
  $('showcase-prev')?.addEventListener('click', () => nextStory(-1));
  $('showcase-next')?.addEventListener('click', () => nextStory(1));
  all('[data-showcase]').forEach(button => button.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    if (event.key === 'Home') showStory(data.showcases[0].id);
    else if (event.key === 'End') showStory(data.showcases[data.showcases.length - 1].id);
    else nextStory(event.key === 'ArrowRight' ? 1 : -1);
    document.querySelector(`[data-showcase="${currentStory}"]`)?.focus();
  }));
  showStory(data.showcases[0].id);
  showcase?.addEventListener('play', () => { pauseComparison(); });

  // Start all clips at the same elapsed time. Do not stretch the shorter originals.
  const methodKeys = ['ours', 'movieagent', 'vimax'];
  const compareVideos = methodKeys.map(method => $('compare-' + method));
  const playButton = $('comparison-play');
  const seek = $('comparison-seek');
  const timeLabel = $('comparison-time');
  let activeComparison = data.comparisons[0];
  let groupPlaying = false;
  let groupTime = 0;
  let animation = 0;
  let preparing = false;
  let generation = 0;
  let comparisonRate = 1;
  const automaticallyPaused = new WeakSet();
  const feedback = document.createElement('p');
  feedback.id = 'comparison-status';
  feedback.className = 'comparison-status';
  feedback.setAttribute('role', 'status');
  feedback.setAttribute('aria-live', 'polite');
  timeLabel?.parentElement?.after(feedback);
  function maxTime() { return Math.max(...methodKeys.map(key => activeComparison.methods[key].duration)); }
  function setComparisonRate(value) {
    const rate = Number(value);
    if (![0.5, 1].includes(rate)) return;
    comparisonRate = rate;
    compareVideos.forEach(video => { video.defaultPlaybackRate = rate; video.playbackRate = rate; });
    press('[data-speed]', String(rate), 'speed');
  }
  function leader() {
    let best = 0;
    methodKeys.forEach((key, index) => { if (activeComparison.methods[key].duration > activeComparison.methods[methodKeys[best]].duration) best = index; });
    return compareVideos[best];
  }
  function renderComparisonTime() {
    const length = maxTime();
    seek.value = String(Math.round(groupTime / length * 1000));
    seek.setAttribute('aria-valuetext', `${clock(groupTime)} of ${clock(length)}`);
    timeLabel.textContent = `${clock(groupTime)} / ${clock(length)}`;
  }
  function syncPlayButton() {
    playButton.textContent = preparing ? 'Loading clips…' : groupPlaying ? 'Pause together' : 'Play together';
    playButton.setAttribute('aria-pressed', String(groupPlaying));
    playButton.disabled = preparing;
  }
  function pauseComparison() {
    // A pause also cancels an outstanding metadata/play request. Otherwise a
    // reset, seek, or offscreen pause could restart playback after loading.
    generation += 1;
    preparing = false;
    groupPlaying = false;
    cancelAnimationFrame(animation);
    compareVideos.forEach(video => video?.pause());
    if (playButton) syncPlayButton();
  }
  function lastFrameTime(video, index) {
    const duration = activeComparison.methods[methodKeys[index]].duration;
    return Math.max(0, Math.min(Number.isFinite(video.duration) ? video.duration : duration, duration) - 0.001);
  }
  function setPosition(seconds) {
    groupTime = Math.max(0, Math.min(seconds, maxTime()));
    compareVideos.forEach((video, index) => {
      if (!video || video.readyState < 1) return;
      video.currentTime = Math.min(groupTime, lastFrameTime(video, index));
    });
    renderComparisonTime();
  }
  function metadata(video) {
    if (video.readyState >= 1) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => finish(new Error('Video metadata timeout')), 20000);
      function finish(error) {
        clearTimeout(timeout);
        video.removeEventListener('loadedmetadata', loaded);
        video.removeEventListener('error', failed);
        error ? reject(error) : resolve();
      }
      function loaded() { finish(); }
      function failed() { finish(new Error('Video unavailable')); }
      video.addEventListener('loadedmetadata', loaded);
      video.addEventListener('error', failed);
      video.preload = 'metadata';
      video.load();
    });
  }
  function tick() {
    if (!groupPlaying) return;
    groupTime = Math.min(leader().currentTime, maxTime());
    compareVideos.forEach((video, index) => {
      const clipDuration = activeComparison.methods[methodKeys[index]].duration;
      if (groupTime >= clipDuration - 0.05) {
        const lastFrame = lastFrameTime(video, index);
        if (video.readyState >= 1 && Math.abs(video.currentTime - lastFrame) > 0.002) video.currentTime = lastFrame;
        if (!video.paused) {
          automaticallyPaused.add(video);
          video.pause();
        }
      } else if (video.readyState >= 2 && Math.abs(video.currentTime - groupTime) > 0.35) {
        video.currentTime = groupTime;
      }
    });
    renderComparisonTime();
    if (leader().ended || groupTime >= maxTime() - 0.06) {
      groupTime = maxTime();
      pauseComparison();
      renderComparisonTime();
      return;
    }
    animation = requestAnimationFrame(tick);
  }
  async function playComparison() {
    if (preparing) return;
    if (groupPlaying) { pauseComparison(); return; }
    const request = generation;
    preparing = true;
    feedback.textContent = '';
    syncPlayButton();
    try {
      await Promise.all(compareVideos.map(metadata));
      if (request !== generation) return;
      if (groupTime >= maxTime() - 0.08) groupTime = 0;
      setPosition(groupTime);
      setComparisonRate(comparisonRate);
      showcase?.pause();
      await Promise.all(compareVideos.map((video, index) => groupTime < activeComparison.methods[methodKeys[index]].duration - 0.04 ? video.play() : Promise.resolve()));
      if (request !== generation) return;
      groupPlaying = true;
      animation = requestAnimationFrame(tick);
    } catch (_) {
      if (request === generation) {
        pauseComparison();
        feedback.textContent = 'A clip could not be loaded. Please try again, or use the individual video controls.';
      }
    } finally {
      if (request === generation) {
        preparing = false;
        syncPlayButton();
      }
    }
  }
  function chooseComparison(id) {
    const item = data.comparisons.find(example => example.id === id);
    if (!item) return;
    pauseComparison();
    activeComparison = item;
    groupTime = 0;
    $('comparison-title').textContent = item.title;
    $('comparison-description').textContent = item.description;
    const watchFor = $('comparison-watchfor');
    if (watchFor) watchFor.replaceChildren(...(item.watchFor || []).map(text => {
      const point = document.createElement('li');
      point.textContent = text;
      return point;
    }));
    compareVideos.forEach((video, index) => {
      const clip = item.methods[methodKeys[index]];
      video.src = clip.video;
      video.poster = clip.poster;
      video.preload = 'none';
      video.muted = true;
      video.setAttribute('aria-label', `${item.title}: ${methodKeys[index] === 'ours' ? 'StoryEngine' : methodKeys[index] === 'vimax' ? 'ViMAX' : 'MovieAgent'}`);
    });
    feedback.textContent = '';
    press('[data-comparison]', id, 'comparison');
    setComparisonRate(comparisonRate);
    renderComparisonTime();
    syncPlayButton();
  }
  all('[data-comparison]').forEach(button => button.addEventListener('click', () => chooseComparison(button.dataset.comparison)));
  all('[data-speed]').forEach(button => button.addEventListener('click', () => setComparisonRate(button.dataset.speed)));
  playButton?.addEventListener('click', playComparison);
  $('comparison-reset')?.addEventListener('click', () => { pauseComparison(); setPosition(0); });
  seek?.addEventListener('input', () => { pauseComparison(); setPosition(Number(seek.value) / 1000 * maxTime()); });
  compareVideos.forEach(video => {
    video.addEventListener('play', () => {
      if (!groupPlaying && !preparing) { showcase?.pause(); }
    });
    video.addEventListener('pause', () => {
      if (automaticallyPaused.has(video)) { automaticallyPaused.delete(video); return; }
      if (groupPlaying && !video.ended && video.currentTime < video.duration - 0.1) pauseComparison();
    });
  });
  chooseComparison(data.comparisons[0].id);

  const metricIds = ['PEC', 'ECS', 'APR', 'SPS', 'LAR', 'GPC', 'MIR', 'LCS', 'Avg'];
  function showResults(id) {
    const name = id === 'wan' ? 'Wan2.2-TI2V-5B' : 'Veo 3.1';
    const rows = data.mainResults.filter(row => row.backbone === name);
    const maxima = Object.fromEntries(metricIds.map(metric => [metric, Math.max(...rows.map(row => row[metric] ?? -Infinity))]));
    $('results-body').innerHTML = rows.map(row => {
      const isOurs = row.method === 'StoryEngine';
      return `<tr${isOurs ? ' class="ours"' : ''}><th scope="row">${escapeHTML(row.method)}</th>` + metricIds.map(metric => {
        const value = row[metric];
        return `<td${value != null && value === maxima[metric] ? ' class="best"' : ''}>${value == null ? '<span aria-label="Not applicable">—</span>' : value.toFixed(4)}</td>`;
      }).join('') + '</tr>';
    }).join('');
    press('[data-backbone]', id, 'backbone');
    $('results-table').setAttribute('aria-label', `Automatic benchmark results with ${name}`);
    const label = $('results-backbone-label');
    if (label) label.textContent = name;
  }
  all('[data-backbone]').forEach(button => button.addEventListener('click', () => showResults(button.dataset.backbone)));
  showResults('veo');

  const dialog = $('figure-dialog');
  all('[data-zoom]').forEach(button => button.addEventListener('click', () => {
    $('figure-dialog-image').src = button.dataset.zoom;
    $('figure-dialog-image').alt = button.querySelector('img')?.alt || button.dataset.caption || 'Paper figure';
    $('figure-dialog-caption').textContent = button.dataset.caption || '';
    if (!dialog.open) dialog.showModal();
  }));
  $('figure-dialog-close')?.addEventListener('click', () => dialog.close());
  dialog?.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });

  $('bibtex').textContent = data.preliminaryBibtex;
  $('copy-bibtex')?.addEventListener('click', async () => {
    const value = data.preliminaryBibtex;
    let copied = false;
    try { await navigator.clipboard.writeText(value); copied = true; } catch (_) {
      const field = document.createElement('textarea');
      field.value = value;
      field.style.cssText = 'position:fixed;opacity:0;pointer-events:none;';
      document.body.append(field);
      field.select();
      copied = document.execCommand('copy');
      field.remove();
      $('copy-bibtex').focus();
    }
    $('copy-status').textContent = copied ? 'Citation copied.' : 'Select the citation below and copy it manually.';
  });

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) { showcase?.pause(); pauseComparison(); }
  });
  new IntersectionObserver(entries => {
    if (!entries[0].isIntersecting) pauseComparison();
  }, { threshold: 0 }).observe($('comparisons'));
  if (showcase) new IntersectionObserver(entries => {
    if (!entries[0].isIntersecting) showcase.pause();
  }, { threshold: 0.05 }).observe(showcase);

  const sectionLinks = all('#nav-links a[href^="#"]');
  const visibleSections = new Map();
  const sectionObserver = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (entry.isIntersecting) visibleSections.set(entry.target.id, entry.target);
      else visibleSections.delete(entry.target.id);
    }
    const active = [...visibleSections.values()].sort((a, b) => Math.abs(a.getBoundingClientRect().top - 90) - Math.abs(b.getBoundingClientRect().top - 90))[0];
    if (!active) return;
    sectionLinks.forEach(link => {
      const isCurrent = link.getAttribute('href') === `#${active.id}`;
      link.classList.toggle('is-current', isCurrent);
      if (isCurrent) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }, { rootMargin: '-12% 0px -55% 0px', threshold: 0 });
  sectionLinks.forEach(link => {
    const section = document.querySelector(link.getAttribute('href'));
    if (section) sectionObserver.observe(section);
  });
})();
