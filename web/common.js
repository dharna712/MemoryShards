// Shared across index.html and try.html: theme toggle, scroll reveals,
// and the header's docked/undocked state.

  // Theme toggle. The theme itself is set by a tiny script in each page's <head>
  // (white by default) so there is no flash of the wrong theme.
  (function () {
    const toggle = document.getElementById('themeToggle');
    toggle.addEventListener('click', () => {
      const next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('memoryshards_theme', next);
    });
  })();

  // Scroll-triggered reveals: an IntersectionObserver flips `is-visible`
  // once and leaves it — replaces an earlier animation-timeline: view()
  // approach that got stuck at partial/zero opacity once a transformed
  // ancestor (the panel depth-push effect) was involved, silently
  // blanking whole sections of text on scroll.
  (function () {
    const targets = document.querySelectorAll('.reveal, .zip-half, .zip-caption');
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduceMotion) {
      targets.forEach(el => el.classList.add('is-visible'));
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      }
    }, { threshold: 0.15, rootMargin: '0px 0px -10% 0px' });
    targets.forEach(el => observer.observe(el));
  })();

  // Header gains a bottom border + stronger backdrop once the page has
  // scrolled past the hero, so it reads as "docked" rather than floating.
  (function () {
    const header = document.querySelector('header.top');
    const onScroll = () => header.classList.toggle('is-scrolled', window.scrollY > 40);
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  })();


// Magnetic buttons: primary controls lean a few pixels toward the pointer.
(function () {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  document.querySelectorAll('button.primary, .cta-link').forEach((el) => {
    el.addEventListener('pointermove', (e) => {
      const r = el.getBoundingClientRect();
      const dx = (e.clientX - (r.left + r.width / 2)) / r.width;
      const dy = (e.clientY - (r.top + r.height / 2)) / r.height;
      el.style.translate = (dx * 8).toFixed(1) + 'px ' + (dy * 6).toFixed(1) + 'px';
    });
    el.addEventListener('pointerleave', () => { el.style.translate = ''; });
  });
})();

// Background videos: <video class="bg-video" data-src data-src-light>. The file is chosen by
// theme, it only plays while on screen (or, for the sticky hero, while near the top), and it
// stays off entirely for reduced-motion and data-saver.
(function () {
  const vids = document.querySelectorAll('video.bg-video');
  if (!vids.length) return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  if (navigator.connection && navigator.connection.saveData) return;
  const isLight = () => document.documentElement.getAttribute('data-theme') === 'light';
  const visible = new Set();

  function sync(v) {
    const src = isLight() ? v.dataset.srcLight : v.dataset.src;
    if (v.getAttribute('src') !== src) {
      v.classList.remove('is-ready');
      if (v.dataset.poster) v.poster = isLight() ? v.dataset.posterLight : v.dataset.poster;
      v.src = src;
    }
    const farBelow = v.hasAttribute('data-sticky') && window.scrollY > window.innerHeight * 2;
    if (visible.has(v) && !farBelow) v.play().catch(() => {}); else v.pause();
  }

  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) visible.add(e.target); else visible.delete(e.target); sync(e.target); });
  }, { rootMargin: '200px' });
  vids.forEach((v) => {
    v.addEventListener('playing', () => v.classList.add('is-ready'));
    io.observe(v);
    if (v.hasAttribute('data-sticky')) { visible.add(v); sync(v); } // the hero is on screen from the first frame
  });
  new MutationObserver(() => vids.forEach(sync)).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  window.addEventListener('scroll', () => vids.forEach((v) => { if (v.hasAttribute('data-sticky')) sync(v); }), { passive: true });
})();
