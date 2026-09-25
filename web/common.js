// Shared across index.html and try.html: theme toggle, scroll reveals,
// and the header's docked/undocked state.

  // Theme toggle: persists the visitor's choice; falls back to OS preference
  // when they've never overridden it (no localStorage entry yet).
  (function () {
    const toggle = document.getElementById('themeToggle');
    const saved = localStorage.getItem('memoryshards_theme');
    if (saved) document.documentElement.setAttribute('data-theme', saved);
    toggle.addEventListener('click', () => {
      const prefersLight = window.matchMedia('(prefers-color-scheme: light)').matches;
      const current = document.documentElement.getAttribute('data-theme') || (prefersLight ? 'light' : 'dark');
      const next = current === 'light' ? 'dark' : 'light';
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
