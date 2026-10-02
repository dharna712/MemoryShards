/* MemoryShards, MARE direction: scroll choreography for the pitch page.
   GSAP + ScrollTrigger + Lenis (self-hosted in ./vendor). One master timeline for the hero (1 time unit == the hero's scroll length). */
(function () {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  gsap.registerPlugin(ScrollTrigger);

  // smooth scroll, driven by GSAP's ticker so ScrollTrigger stays in sync
  let lenis = null;
  if (!reduced) {
    lenis = new Lenis({ lerp: 0.1, anchors: true });
    lenis.on('scroll', ScrollTrigger.update);
    gsap.ticker.add((t) => lenis.raf(t * 1000));
    gsap.ticker.lagSmoothing(0);
  }
  const toTop = $('#toTop');
  if (toTop) toTop.addEventListener('click', (e) => { e.preventDefault(); lenis ? lenis.scrollTo(0, { duration: 2 }) : scrollTo({ top: 0 }); });

  // ---------- hero: shards rejoin, then the photograph opens to full bleed ----------
  const stage = $('#stage'), pieces = $('#pieces'), whole = $('#whole');
  const shards = $$('.shard', pieces);
  const SCATTER = [[-13, -13, -14], [-2, -17, 9], [12, -11, 16], [16, 4, -11], [11, 14, 13], [-2, 17, -9], [-13, 12, 11], [-17, -3, -15]]; // x%, y%, rotate: the original shatter, kept
  if (!reduced) {
    shards.forEach((s, i) => gsap.set(s, { xPercent: SCATTER[i][0] * 1.6, yPercent: SCATTER[i][1] * 1.6, rotation: SCATTER[i][2] * 1.5 }));
    gsap.from('.hero-title .line', { yPercent: 112, duration: 1.4, ease: 'expo.out', stagger: 0.12, delay: 0.15 });
    gsap.from('.top', { opacity: 0, y: -10, duration: 1, delay: 0.6 });

    const tl = gsap.timeline({
      defaults: { ease: 'none', immediateRender: false },
      scrollTrigger: { trigger: '.hero', start: 'top top', end: 'bottom bottom', scrub: 0.6, invalidateOnRefresh: true },
    });
    tl.to(shards, { xPercent: 0, yPercent: 0, rotation: 0, duration: 0.4, ease: 'power2.inOut', stagger: 0.014 }, 0)
      .to('#cracks', { opacity: 0, duration: 0.3 }, 0.18)
      .to('.hero-ask', { autoAlpha: 0, duration: 0.12 }, 0.03)
      // the shards tile the whole frame exactly, so swapping to the single photo here is invisible
      .set(pieces, { autoAlpha: 0 }, 0.43)
      .set(whole, { autoAlpha: 1 }, 0.43)
      .to(stage, { width: () => innerWidth, height: () => innerHeight, duration: 0.42, ease: 'power2.inOut' }, 0.43)
      .to('.hero-title', { yPercent: -70, autoAlpha: 0, duration: 0.28, ease: 'power2.in' }, 0.42)
      .to('#scrim', { opacity: 1, duration: 0.18 }, 0.78)
      .fromTo('#heroEnd', { opacity: 0, y: 40 }, { opacity: 1, y: 0, duration: 0.18, ease: 'power2.out' }, 0.8)
      .to('#heroCredit', { opacity: 1, duration: 0.14 }, 0.88)
      .set({}, {}, 1); // pad so the timeline length equals the scroll length
  }

  // header ink: light over the full-bleed photo and the dark footer, dark everywhere else
  const topbar = $('.top');
  if (!reduced) {
    ScrollTrigger.create({
      trigger: '.hero', start: 'top top', end: 'bottom bottom',
      onUpdate: (self) => topbar.classList.toggle('on-photo', self.progress > 0.52),
      onLeave: () => topbar.classList.remove('on-photo'),
      onLeaveBack: () => topbar.classList.remove('on-photo'),
    });
    ScrollTrigger.create({ trigger: '.foot', start: 'top 40px', end: 'bottom top', toggleClass: { targets: topbar, className: 'on-photo' } });
  } else {
    topbar.classList.add('on-photo');
  }

  // ---------- statement: words ink in as you read ----------
  const st = $('#statement');
  if (st) {
    const words = st.textContent.trim().split(/\s+/);
    const payFrom = words.length - 4; // "a day out."
    st.setAttribute('aria-label', st.textContent.trim());
    st.innerHTML = words.map((w, i) => `<span class="w${i >= payFrom ? ' pay' : ''}" aria-hidden="true">${w} </span>`).join('');
    if (!reduced) {
      gsap.fromTo('.statement .w', { opacity: 0.16 }, {
        opacity: 1, ease: 'none', stagger: 0.12,
        scrollTrigger: { trigger: '.statement', start: 'top 72%', end: 'bottom 58%', scrub: true },
      });
    }
  }

  // ---------- pipeline: pinned horizontal essay ----------
  if (!reduced) {
    const track = $('#track');
    const dist = () => track.scrollWidth - innerWidth;
    gsap.to(track, {
      x: () => -dist(), ease: 'none',
      scrollTrigger: { trigger: '.stages', pin: true, scrub: 0.6, start: 'top top', end: () => '+=' + dist(), invalidateOnRefresh: true },
    });
    gsap.to('.progress i', {
      scaleX: 1, ease: 'none',
      scrollTrigger: { trigger: '.stages', start: 'top top', end: () => '+=' + dist(), scrub: true, invalidateOnRefresh: true },
    });
  }

  // ---------- proof: numerals count up once ----------
  $$('.proof-big').forEach((el) => {
    const target = parseFloat(el.dataset.count), dec = parseInt(el.dataset.decimals || '0', 10), suf = el.dataset.suffix || '';
    if (reduced) return;
    el.textContent = (0).toFixed(dec) + suf;
    const o = { v: 0 };
    gsap.to(o, {
      v: target, duration: 1.8, ease: 'power3.out',
      scrollTrigger: { trigger: el, start: 'top 85%', once: true },
      onUpdate: () => { el.textContent = (dec ? o.v.toFixed(dec) : Math.round(o.v)) + suf; },
    });
  });

  // ---------- index: photo follows the cursor ----------
  const peek = $('#peek'), peekImg = $('#peekImg');
  if (peek) {
    const mx = gsap.quickTo(peek, 'x', { duration: reduced ? 0 : 0.5, ease: 'power3' });
    const my = gsap.quickTo(peek, 'y', { duration: reduced ? 0 : 0.5, ease: 'power3' });
    addEventListener('pointermove', (e) => { mx(e.clientX + 26); my(e.clientY - 100); });
    $$('.row').forEach((r) => {
      const show = () => { peekImg.src = r.dataset.img; peek.classList.add('on'); };
      const hide = () => peek.classList.remove('on');
      r.addEventListener('pointerenter', show);
      r.addEventListener('pointerleave', hide);
      r.addEventListener('focus', show);
      r.addEventListener('blur', hide);
    });
    if (!reduced) {
      $$('#rows li').forEach((li) => gsap.from(li, { yPercent: 40, opacity: 0, duration: 0.9, ease: 'expo.out', scrollTrigger: { trigger: li, start: 'top 92%', once: true } }));
    }
  }

  // ---------- footer wordmark ----------
  if (!reduced) {
    gsap.from('.big-word span', { yPercent: 105, duration: 1.3, ease: 'expo.out', stagger: 0.05, scrollTrigger: { trigger: '.foot', start: 'top 72%', once: true } });
  }

  const refresh = () => ScrollTrigger.refresh();
  addEventListener('load', refresh);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(refresh);
})();
