"""Builds web/mare/try.html from the untouched web/try.html: same markup and the same 30KB script (API behaviour unchanged),
with paths fixed, the header/footer swapped for the MARE ones, the video backdrop removed and the MARE stylesheets layered on top.
Run from the repo root: python tools/build_mare_try.py"""
import re

src = open('web/try.html', encoding='utf8').read()
s = src

# head: drop webfont links, theme script (always light), swap stylesheets
s = re.sub(r'<link rel="preconnect"[^>]*>\s*', '', s)
s = re.sub(r'<link href="https://(fonts\.googleapis|api\.fontshare)[^>]*>\s*', '', s)
s = re.sub(r'<script>\s*document\.documentElement\.setAttribute\(\'data-theme\'.*?</script>\s*', '', s, flags=re.S)
s = s.replace('<html lang="en">', '<html lang="en" data-theme="light">')
s = re.sub(r'<meta name="theme-color"[^>]*>', '<meta name="theme-color" content="#f3efe6">', s)
s = s.replace('<link rel="stylesheet" href="style.css">',
              '<link rel="preload" href="fonts/instrument-serif-latin-400-normal.woff2" as="font" type="font/woff2" crossorigin>\n'
              '<link rel="stylesheet" href="../style.css">\n<link rel="stylesheet" href="mare.css">\n<link rel="stylesheet" href="try-skin.css">')
s = re.sub(r'<title>.*?</title>', '<title>Try it | MemoryShards</title>', s, flags=re.S)

# header
header = '''<header class="top">
  <a class="wordmark" href="index.html" aria-label="MemoryShards, home">Memory<em>Shards</em></a>
  <nav class="top-nav" aria-label="Sections">
    <a href="index.html">Overview</a>
    <a href="index.html#pipeline">How it works</a>
    <a href="index.html#timeline">Example</a>
  </nav>
  <div class="byline">AI memory reconstruction</div>
</header>'''
s = re.sub(r'<header class="top">.*?</header>', header, s, count=1, flags=re.S)

# video backdrop out
s = re.sub(r'<div class="page-video" aria-hidden="true">.*?</div>\s*', '', s, count=1, flags=re.S)

# intro link goes back to the MARE overview
s = s.replace('<a href="index.html" class="back-link">&larr; Back to overview</a>', '<a href="index.html" class="back-link">&larr; Back to overview</a>')

# footer
footer = '''<footer class="foot" id="end">
  <div class="foot-top">
    <p>A day is a handful of fragments. This puts them back.</p>
    <nav class="foot-links" aria-label="Footer">
      <a href="index.html">Overview</a>
      <a href="../try.html">View the original design</a>
      <a href="#" id="toTop">Back to top &uarr;</a>
    </nav>
  </div>
  <div class="credits" aria-label="Reference photo sources">
    <span><b>Marine Drive</b> Rajarshi Mitra, CC BY 2.0</span>
    <span><b>Gateway of India</b> Sharique Jamal, CC BY-SA 4.0</span>
    <span><b>Restaurant thali</b> Kanikatwl, CC BY-SA 4.0</span>
    <span><b>Four Seasons Mumbai</b> Creater903a, CC0</span>
    <span>Other photographs: Wikimedia Commons, credits in <a href="../samples/credits.json">samples/credits.json</a></span>
  </div>
</footer>'''
s = re.sub(r'<div class="wrap above-stack">\s*<footer>.*?</footer>\s*</div>', footer, s, count=1, flags=re.S)

# asset paths used by the script
s = s.replace("fetch('media/land.json')", "fetch('../media/land.json')")
s = s.replace("fetch('samples/credits.json')", "fetch('../samples/credits.json')")
s = s.replace("fetch('samples/' + name)", "fetch('../samples/' + name)")

# the shared helper is the MARE copy (no theme toggle); back-to-top works without a smooth-scroll library
s = s.replace('<script src="common.js"></script>', '<script src="common.js"></script>\n<script>document.getElementById("toTop").addEventListener("click", function (e) { e.preventDefault(); scrollTo({ top: 0, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" }); });</script>')

open('web/mare/try.html', 'w', encoding='utf8').write(s)
print('wrote web/mare/try.html', len(s), 'bytes')
for needle in ['../media/land.json', '../samples/credits.json', "'../samples/' + name", 'class="foot"', 'try-skin.css']:
    print(needle, needle in s)
