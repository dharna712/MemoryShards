"""Regression test: a crafted upload filename must not write outside the per-request temp directory.
Run: python src/test_filename_traversal.py"""
import io, os, sys, tempfile
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api

client = api.app.test_client()
tmp = tempfile.gettempdir()
two_up = os.path.dirname(os.path.dirname(os.path.join(tmp, "x")))
cases = [("../../ms_traversal_marker.txt", os.path.join(two_up, "ms_traversal_marker.txt")),
         (os.path.join(tmp, "ms_abs_marker.txt"), os.path.join(tmp, "ms_abs_marker.txt")),
         ("..\..\ms_traversal_marker2.txt", os.path.join(two_up, "ms_traversal_marker2.txt"))]
bad = []
for name, target in cases:
    if os.path.exists(target):
        os.remove(target)
    r = client.post("/api/timeline", data={"photos": (io.BytesIO(b"harmless marker"), name)}, content_type="multipart/form-data")
    if os.path.exists(target):
        bad.append(name); os.remove(target)
    print(f"{'FAIL' if os.path.exists(target) or name in bad else 'PASS'}  filename {name!r}: HTTP {r.status_code}")
sys.exit(1 if bad else 0)
