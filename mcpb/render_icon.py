"""mcpb/render_icon.py — render the OPYT tile mark to every raster the repo ships.

Two outputs, one mark: mcpb/icon.png (the Claude Desktop extension icon) and
website/favicon.ico (16/32/48, the browser favicon). Both are checked in; re-run this by
hand when the mark changes.

THE MARK IS NOT DEFINED HERE. Upstream is the live site: the `.tile` SVG in useopyt.com's
markup plus the `.tile-*` rules in its style.css, transcribed once more in
website/favicon.svg, which is what a browser that supports SVG favicons actually renders.
The chevron and the cursor below are a transcription of it:

    <path d="M18 16 L36 32 L18 48" stroke=#fff stroke-width=7/> the chevron: butt caps, miter join
    <rect x=40 y=41 width=11 height=7 fill=#dd3418/>            the block cursor, in the accent
    <rect width=64 height=64 fill=#000000/>                     sharp-cornered terminal tile

The tile background is a deliberate divergence, not a transcription. On the site
`.tile-bg` is `currentColor`, so the tile inverts with the page theme: #141414 on light,
#ececec on dark. A raster cannot invert, so it has to pick one form. It renders the dark
tile at PURE BLACK, darker than the site's #141414, so the mark reads as fully black on the
X avatar and the Claude Desktop extension icon.

If the chevron or the cursor changes, change it in the site's markup, in
website/favicon.svg and here, then re-run. Do not invent one.

Why this is Python and not an SVG the build rasterizes: no SVG rasterizer on this machine draws
a stroked <path> correctly — ImageMagick's internal renderer silently dropped the chevron and
emitted a bare tile with a red block, which is how the wrong icon shipped once already. So the
stroke outline is computed here (offset lines plus the miter intersection) and filled as one
polygon at 8x supersample.

    python mcpb/render_icon.py        # needs Pillow
"""
from __future__ import annotations

import math
import struct
from pathlib import Path

from PIL import Image, ImageDraw

BG, GLYPH, ACCENT = (0, 0, 0, 255), (255, 255, 255, 255), (221, 52, 24, 255)
P0, P1, P2 = (18.0, 16.0), (36.0, 32.0), (18.0, 48.0)     # the chevron's three points
STROKE = 7.0
CURSOR = (40.0, 41.0, 51.0, 48.0)                          # x0, y0, x1, y1
SIZE, SUPERSAMPLE = 512, 8


def _sub(a, b): return (a[0] - b[0], a[1] - b[1])
def _add(a, b): return (a[0] + b[0], a[1] + b[1])
def _mul(a, k): return (a[0] * k, a[1] * k)


def _unit(v):
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n)


def _outward(d):
    """The normal on the chevron's convex side, which faces +x."""
    n = (d[1], -d[0])
    return n if n[0] > 0 else (-n[0], -n[1])


def _intersect(p, d, q, e):
    den = d[0] * e[1] - d[1] * e[0]
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return _add(p, _mul(d, t))


def stroke_outline():
    """The chevron's filled outline: butt caps at both ends, a mitered point at the vertex."""
    h = STROKE / 2.0
    dA, dB = _unit(_sub(P1, P0)), _unit(_sub(P2, P1))
    nA, nB = _outward(dA), _outward(dB)
    iA, iB = _mul(nA, -1), _mul(nB, -1)
    miter = _intersect(_add(P0, _mul(nA, h)), dA, _add(P1, _mul(nB, h)), dB)
    inner = _intersect(_add(P0, _mul(iA, h)), dA, _add(P1, _mul(iB, h)), dB)
    return [_add(P0, _mul(nA, h)), miter, _add(P2, _mul(nB, h)),
            _add(P2, _mul(iB, h)), inner, _add(P0, _mul(iA, h))]


def write_ico(img: "Image.Image", out: Path, sizes=(48, 32, 16)) -> None:
    """Write a multi-size .ico with BMP/DIB frames.

    Pillow's own ICO writer emits PNG-compressed frames with wPlanes=0. That is legal but
    it is the least widely decoded form of the format, and Safari would not display it.
    Every frame here is an uncompressed 32-bit bottom-up DIB with a doubled biHeight and a
    zeroed AND mask, which is what a favicon.ico is in practice.
    """
    frames = []
    for n in sizes:
        px = img.resize((n, n), Image.LANCZOS).convert("RGBA")
        rows = [b"".join(struct.pack("<4B", *(px.getpixel((x, y))[i] for i in (2, 1, 0, 3)))
                         for x in range(n))
                for y in reversed(range(n))]
        bits = b"".join(rows)
        mask_stride = ((n + 31) // 32) * 4                 # 1bpp rows pad to 4 bytes
        mask = b"\x00" * (mask_stride * n)                 # 0 = opaque; the tile has no alpha
        header = struct.pack("<IiiHHIIiiII", 40, n, n * 2, 1, 32, 0, len(bits) + len(mask),
                             0, 0, 0, 0)
        frames.append(header + bits + mask)

    offset = 6 + 16 * len(frames)
    directory = b""
    for n, data in zip(sizes, frames):
        directory += struct.pack("<BBBBHHII", n, n, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    out.write_bytes(struct.pack("<HHH", 0, 1, len(frames)) + directory + b"".join(frames))


def main() -> None:
    k = SIZE * SUPERSAMPLE / 64.0                          # SVG units -> supersampled pixels
    img = Image.new("RGBA", (SIZE * SUPERSAMPLE,) * 2, BG)
    d = ImageDraw.Draw(img)
    d.polygon([(x * k, y * k) for x, y in stroke_outline()], fill=GLYPH)
    d.rectangle([v * k for v in CURSOR], fill=ACCENT)
    repo = Path(__file__).resolve().parent.parent
    png = repo / "mcpb" / "icon.png"
    img.resize((SIZE, SIZE), Image.LANCZOS).save(png)
    # The .ico is downsampled from the same supersampled draw, not from icon.png, so the
    # 16px frame loses as little of the chevron as Lanczos allows.
    ico = repo / "website" / "favicon.ico"
    write_ico(img, ico)
    print(f"wrote {png}\nwrote {ico}")


if __name__ == "__main__":
    main()
