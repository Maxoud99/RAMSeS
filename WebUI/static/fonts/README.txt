Vendored webfonts
=================

The UI has no runtime network dependency, so the two faces it uses are served
from this directory: IBM Plex Sans for every word of the interface, the
headings and the narratives, and IBM Plex Mono for detector names, numbers in
tables, the command preview, the run log and the report. They are Carbon's own
type families. Both are licensed under the SIL Open Font License 1.1; the
licence text, with IBM's copyright notice for each family, is OFL.txt.

Source
------
Unmodified Fontsource files, downloaded on 2026-10-21 from jsDelivr:

  https://cdn.jsdelivr.net/npm/@fontsource/ibm-plex-sans@5.3.0/files/<file>
  https://cdn.jsdelivr.net/npm/@fontsource/ibm-plex-mono@5.3.0/files/<file>

Fontsource repackages the upstream IBM Plex releases without changing the
outlines and splits each weight into unicode-range subsets. The unicode-range
values in ramses.css (section 1) are copied from the packages' own 400.css,
so a browser fetches a subset only for a page that uses one of its characters.

Files
-----
  ibm-plex-sans-latin-400-normal.woff2       body text, labels, headings 03/04
  ibm-plex-sans-latin-600-normal.woff2       headings 01/02, emphasis
  ibm-plex-sans-latin-ext-400-normal.woff2   Latin Extended (Ŝ, ŝ, the combining
  ibm-plex-sans-latin-ext-600-normal.woff2     tilde) in the documentation
  ibm-plex-sans-greek-400-normal.woff2       μ, σ, Σ, τ, θ in the documentation
  ibm-plex-sans-greek-600-normal.woff2         and the narratives
  ibm-plex-mono-latin-400-normal.woff2       detector names, numbers, code
  ibm-plex-mono-latin-600-normal.woff2       family names in the detector pool
  ibm-plex-mono-latin-ext-400-normal.woff2   Latin Extended in code
  ibm-plex-mono-latin-ext-600-normal.woff2

Only weights 400 and 600 are used (Carbon's productive type set). There is no
italic: nothing in the interface is set in one. Plex Mono has no Greek subset
upstream, and neither family draws the arrows, ✓/✕, the maths operators
(≠, ∈, ‖) or the modifier letters (ᵀ, ₛ); those few characters fall back to
the platform's own face. The interface draws its own state marks as inline
SVG icons (ramses.css section 2) rather than as characters, so the fallback
only touches running text.

Checksums (sha256 of the files as shipped, identical to the jsDelivr copies)
---------------------------------------------------------------------------
08949f728dc52d528e69b1667d15c89a5686a4ee9a296ff90983985f99c380f7  ibm-plex-mono-latin-400-normal.woff2
0d1f0b8d0722224e32e9f28261bdc86c79115be73444ae5eceb73976a1bcdf83  ibm-plex-mono-latin-600-normal.woff2
6bc0f226a5b7884a8170e3f62c63d7675609d4631bdc5931b5cdab81821f00eb  ibm-plex-mono-latin-ext-400-normal.woff2
32057cf50dd14bdb21a2c93766c4a2c43e4abe688ea3922df3203cac7751a98b  ibm-plex-mono-latin-ext-600-normal.woff2
02296067bc904d2ad19b49445bdaf6586cc44704e668e16e248f708e5005a290  ibm-plex-sans-greek-400-normal.woff2
0a9ebd9019415ee9f161b38fb24b5ee7736e2c8b85bf1e79e1e4c7c0010d55cb  ibm-plex-sans-greek-600-normal.woff2
3b646991d30055a93a4ecc499713d4347953a74a947ecab435ab72070cbdab0e  ibm-plex-sans-latin-400-normal.woff2
8960851d691c054ed38e259bdcf1a6190d157b4203ed5bb32c632a863fb8ec2f  ibm-plex-sans-latin-600-normal.woff2
c93d2a12aaa280f68b9ab7b726ff8dfedda67c99ef9abed047c1847a1cc6d583  ibm-plex-sans-latin-ext-400-normal.woff2
b25dfd4f979e442ae1e25cd0894463434cf01ba21ac1a35d39f4a82bd4cc060e  ibm-plex-sans-latin-ext-600-normal.woff2

Check, from this directory:  grep -E '^[0-9a-f]{64}  ' README.txt | sha256sum -c
