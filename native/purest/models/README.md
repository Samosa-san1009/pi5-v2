# OpenCV Haar cascades

Bundled CPU-only face and eye classifiers from OpenCV's `data/haarcascades`:

- `haarcascade_frontalface_default.xml`
- `haarcascade_eye_tree_eyeglasses.xml`

Upstream: https://github.com/opencv/opencv/tree/4.x/data/haarcascades
Copied from the installed OpenCV Python 4.14.0 data distribution. Each XML
retains its copyright and redistribution license at the start of the file.
`SHA256SUMS` identifies the exact bundled files; the rig needs no model download.
These cascades classify face/eye windows; they do not segment pupils.
