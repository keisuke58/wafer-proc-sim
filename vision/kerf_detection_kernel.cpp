// kerf_detection_kernel.cpp — C++ pybind11 kernel for blade kerf measurement
//
// Implements Sobel-X edge detection on grayscale float32 images.
// No OpenCV dependency: pure C++14 + NumPy array interface.
// Drop-in for real camera images from DISCO DFL7160 blade camera.
//
// Functions exported:
//   detect_kerf_width(image, threshold, pixel_per_mm, use_zero_crossing) → dict
//   detect_blade_tip(image, threshold)                                    → dict
//   measure_chipping(image, left_px, right_px, ppmm)                      → dict
//
// Detection methods:
//   zero_crossing (default): signed Sobel-X column projection, find + → −
//     transitions. Wall centre = zero-crossing position (unbiased, sub-pixel).
//   peak: |Sobel-X| column projection, leftmost/rightmost local max above
//     threshold. Biased by +2σ_wall; kept for legacy compatibility.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

// ── Sobel-X helpers ──────────────────────────────────────────────────────────

// |Sobel-X| — unsigned magnitude (legacy peak method)
static std::vector<float> sobel_x(const float* img, int H, int W) {
    std::vector<float> out(H * W, 0.0f);
    for (int r = 1; r < H - 1; ++r)
        for (int c = 1; c < W - 1; ++c) {
            float gx = -img[(r-1)*W+(c-1)] - 2.0f*img[r*W+(c-1)] - img[(r+1)*W+(c-1)]
                       +img[(r-1)*W+(c+1)] + 2.0f*img[r*W+(c+1)] + img[(r+1)*W+(c+1)];
            out[r*W+c] = std::abs(gx);
        }
    return out;
}

// Signed Sobel-X — preserves gradient direction (zero-crossing method)
// At a bright wall: positive on the left flank, negative on the right flank.
// The + → − zero-crossing is exactly at the wall centre, independent of σ.
static std::vector<float> sobel_x_signed(const float* img, int H, int W) {
    std::vector<float> out(H * W, 0.0f);
    for (int r = 1; r < H - 1; ++r)
        for (int c = 1; c < W - 1; ++c) {
            float gx = -img[(r-1)*W+(c-1)] - 2.0f*img[r*W+(c-1)] - img[(r+1)*W+(c-1)]
                       +img[(r-1)*W+(c+1)] + 2.0f*img[r*W+(c+1)] + img[(r+1)*W+(c+1)];
            out[r*W+c] = gx;   // no abs
        }
    return out;
}

// Column projection (sum over rows); works for both signed and unsigned
static std::vector<float> col_proj(const std::vector<float>& v, int H, int W) {
    std::vector<float> proj(W, 0.0f);
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            proj[c] += v[r * W + c];
    return proj;
}

// 3-tap box smoother (reduces per-column sensor noise before zero-crossing)
static std::vector<float> smooth3(const std::vector<float>& p) {
    int W = static_cast<int>(p.size());
    std::vector<float> s(p);
    for (int c = 1; c < W - 1; ++c)
        s[c] = (p[c-1] + p[c] + p[c+1]) / 3.0f;
    return s;
}

// ── Wall-finding: zero-crossing of signed Sobel projection ──────────────────
// A bright kerf wall produces a signed projection that rises then falls.
// The + → − zero-crossing is the wall centre (independent of wall σ).
//
// Algorithm: segment-qualified sign change
//   1. Track the running max of each positive run.
//   2. At each + → ≤0 transition, accept it as a wall crossing if:
//        (a) the preceding positive run had max > adaptive threshold  AND
//        (b) the suffix minimum after the crossing is < −adaptive threshold
//      This handles both sharp walls (zero-crossing within 1 px) and worn walls
//      (gradual transition over many px), without requiring both sides of the
//      threshold to appear in the same column.
//   3. Sub-pixel accuracy via linear interpolation at the sign-change column.
//
// Adaptive threshold: thr = max(|proj|) × min_signal_frac
//   Wall peak  ≈ wall_brightness × H ≈ 50 000 for H=128, brightness=220.
//   Noise peak ≈ noise_σ × sqrt(H) × Sobel_gain ≈ 200–2 000.
//   At min_signal_frac = 0.10, thr ≈ 5 000 → kills noise, keeps walls.
static std::pair<float,float> find_walls_zero_crossing(
    const std::vector<float>& proj_signed,
    float min_signal_frac
) {
    int W = static_cast<int>(proj_signed.size());
    auto s = smooth3(proj_signed);

    // Adaptive threshold
    float max_abs = 0.0f;
    for (float v : s) max_abs = std::max(max_abs, std::abs(v));
    float thr = max_abs * min_signal_frac;
    if (thr < 1.0f) return std::make_pair(-1.0f, -1.0f);

    // Suffix minimum for O(1) lookahead
    std::vector<float> suf_min(W + 1, 0.0f);
    for (int c = W - 1; c >= 0; --c)
        suf_min[c] = std::min(suf_min[c + 1], s[c]);

    // Sweep: record + → ≤0 transitions qualified by amplitude
    std::vector<float> zc;
    float seg_max = 0.0f;
    for (int c = 0; c < W; ++c) {
        if (s[c] > 0.0f) {
            seg_max = std::max(seg_max, s[c]);
        } else {
            if (c > 0 && s[c-1] > 0.0f) {
                // + → ≤0 sign change detected
                if (seg_max > thr && suf_min[c] < -thr) {
                    float frac = s[c-1] / (s[c-1] - s[c] + 1e-9f);
                    zc.push_back(static_cast<float>(c - 1) + frac);
                }
                seg_max = 0.0f;
            }
        }
    }

    if (zc.size() >= 2) return std::make_pair(zc.front(), zc.back());
    if (zc.size() == 1) return std::make_pair(zc[0], -1.0f);
    return std::make_pair(-1.0f, -1.0f);
}

// ── Wall-finding: peak method (legacy) ──────────────────────────────────────
// Leftmost/rightmost local max of |Sobel-X| projection above threshold.
// Biased by +2σ_wall; threshold must be tuned per image.
static std::pair<int,int> find_kerf_edges(const std::vector<float>& proj, float thr) {
    int W = static_cast<int>(proj.size());
    int left = -1, right = -1;
    for (int c = 1; c < W - 1; ++c) {
        if (proj[c] > thr && proj[c] >= proj[c-1] && proj[c] >= proj[c+1]) {
            if (left < 0) left = c;
            right = c;
        }
    }
    if (left < 0 || left == right) return std::make_pair(-1, -1);
    return std::make_pair(left, right);
}

// ── Public API ──────────────────────────────────────────────────────────────

py::dict detect_kerf_width(
    py::array_t<float> image,
    float threshold,        // used only by peak method
    float pixel_per_mm,
    bool  use_zero_crossing // true = signed ZC (default); false = unsigned peak (legacy)
) {
    auto buf = image.request();
    if (buf.ndim != 2)
        throw std::runtime_error("detect_kerf_width: image must be 2-D H×W");
    int H = static_cast<int>(buf.shape[0]);
    int W = static_cast<int>(buf.shape[1]);
    const float* ptr = static_cast<const float*>(buf.ptr);

    float left_subpx, right_subpx;
    std::string method_used;

    if (use_zero_crossing) {
        // Signed Sobel-X projection → + → − zero-crossings = wall centres
        // Adaptive threshold: 6 % of max |proj|.
        // Wall peak ≈ 50 000 (substrate=60, wall=160, H=128, sharp blade).
        // Substrate noise floor ≈ 200–2 000.  At 6 % → thr ≈ 3 000: well above
        // noise but low enough to catch narrow (< 1 mm) or worn-blade walls where
        // overlapping Gaussian profiles reduce the peak signed projection.
        const float min_signal_frac = 0.06f;
        auto sx  = sobel_x_signed(ptr, H, W);
        auto prj = col_proj(sx, H, W);
        auto walls = find_walls_zero_crossing(prj, min_signal_frac);
        left_subpx  = walls.first;
        right_subpx = walls.second;
        method_used = "zero_crossing";
    } else {
        // Legacy: unsigned |Sobel-X| peak search
        auto edge  = sobel_x(ptr, H, W);
        auto proj  = col_proj(edge, H, W);
        auto edges = find_kerf_edges(proj, threshold);
        left_subpx  = static_cast<float>(edges.first);
        right_subpx = static_cast<float>(edges.second);
        method_used = "peak";
    }

    py::dict res;
    if (left_subpx < 0.0f || right_subpx < 0.0f) {
        res["kerf_width_mm"]  = -1.0f;
        res["left_edge_px"]   = -1;
        res["right_edge_px"]  = -1;
        res["left_wall_subpx"]  = -1.0f;
        res["right_wall_subpx"] = -1.0f;
        res["confidence"]     = 0.0f;
        res["method"]         = method_used;
        res["status"]         = std::string("no_kerf_found");
        return res;
    }

    float width_mm = (right_subpx - left_subpx) / pixel_per_mm;
    // Confidence: how symmetric the two wall signals are (heuristic)
    float conf = std::max(0.0f, 1.0f - std::abs(right_subpx - left_subpx -
                          (right_subpx - left_subpx)) / (right_subpx - left_subpx + 1e-6f));
    conf = std::min(1.0f, (right_subpx - left_subpx) / (static_cast<float>(W) * 0.05f));
    res["kerf_width_mm"]    = width_mm;
    res["left_edge_px"]     = static_cast<int>(std::round(left_subpx));
    res["right_edge_px"]    = static_cast<int>(std::round(right_subpx));
    res["left_wall_subpx"]  = left_subpx;
    res["right_wall_subpx"] = right_subpx;
    res["confidence"]       = conf;
    res["method"]           = method_used;
    res["status"]           = std::string("ok");
    return res;
}

py::dict detect_blade_tip(py::array_t<float> image, float threshold) {
    auto buf = image.request();
    if (buf.ndim != 2)
        throw std::runtime_error("detect_blade_tip: image must be 2-D H×W");
    int H = static_cast<int>(buf.shape[0]);
    int W = static_cast<int>(buf.shape[1]);
    const float* ptr = static_cast<const float*>(buf.ptr);

    int tip_row = -1;
    for (int r = 0; r < H && tip_row < 0; ++r)
        for (int c = 1; c < W - 1; ++c)
            if (ptr[r*W+c] > threshold) { tip_row = r; break; }

    py::dict res;
    if (tip_row < 0) {
        res["tip_x"] = -1.0f; res["tip_y"] = -1.0f;
        res["confidence"] = 0.0f; res["status"] = std::string("no_tip_found");
        return res;
    }
    float sum_c = 0.0f, sum_w = 0.0f;
    for (int c = 0; c < W; ++c) {
        float v = ptr[tip_row * W + c];
        if (v > threshold) { sum_c += v * c; sum_w += v; }
    }
    float tip_x = sum_w > 0.0f ? sum_c / sum_w : 0.0f;
    res["tip_x"]      = tip_x;
    res["tip_y"]      = static_cast<float>(tip_row);
    res["confidence"] = std::min(1.0f, sum_w / (threshold * W));
    res["status"]     = std::string("ok");
    return res;
}

py::dict measure_chipping(
    py::array_t<float> image,
    float kerf_left_px,
    float kerf_right_px,
    float pixel_per_mm
) {
    auto buf = image.request();
    if (buf.ndim != 2)
        throw std::runtime_error("measure_chipping: image must be 2-D H×W");
    int H = static_cast<int>(buf.shape[0]);
    int W = static_cast<int>(buf.shape[1]);
    const float* ptr = static_cast<const float*>(buf.ptr);

    int left  = std::max(0, static_cast<int>(kerf_left_px  - 5.0f));
    int right = std::min(W-1, static_cast<int>(kerf_right_px + 5.0f));

    int chip = 0, total = 0;
    for (int r = 0; r < H; ++r) {
        for (int c = left; c <= right; ++c) {
            if (static_cast<float>(c) < kerf_left_px ||
                static_cast<float>(c) > kerf_right_px) {
                ++total;
                if (ptr[r*W+c] < 50.0f) ++chip;
            }
        }
    }
    float ratio    = total > 0 ? static_cast<float>(chip) / total : 0.0f;
    float chip_um  = ratio * 5.0f / pixel_per_mm * 1000.0f;

    py::dict res;
    res["chipping_um"]    = chip_um;
    res["chipping_ratio"] = ratio;
    res["status"]         = std::string("ok");
    return res;
}

PYBIND11_MODULE(_kerf_detection_kernel, m) {
    m.doc() = "Kerf width + blade tip detection kernels (pure C++14, no OpenCV)";

    m.def("detect_kerf_width", &detect_kerf_width,
          py::arg("image"),
          py::arg("threshold") = 1000.0f,
          py::arg("pixel_per_mm") = 10.0f,
          py::arg("use_zero_crossing") = true,
          "Kerf wall detection via Sobel-X.\n"
          "\n"
          "use_zero_crossing=True  (default, recommended):\n"
          "  Signed Sobel-X column projection; wall centre = + → − zero-crossing.\n"
          "  Sub-pixel accurate; no threshold tuning; unaffected by blade wear σ.\n"
          "\n"
          "use_zero_crossing=False  (legacy peak method):\n"
          "  |Sobel-X| column projection; leftmost/rightmost peak above threshold.\n"
          "  Biased by +2σ_wall; threshold must be hand-tuned per image.\n"
          "\n"
          "Returns: kerf_width_mm, left_edge_px, right_edge_px,\n"
          "         left_wall_subpx, right_wall_subpx, confidence, method, status.");

    m.def("detect_blade_tip", &detect_blade_tip,
          py::arg("image"), py::arg("threshold") = 200.0f,
          "Sub-pixel blade tip detection via brightness centroid. "
          "Returns: tip_x, tip_y, confidence, status.");

    m.def("measure_chipping", &measure_chipping,
          py::arg("image"), py::arg("kerf_left_px"), py::arg("kerf_right_px"),
          py::arg("pixel_per_mm") = 10.0f,
          "Estimate chipping area (dark pixels in 5-px border outside kerf walls). "
          "Returns: chipping_um, chipping_ratio, status.");
}
