#include "wproc/street_detect.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace wproc::vision {

namespace {

cv::Mat to_u8(const Image& img) {
    cv::Mat f(img.height(), img.width(), CV_32F,
              const_cast<float*>(img.data()));
    cv::Mat u8;
    f.convertTo(u8, CV_8U);
    return u8;
}

Image from_f32(const cv::Mat& m) {
    CV_Assert(m.type() == CV_32F && m.isContinuous());
    Image out(m.cols, m.rows);
    std::copy_n(m.ptr<float>(), static_cast<std::size_t>(m.cols) * m.rows,
                out.data());
    return out;
}

constexpr double kPi = 3.14159265358979323846;

}  // namespace

double estimate_skew_deg(const Image& img) {
    if (img.width() < 8 || img.height() < 8) return 0.0;
    cv::Mat u8 = to_u8(img), edges;
    cv::Canny(u8, edges, 50, 150);

    std::vector<cv::Vec2f> lines;
    int thresh = std::max(20, img.height() / 3);
    cv::HoughLines(edges, lines, 1.0, kPi / 180.0, thresh);

    // A vertical line has normal angle theta ~ 0 (or pi). Map each line's
    // theta to a signed angle from vertical in (-90, 90]; keep near-vertical
    // ones and take the median (robust to a few spurious lines).
    std::vector<double> angles;
    for (const auto& l : lines) {
        double theta = l[1];
        double a = theta <= kPi / 2.0 ? theta : theta - kPi;  // (-90,90]
        double deg = a * 180.0 / kPi;
        if (std::abs(deg) < 30.0) angles.push_back(deg);
    }
    if (angles.empty()) return 0.0;
    std::sort(angles.begin(), angles.end());
    return angles[angles.size() / 2];
}

Image deskew(const Image& img, double skew_deg) {
    if (std::abs(skew_deg) < 1e-6) return img;
    cv::Mat f(img.height(), img.width(), CV_32F,
              const_cast<float*>(img.data()));
    cv::Point2f center(img.width() / 2.0f, img.height() / 2.0f);
    // Undo the skew: rotate by +skew_deg (OpenCV positive = CCW), which is the
    // inverse of the tilt the estimator reports, so streets stand up straight.
    cv::Mat rot = cv::getRotationMatrix2D(center, skew_deg, 1.0);
    cv::Mat out;
    cv::warpAffine(f, out, rot, f.size(), cv::INTER_LINEAR,
                   cv::BORDER_REPLICATE);
    return from_f32(out);
}

std::vector<double> detect_streets(const Image& img, int min_width_px) {
    std::vector<double> centers;
    const int W = img.width(), H = img.height();
    if (W < 3 || H < 1) return centers;

    // Mean intensity per column; dark streets sit well below the bright
    // substrate. Threshold midway between the darkest and brightest columns.
    std::vector<double> cm(W, 0.0);
    double cmin = 1e300, cmax = -1e300;
    for (int x = 0; x < W; ++x) {
        double s = 0.0;
        for (int y = 0; y < H; ++y) s += img.at(x, y);
        cm[x] = s / H;
        cmin = std::min(cmin, cm[x]);
        cmax = std::max(cmax, cm[x]);
    }
    if (cmax - cmin < 1e-6) return centers;
    double thr = 0.5 * (cmin + cmax);

    // Group consecutive dark columns; centre = intensity-weighted midpoint.
    int run_start = -1;
    auto flush = [&](int lo, int hi) {
        if (hi - lo + 1 < min_width_px) return;
        double num = 0.0, den = 0.0;
        for (int x = lo; x <= hi; ++x) {
            double w = thr - cm[x];  // darker => heavier
            num += w * x;
            den += w;
        }
        centers.push_back(den > 0 ? num / den : 0.5 * (lo + hi));
    };
    for (int x = 0; x < W; ++x) {
        bool dark = cm[x] < thr;
        if (dark && run_start < 0) run_start = x;
        if (!dark && run_start >= 0) {
            flush(run_start, x - 1);
            run_start = -1;
        }
    }
    if (run_start >= 0) flush(run_start, W - 1);
    return centers;
}

}  // namespace wproc::vision
