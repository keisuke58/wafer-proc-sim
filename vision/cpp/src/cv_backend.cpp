#include "wproc/cv_backend.hpp"

#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace wproc::cv_backend {

namespace {

cv::Mat to_mat(const Image& img) {
    // Wrap without copy (CV_32F, row-major float matches Image layout),
    // then clone so OpenCV owns its buffer independently of `img`.
    cv::Mat view(img.height(), img.width(), CV_32F,
                 const_cast<float*>(img.data()));
    return view.clone();
}

Image to_image(const cv::Mat& m) {
    CV_Assert(m.type() == CV_32F && m.isContinuous());
    Image out(m.cols, m.rows);
    std::copy_n(m.ptr<float>(), static_cast<std::size_t>(m.cols) * m.rows,
                out.data());
    return out;
}

}  // namespace

Image gaussian_blur(const Image& src, float sigma) {
    // Mirror the core backend's contract: reject non-positive, NaN, and
    // oversized sigma before the float->int radius conversion (UB otherwise).
    if (sigma <= 0.0f)
        throw std::invalid_argument("cv gaussian_blur: sigma must be > 0");
    if (!(sigma <= 1000.0f))
        throw std::invalid_argument("cv gaussian_blur: sigma too large (max 1000)");
    const int radius = static_cast<int>(std::ceil(3.0f * sigma));
    const int ksize = 2 * radius + 1;
    cv::Mat in = to_mat(src), out;
    cv::GaussianBlur(in, out, cv::Size(ksize, ksize), sigma, sigma,
                     cv::BORDER_REPLICATE);
    return to_image(out);
}

Gradient sobel(const Image& src) {
    cv::Mat in = to_mat(src), gx, gy;
    cv::Sobel(in, gx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    cv::Sobel(in, gy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
    return Gradient{to_image(gx), to_image(gy)};
}

float otsu_threshold(const Image& src) {
    // OpenCV's Otsu requires 8-bit input; clamp/round the float image the same
    // way the from-scratch histogram does.
    cv::Mat in = to_mat(src), u8, discard;
    in.convertTo(u8, CV_8U);  // saturating round, matches bin_of() clamping
    double thr = cv::threshold(u8, discard, 0, 255,
                               cv::THRESH_BINARY | cv::THRESH_OTSU);
    return static_cast<float>(thr);
}

}  // namespace wproc::cv_backend
