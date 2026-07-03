// test_street.cpp — skew estimation, deskew, and multi-street detection.
// Built with the OpenCV backend (uses warpAffine to synthesize a tilted input).
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "wproc/filters.hpp"
#include "wproc/image.hpp"
#include "wproc/street_detect.hpp"

using namespace wproc;

static int g_total = 0, g_failed = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        ++g_total;                                                      \
        if (!(cond)) {                                                  \
            ++g_failed;                                                 \
            std::cerr << "FAIL [" << __LINE__ << "]: " #cond "\n";      \
        }                                                               \
    } while (0)
#define CHECK_NEAR(a, b, tol)                                                  \
    do {                                                                       \
        ++g_total;                                                             \
        double _d = std::fabs(double(a) - double(b));                          \
        if (_d > (tol)) {                                                      \
            ++g_failed;                                                        \
            std::cerr << "FAIL [" << __LINE__ << "]: |" << (a) << " - " << (b) \
                      << "| = " << _d << " > " << (tol) << "\n";               \
        }                                                                      \
    } while (0)

// Three vertical dark streets on a bright substrate.
static Image make_streets() {
    Image img(300, 200, 170.0f);
    const int cx[3] = {60, 150, 240};
    for (int y = 0; y < 200; ++y)
        for (int s = 0; s < 3; ++s)
            for (int x = cx[s] - 7; x <= cx[s] + 7; ++x) img.at(x, y) = 40.0f;
    return gaussian_blur(img, 1.0f);
}

// Rotate an Image clockwise by deg (for building a tilted test input).
static Image rotate_cw(const Image& img, double deg) {
    cv::Mat f(img.height(), img.width(), CV_32F,
              const_cast<float*>(img.data()));
    cv::Point2f c(img.width() / 2.0f, img.height() / 2.0f);
    cv::Mat rot = cv::getRotationMatrix2D(c, deg, 1.0), out;
    cv::warpAffine(f, out, rot, f.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    Image r(img.width(), img.height());
    std::copy_n(out.ptr<float>(),
                static_cast<std::size_t>(img.width()) * img.height(), r.data());
    return r;
}

int main() {
    Image base = make_streets();

    // Aligned image: ~0 skew, three streets recovered near 60/150/240.
    CHECK_NEAR(vision::estimate_skew_deg(base), 0.0, 1.0);
    auto c0 = vision::detect_streets(base);
    CHECK(c0.size() == 3);
    if (c0.size() == 3) {
        CHECK_NEAR(c0[0], 60.0, 2.0);
        CHECK_NEAR(c0[1], 150.0, 2.0);
        CHECK_NEAR(c0[2], 240.0, 2.0);
    }

    // Tilt by 7 degrees; the estimator recovers the magnitude, and deskewing
    // by the estimate straightens the image (skew round-trips to ~0).
    Image tilted = rotate_cw(base, 7.0);
    double skew = vision::estimate_skew_deg(tilted);
    std::cout << "estimated skew = " << skew << " deg\n";
    CHECK_NEAR(std::abs(skew), 7.0, 1.5);

    Image fixed = vision::deskew(tilted, skew);
    CHECK_NEAR(vision::estimate_skew_deg(fixed), 0.0, 1.5);  // straightened
    auto c1 = vision::detect_streets(fixed);
    std::cout << "streets after deskew: " << c1.size() << "\n";
    CHECK(c1.size() == 3);
    if (c1.size() == 3) {
        CHECK_NEAR(c1[0], 60.0, 5.0);
        CHECK_NEAR(c1[2], 240.0, 5.0);
    }

    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "STREET DETECT OK\n";
    return 0;
}
