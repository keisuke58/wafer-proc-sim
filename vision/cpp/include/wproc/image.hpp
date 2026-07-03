// image.hpp — core image containers for the wafer-processing vision toolkit.
//
// Pure C++17, no external dependencies. Two containers:
//   Image       : single-channel float image (working buffer for all algorithms)
//   ColorImage  : 8-bit RGB image (annotated output only)
//
// Row-major storage; pixel (x, y) with x in [0, width), y in [0, height).
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace wproc {

// Single-channel float image. Values are conventionally 0..255 but algorithms
// tolerate arbitrary ranges (e.g. signed gradients).
class Image {
public:
    Image() = default;
    Image(int w, int h, float fill = 0.0f)
        : width_(w), height_(h), data_(check_size(w, h), fill) {}

    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    bool empty() const noexcept { return width_ == 0 || height_ == 0; }
    std::size_t size() const noexcept { return data_.size(); }

    float& at(int x, int y) { return data_[index(x, y)]; }
    float at(int x, int y) const { return data_[index(x, y)]; }

    // Clamped access: coordinates outside the image are clamped to the border.
    // Used by convolution kernels so they need no special edge handling.
    float at_clamped(int x, int y) const noexcept {
        if (x < 0) x = 0; else if (x >= width_) x = width_ - 1;
        if (y < 0) y = 0; else if (y >= height_) y = height_ - 1;
        return data_[static_cast<std::size_t>(y) * width_ + x];
    }

    float* data() noexcept { return data_.data(); }
    const float* data() const noexcept { return data_.data(); }

private:
    static std::size_t check_size(int w, int h) {
        if (w < 0 || h < 0)
            throw std::invalid_argument("Image: negative dimension");
        return static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    }
    std::size_t index(int x, int y) const {
        if (x < 0 || x >= width_ || y < 0 || y >= height_)
            throw std::out_of_range("Image::at: coordinate out of range");
        return static_cast<std::size_t>(y) * width_ + x;
    }

    int width_ = 0;
    int height_ = 0;
    std::vector<float> data_;
};

struct Rgb {
    std::uint8_t r = 0, g = 0, b = 0;
};

// 8-bit RGB image for annotated inspection output.
class ColorImage {
public:
    ColorImage() = default;
    ColorImage(int w, int h)
        : width_(w), height_(h), data_(check_size(w, h) * 3, 0) {}

    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    bool empty() const noexcept { return width_ == 0 || height_ == 0; }

    void set(int x, int y, Rgb c) {
        if (x < 0 || x >= width_ || y < 0 || y >= height_) return;  // clip
        std::size_t i = (static_cast<std::size_t>(y) * width_ + x) * 3;
        data_[i] = c.r;
        data_[i + 1] = c.g;
        data_[i + 2] = c.b;
    }

    std::uint8_t* data() noexcept { return data_.data(); }
    const std::uint8_t* data() const noexcept { return data_.data(); }

private:
    // Validate before the vector is sized: a negative dimension cast to
    // size_t would wrap to a huge allocation instead of throwing cleanly.
    static std::size_t check_size(int w, int h) {
        if (w < 0 || h < 0)
            throw std::invalid_argument("ColorImage: negative dimension");
        return static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    }

    int width_ = 0;
    int height_ = 0;
    std::vector<std::uint8_t> data_;
};

}  // namespace wproc
