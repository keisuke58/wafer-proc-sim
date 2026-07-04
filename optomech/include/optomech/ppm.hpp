// ppm.hpp — minimal 8-bit RGB image + binary PPM writer for demo figures.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace optm {

struct Rgb { std::uint8_t r = 0, g = 0, b = 0; };

class Canvas {
public:
    Canvas(int w, int h, Rgb fill = {255, 255, 255});
    int width() const { return w_; }
    int height() const { return h_; }
    void set(int x, int y, Rgb c);
    void line(int x0, int y0, int x1, int y1, Rgb c);
    void rect(int x0, int y0, int x1, int y1, Rgb c);
    void disk(int cx, int cy, int r, Rgb c);
    const std::uint8_t* data() const { return d_.data(); }

private:
    int w_, h_;
    std::vector<std::uint8_t> d_;
};

bool write_ppm(const std::string& path, const Canvas& c);

}  // namespace optm
