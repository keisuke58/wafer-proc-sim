#include "wproc/io.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace wproc {

namespace {

// Read the next whitespace-separated token from a Netpbm header, skipping
// '#' comment lines as required by the format.
std::string next_token(std::istream& in) {
    std::string tok;
    while (in >> tok) {
        if (!tok.empty() && tok[0] == '#') {
            std::string discard;
            std::getline(in, discard);  // skip rest of comment line
            continue;
        }
        return tok;
    }
    throw std::runtime_error("read_pgm: unexpected end of header");
}

int to_int(const std::string& s, const char* what) {
    try {
        return std::stoi(s);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("read_pgm: invalid ") + what);
    }
}

std::uint8_t clamp_u8(float v) {
    float r = std::lround(v);
    if (r < 0.0f) return 0;
    if (r > 255.0f) return 255;
    return static_cast<std::uint8_t>(r);
}

}  // namespace

Image read_pgm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("read_pgm: cannot open " + path);

    std::string magic = next_token(in);
    if (magic != "P2" && magic != "P5")
        throw std::runtime_error("read_pgm: unsupported magic '" + magic +
                                 "' (only P2/P5)");

    int w = to_int(next_token(in), "width");
    int h = to_int(next_token(in), "height");
    int maxval = to_int(next_token(in), "maxval");
    if (w <= 0 || h <= 0)
        throw std::runtime_error("read_pgm: non-positive dimensions");
    if (maxval <= 0 || maxval > 255)
        throw std::runtime_error("read_pgm: only 8-bit images (maxval 1..255)");

    Image img(w, h);
    const std::size_t n = static_cast<std::size_t>(w) * h;

    if (magic == "P2") {
        for (std::size_t i = 0; i < n; ++i) {
            std::string t = next_token(in);
            img.data()[i] = static_cast<float>(to_int(t, "pixel"));
        }
    } else {  // P5: single whitespace byte then raw bytes
        in.get();  // consume the single separator after maxval
        std::vector<char> buf(n);
        in.read(buf.data(), static_cast<std::streamsize>(n));
        if (static_cast<std::size_t>(in.gcount()) != n)
            throw std::runtime_error("read_pgm: truncated pixel data");
        for (std::size_t i = 0; i < n; ++i)
            img.data()[i] = static_cast<float>(static_cast<std::uint8_t>(buf[i]));
    }
    return img;
}

void write_pgm(const std::string& path, const Image& img) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("write_pgm: cannot open " + path);
    out << "P5\n" << img.width() << ' ' << img.height() << "\n255\n";
    const std::size_t n = img.size();
    std::vector<char> buf(n);
    for (std::size_t i = 0; i < n; ++i)
        buf[i] = static_cast<char>(clamp_u8(img.data()[i]));
    out.write(buf.data(), static_cast<std::streamsize>(n));
    if (!out) throw std::runtime_error("write_pgm: write failed for " + path);
}

void write_ppm(const std::string& path, const ColorImage& img) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("write_ppm: cannot open " + path);
    out << "P6\n" << img.width() << ' ' << img.height() << "\n255\n";
    const std::size_t n = static_cast<std::size_t>(img.width()) * img.height() * 3;
    out.write(reinterpret_cast<const char*>(img.data()),
              static_cast<std::streamsize>(n));
    if (!out) throw std::runtime_error("write_ppm: write failed for " + path);
}

}  // namespace wproc
