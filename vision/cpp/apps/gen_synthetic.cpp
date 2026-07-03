// gen_synthetic.cpp — generate a synthetic wafer dicing-kerf image (PGM).
//
// Produces a repeatable test image so the toolkit runs end-to-end with no
// external data: a dark vertical kerf channel on a bright silicon substrate,
// with a couple of chipping defects on the walls and Gaussian sensor noise.
//
// Usage:
//   gen_synthetic <out.pgm> [--width-um W] [--chip-um D] [--noise S] [--seed N]
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>

#include "wproc/filters.hpp"
#include "wproc/image.hpp"
#include "wproc/io.hpp"

using namespace wproc;

namespace {

double arg_double(int argc, char** argv, const char* flag, double def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], flag) == 0) return std::atof(argv[i + 1]);
    return def;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: gen_synthetic <out.pgm> [--width-um W] "
                     "[--chip-um D] [--noise S] [--seed N]\n";
        return 2;
    }
    const std::string out_path = argv[1];
    const double um_per_px = 0.5;
    const double width_um = arg_double(argc, argv, "--width-um", 40.0);
    const double chip_um = arg_double(argc, argv, "--chip-um", 6.0);
    const float noise_sigma = static_cast<float>(arg_double(argc, argv, "--noise", 6.0));
    const unsigned seed =
        static_cast<unsigned>(arg_double(argc, argv, "--seed", 12345));

    const int W = 400, H = 200;
    const float substrate = 150.0f, channel = 40.0f;

    const double half_px = (width_um / um_per_px) / 2.0;
    const double cx = W / 2.0;
    const double left = cx - half_px;
    const double right = cx + half_px;

    Image img(W, H, substrate);

    // Dark kerf channel.
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (x >= left && x <= right) img.at(x, y) = channel;

    // Chipping defects: dark protrusions from the walls into the substrate.
    const int chip_px = static_cast<int>((chip_um / um_per_px) + 0.5);
    auto add_chip = [&](int wall_x, int dir, int y0, int y1, int depth) {
        for (int y = y0; y <= y1 && y < H; ++y)
            for (int d = 0; d <= depth; ++d) {
                int x = wall_x + dir * d;
                if (x >= 0 && x < W) img.at(x, y) = channel;
            }
    };
    // Left wall chip (into the die, i.e. to the left of the left wall).
    add_chip(static_cast<int>(left), -1, 70, 95, chip_px);
    // Right wall chip, shallower.
    add_chip(static_cast<int>(right), +1, 120, 140,
             std::max(1, (chip_px * 3) / 5));

    // Soften walls to a realistic few-pixel transition.
    img = wproc::gaussian_blur(img, 1.0f);

    // Gaussian sensor noise.
    std::mt19937 gen(seed);
    std::normal_distribution<float> noise(0.0f, noise_sigma);
    for (std::size_t i = 0; i < img.size(); ++i) img.data()[i] += noise(gen);

    wproc::write_pgm(out_path, img);
    std::cout << "wrote " << out_path << " (" << W << "x" << H
              << ", kerf~" << width_um << "um, chip~" << chip_um << "um)\n";
    return 0;
}
