// kerf_inspect_main.cpp — CLI front-end for dicing-kerf inspection.
//
// Reads a grayscale PGM scribe-line image, runs the inspection pipeline, prints
// a JSON report to stdout, and optionally writes an annotated PPM overlay.
//
// Usage:
//   kerf_inspect <in.pgm> [--um-per-px U] [--nominal-um N] [--tol-um T]
//                [--max-chip-um C] [--blur S] [--annotate out.ppm] [--quiet]
#include <iostream>
#include <string>

#include "cli_args.hpp"
#include "wproc/io.hpp"
#include "wproc/kerf_inspect.hpp"

using namespace wproc;
using wproc_cli::arg_double;
using wproc_cli::arg_str;
using wproc_cli::has_flag;

int main(int argc, char** argv) {
    if (argc < 2 || argv[1][0] == '-') {
        std::cerr << "usage: kerf_inspect <in.pgm> [--um-per-px U] "
                     "[--nominal-um N] [--tol-um T] [--max-chip-um C] "
                     "[--blur S] [--annotate out.ppm] [--quiet]\n";
        return 2;
    }
    const std::string in_path = argv[1];

    KerfInspector::Params p;
    p.um_per_px = arg_double(argc, argv, "--um-per-px", p.um_per_px);
    p.nominal_width_um = arg_double(argc, argv, "--nominal-um", p.nominal_width_um);
    p.width_tol_um = arg_double(argc, argv, "--tol-um", p.width_tol_um);
    p.max_allowed_chip_um = arg_double(argc, argv, "--max-chip-um", p.max_allowed_chip_um);
    p.blur_sigma = static_cast<float>(arg_double(argc, argv, "--blur", p.blur_sigma));
    const char* annotate_path = arg_str(argc, argv, "--annotate", nullptr);
    const bool quiet = has_flag(argc, argv, "--quiet");

    try {
        Image img = read_pgm(in_path);
        KerfInspector inspector(p);
        KerfResult r = inspector.inspect(img);

        if (!quiet) std::cout << to_json(r);

        if (annotate_path) {
            ColorImage overlay = inspector.annotate(img, r);
            write_ppm(annotate_path, overlay);
            if (!quiet) std::cerr << "annotated overlay -> " << annotate_path << "\n";
        }
        // Exit code encodes the verdict: 0 = pass, 1 = fail/not found.
        return r.pass ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 3;
    }
}
