// cli_args.hpp — tiny flag-based CLI parsing helpers shared by the wproc apps.
#pragma once

#include <cstdlib>
#include <cstring>

namespace wproc_cli {

inline double arg_double(int argc, char** argv, const char* flag, double def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], flag) == 0) return std::atof(argv[i + 1]);
    return def;
}

inline const char* arg_str(int argc, char** argv, const char* flag,
                           const char* def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return def;
}

inline bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

}  // namespace wproc_cli
