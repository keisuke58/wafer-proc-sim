// NSGA-II non-dominated sort + crowding distance kernel (pybind11).
// Mirrors fast_nondominated_sort() + crowding_distance() in
// optimization/multiobjective_pipeline.py.
//
// WHY C++: fast_nondominated_sort has an O(n² × n_obj) nested Python loop that
// runs every generation of the optimizer. At pop_size=60, n_gen=30, n_obj=6 that
// is 60² × 30 = 108,000 Python-dispatched dominance comparisons per run — the
// dominant runtime cost. The C++ kernel does those comparisons at native speed
// with contiguous memory access, no Python-object overhead.
//
// crowding_distance's inner sort+sweep is also replaced to eliminate the Python
// loop over objectives and front members.
//
// Build: bash build_all_kernels.sh

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>

namespace py = pybind11;
using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

// f_a dominates f_b: f_a[k] <= f_b[k] for all k, strict for at least one k.
static inline bool dominates(const double* a, const double* b, int m) {
    bool any_less = false;
    for (int k = 0; k < m; ++k) {
        if (a[k] > b[k]) return false;
        if (a[k] < b[k]) any_less = true;
    }
    return any_less;
}

// Fast non-dominated sort.
// Returns a Python list of fronts; each front is a list of row indices into F.
static py::list nondominated_sort(Arr F_arr) {
    if (F_arr.ndim() != 2)
        throw std::runtime_error("nondominated_sort: F must be 2-D (n_pop × n_obj)");

    const int  n   = (int)F_arr.shape(0);
    const int  m   = (int)F_arr.shape(1);
    const double* data = F_arr.data();

    std::vector<int>              dom_count(n, 0);
    std::vector<std::vector<int>> dominated_by(n);

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (dominates(data + i * m, data + j * m, m)) {
                dominated_by[i].push_back(j);
                dom_count[j]++;
            } else if (dominates(data + j * m, data + i * m, m)) {
                dominated_by[j].push_back(i);
                dom_count[i]++;
            }
        }
    }

    std::vector<std::vector<int>> fronts(1);
    for (int i = 0; i < n; ++i)
        if (dom_count[i] == 0) fronts[0].push_back(i);

    int k = 0;
    while (!fronts[k].empty()) {
        std::vector<int> next;
        for (int i : fronts[k])
            for (int j : dominated_by[i])
                if (--dom_count[j] == 0) next.push_back(j);
        fronts.push_back(std::move(next));
        k++;
    }
    fronts.pop_back(); // trailing empty front

    py::list result;
    for (auto& fr : fronts) {
        py::list front_list;
        for (int idx : fr) front_list.append(idx);
        result.append(front_list);
    }
    return result;
}

// Crowding distance for a single front (list of indices into F).
// Returns a 1-D double array of length len(front).
static py::array_t<double> crowding_distance(Arr F_arr, py::list front_py) {
    if (F_arr.ndim() != 2)
        throw std::runtime_error("crowding_distance: F must be 2-D");

    const auto F = F_arr.unchecked<2>();
    const int  m = (int)F_arr.shape(1);

    std::vector<int> front;
    front.reserve(front_py.size());
    for (auto h : front_py) front.push_back(py::cast<int>(h));
    const int l = (int)front.size();

    auto dist  = py::array_t<double>(l);
    auto bdist = dist.mutable_unchecked<1>();
    for (int i = 0; i < l; ++i) bdist(i) = 0.0;

    if (l <= 2) {
        for (int i = 0; i < l; ++i)
            bdist(i) = std::numeric_limits<double>::infinity();
        return dist;
    }

    // Build local objective matrix for front members.
    std::vector<double> Ff(l * m);
    for (int i = 0; i < l; ++i)
        for (int k = 0; k < m; ++k)
            Ff[i * m + k] = F(front[i], k);

    std::vector<int> idx(l);
    for (int k = 0; k < m; ++k) {
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b){ return Ff[a*m+k] < Ff[b*m+k]; });

        const double fmin = Ff[idx[0] * m + k];
        const double fmax = Ff[idx[l-1] * m + k];
        bdist(idx[0])   = std::numeric_limits<double>::infinity();
        bdist(idx[l-1]) = std::numeric_limits<double>::infinity();
        if (fmax - fmin < 1e-15) continue;
        const double span = fmax - fmin;
        for (int i = 1; i < l - 1; ++i)
            bdist(idx[i]) += (Ff[idx[i+1]*m+k] - Ff[idx[i-1]*m+k]) / span;
    }
    return dist;
}

PYBIND11_MODULE(_nsga2_kernel, m) {
    m.doc() = "C++ NSGA-II non-dominated sort + crowding distance kernel (pybind11)";
    m.def("nondominated_sort", &nondominated_sort,
          "Fast non-dominated sort of F (n_pop × n_obj) -> list of fronts "
          "(each front is a list of row indices)",
          py::arg("F"));
    m.def("crowding_distance", &crowding_distance,
          "Crowding distance for one front -> 1-D float array of length len(front)",
          py::arg("F"), py::arg("front"));
}
