// blade_wear_kernel.cpp — Physics-based blade wear & optimal replacement
//
// AP試験 / DISCO engineer concepts:
//   - Archard 摩耗則: V_wear = k × F_N × L / H  (摩耗体積 ∝ 荷重×滑り距離/硬度)
//   - Taylor 工具寿命: V_cut × T^n = C  (切削速度×寿命^n = 定数)
//   - コスト最適化: 交換コスト + ダウンタイムコスト → 最適ウェーハ枚数/ブレード
//   - 素材依存性: SiC は Si の 2× 摩耗速度 → 消耗品単価が上がる根拠
//
// DISCO context:
//   - ブレードは DISCO 売上の核（消耗品ビジネス）
//   - 摩耗予測 → 交換タイミング最適化 → 稼働率向上 + ダイシングコスト削減
//   - DFL7160 標準ブレード: φ54 mm, #2000 ダイヤ砥粒, CBN バインダ
//   - SiC EV 向け: 加工硬度 25 GPa vs Si 13 GPa → ブレード寿命 40-60% 短縮

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace py = pybind11;

// ── Material database ─────────────────────────────────────────────────────────
struct Material {
    std::string name;
    double hardness_GPa;   // Vickers hardness (converted to GPa)
    double K_IC;           // Fracture toughness (MPa·√m)
    double E_GPa;          // Young's modulus (GPa)
    double archard_k;      // Archard wear coefficient (dimensionless, ~1e-5 to 1e-3)
};

static const Material MATERIALS[] = {
    {"Si",      13.0,  0.70, 130.0, 8.0e-5},
    {"SiC",     28.0,  2.80, 450.0, 3.5e-5},  // harder → more blade wear despite lower k
    {"GaN",     15.0,  0.60, 295.0, 1.1e-4},
    {"Sapphire",23.0,  2.10, 400.0, 4.0e-5},
    {"Diamond", 78.0,  5.00, 1050.0,1.0e-6},
};
static const int N_MATERIALS = 5;

static const Material& find_material(const std::string& name) {
    for (int i = 0; i < N_MATERIALS; ++i)
        if (MATERIALS[i].name == name) return MATERIALS[i];
    return MATERIALS[0];   // default Si
}

// ── Archard wear law ─────────────────────────────────────────────────────────
// Worn volume per unit sliding distance: dV/dL = k × F_N / H
// V_wear [mm³] = k × F_N [N] × L [mm] / H [MPa]
static double archard_wear_volume_mm3(
    double k_wear,           // dimensionless wear coefficient
    double F_N_newtons,      // normal force [N]
    double slide_dist_mm,    // total sliding distance [mm]
    double hardness_GPa      // workpiece hardness [GPa]
) {
    double H_MPa = hardness_GPa * 1000.0;   // GPa → MPa
    return k_wear * F_N_newtons * slide_dist_mm / H_MPa;
}

// Normal force estimate from dicing parameters (empirical model)
// F_N ≈ k_f × depth [µm] × feed [mm/s] × hardness_scaling
static double estimate_normal_force(
    double depth_um,          // blade bite depth [µm]
    double feed_mm_s,         // feed rate [mm/s]
    double hardness_GPa
) {
    // Empirical: F ∝ depth × feed × sqrt(H/H_Si)
    double k_f = 0.12;   // N/(µm·mm/s)
    return k_f * depth_um * feed_mm_s * std::sqrt(hardness_GPa / 13.0);
}

// ── Taylor's tool life equation ───────────────────────────────────────────────
// V × T^n = C  →  T = (C/V)^(1/n)
// For dicing blades: V ≡ peripheral speed (m/min), T in minutes
// n ≈ 0.25-0.45 (resin-bond diamond on hard ceramics)
// C = constant determined by blade-material system
static double taylor_tool_life_min(
    double V_m_min,          // cutting (peripheral) speed [m/min]
    double n_taylor,         // Taylor exponent
    double C_taylor          // Taylor constant
) {
    if (V_m_min <= 0.0 || n_taylor <= 0.0) return 0.0;
    return std::pow(C_taylor / V_m_min, 1.0 / n_taylor);
}

// ── Per-wafer wear and blade lifetime ────────────────────────────────────────

py::dict archard_analysis(
    const std::string& material_name,
    double depth_um,           // cut depth [µm]
    double feed_mm_s,          // feed rate [mm/s]
    double spindle_krpm,       // spindle speed [krpm]
    double blade_diam_mm,      // blade diameter [mm]
    double wafer_diam_mm,      // wafer diameter [mm]
    double blade_max_wear_mm3  // allowable wear volume before replacement [mm³]
) {
    const Material& m = find_material(material_name);

    // Peripheral speed: V = π × D × n
    double V_m_min = M_PI * blade_diam_mm * 1e-3 * spindle_krpm * 1000.0;  // m/min

    // Normal force per cut
    double F_N = estimate_normal_force(depth_um, feed_mm_s, m.hardness_GPa);

    // Sliding distance per wafer: assume one cut across full wafer diameter
    // Multiple cuts: n_cuts ≈ wafer_diam / street_pitch; use representative 1-cut model
    double L_per_wafer_mm = wafer_diam_mm;   // single-cut approximation

    // Wear per wafer
    double V_per_wafer = archard_wear_volume_mm3(m.archard_k, F_N,
                                                  L_per_wafer_mm, m.hardness_GPa);

    // Wafers per blade
    double wafers_per_blade = (V_per_wafer > 0.0) ?
        blade_max_wear_mm3 / V_per_wafer : 1e9;

    py::dict r;
    r["material"]             = material_name;
    r["hardness_GPa"]         = m.hardness_GPa;
    r["K_IC_MPa_sqrt_m"]      = m.K_IC;
    r["E_GPa"]                = m.E_GPa;
    r["archard_k"]            = m.archard_k;
    r["F_N_newtons"]          = F_N;
    r["peripheral_speed_m_min"] = V_m_min;
    r["wear_per_wafer_mm3"]   = V_per_wafer;
    r["wafers_per_blade"]     = static_cast<int>(wafers_per_blade);
    r["blade_max_wear_mm3"]   = blade_max_wear_mm3;
    r["wear_mechanism"]       = std::string(
        m.hardness_GPa > 20.0 ? "abrasive_dominant" : "adhesive_dominant");
    return r;
}

// ── Taylor analysis ───────────────────────────────────────────────────────────

py::dict taylor_analysis(
    const std::string& material_name,
    double spindle_krpm,
    double blade_diam_mm,
    double n_taylor,
    double C_taylor
) {
    double V_m_min = M_PI * blade_diam_mm * 1e-3 * spindle_krpm * 1000.0;
    double T_min = taylor_tool_life_min(V_m_min, n_taylor, C_taylor);

    // Convert minutes → approximate wafers (assume ~10 s per wafer)
    double wafers_est = T_min * 60.0 / 10.0;

    py::dict r;
    r["material"]              = material_name;
    r["peripheral_speed_m_min"]= V_m_min;
    r["n_taylor"]              = n_taylor;
    r["C_taylor"]              = C_taylor;
    r["tool_life_min"]         = T_min;
    r["tool_life_wafers_est"]  = static_cast<int>(wafers_est);
    r["note"] = std::string(
        "Taylor T^n=C: lower spindle speed extends life but reduces throughput.");
    return r;
}

// ── Optimal replacement cost model ───────────────────────────────────────────
// Total cost per wafer = (blade_cost + downtime_cost) / wafers_per_blade
// Blade wears probabilistically: model as Weibull with given mean
// Optimal changeout minimizes expected cost considering early failure penalty
py::dict optimize_replacement(
    double wafers_per_blade_mean,  // expected blade life [wafers]
    double blade_cost_usd,         // blade purchase cost [USD]
    double downtime_cost_per_changeout_usd,  // planned changeout cost [USD]
    double unplanned_penalty_usd,  // extra cost if blade breaks mid-lot [USD]
    double weibull_beta            // Weibull shape (2.0 = wear-out)
) {
    // Planned-replacement cost at T wafers:
    //   C(T) = (blade_cost + downtime_cost) / T
    //        + unplanned_penalty × P(failure before T) / T  (simplified)
    // P(failure < T | Weibull(β, η)) where η = mean / Γ(1+1/β)
    // Approximate: minimize over integer T

    // Gamma(1 + 1/β) approximation (Stirling)
    double inv_beta = 1.0 / weibull_beta;
    double gamma_val = std::exp(std::lgamma(1.0 + inv_beta));
    double eta = wafers_per_blade_mean / gamma_val;   // Weibull scale

    double best_T = wafers_per_blade_mean;
    double best_cost = 1e18;
    std::vector<double> costs;

    int T_max = static_cast<int>(wafers_per_blade_mean * 2.0);
    for (int T = 1; T <= T_max; ++T) {
        // Weibull CDF: F(T) = 1 - exp(-(T/η)^β)
        double F_T = 1.0 - std::exp(-std::pow(T / eta, weibull_beta));
        // Expected cost per wafer
        double cost_planned   = (blade_cost_usd + downtime_cost_per_changeout_usd)
                                 / static_cast<double>(T);
        double cost_unplanned = unplanned_penalty_usd * F_T / static_cast<double>(T);
        double total = cost_planned + cost_unplanned;
        costs.push_back(total);
        if (total < best_cost) { best_cost = total; best_T = T; }
    }

    // Sensitivity: 10% more wafers
    int T_10pct = static_cast<int>(best_T * 1.10);
    double cost_10pct_over = (T_10pct <= T_max && T_10pct > 0) ?
        costs[T_10pct - 1] : best_cost;

    py::dict r;
    r["optimal_wafers_per_blade"] = static_cast<int>(best_T);
    r["cost_per_wafer_usd"]       = best_cost;
    r["blade_cost_usd"]           = blade_cost_usd;
    r["downtime_cost_usd"]        = downtime_cost_per_changeout_usd;
    r["unplanned_penalty_usd"]    = unplanned_penalty_usd;
    r["weibull_eta"]              = eta;
    r["weibull_beta"]             = weibull_beta;
    r["cost_if_10pct_overrun_usd"]= cost_10pct_over;
    r["overrun_cost_increase_pct"]= 100.0 * (cost_10pct_over - best_cost) / best_cost;
    return r;
}

// ── Full demo: Si vs SiC comparison ──────────────────────────────────────────
py::list blade_wear_demo() {
    struct Scenario {
        std::string material;
        double feed_mm_s;
        double depth_um;
        double spindle_krpm;
    };

    // Representative DISCO DFL7160 parameters
    std::vector<Scenario> scenarios = {
        {"Si",  10.0, 50.0, 30.0},   // standard Si: fast feed, moderate depth
        {"SiC",  3.0, 30.0, 25.0},   // SiC: slower feed, shallower to limit chipping
        {"GaN",  6.0, 40.0, 28.0},   // GaN: intermediate
        {"Sapphire", 2.0, 20.0, 20.0},
    };

    py::list results;
    for (const auto& sc : scenarios) {
        auto archard = archard_analysis(sc.material, sc.depth_um, sc.feed_mm_s,
                                         sc.spindle_krpm, 54.0, 200.0, 0.05);
        auto taylor  = taylor_analysis(sc.material, sc.spindle_krpm, 54.0,
                                        0.35, 120.0);
        auto opt     = optimize_replacement(
            py::cast<double>(archard["wafers_per_blade"]),
            85.0,   // blade cost USD
            200.0,  // planned changeout cost (30-min downtime)
            500.0,  // unplanned break penalty
            2.0     // Weibull β = wear-out
        );

        py::dict row;
        row["material"]              = sc.material;
        row["feed_mm_s"]             = sc.feed_mm_s;
        row["depth_um"]              = sc.depth_um;
        row["hardness_GPa"]          = py::cast<double>(archard["hardness_GPa"]);
        row["wear_per_wafer_mm3"]    = py::cast<double>(archard["wear_per_wafer_mm3"]);
        row["wafers_per_blade"]      = py::cast<int>(archard["wafers_per_blade"]);
        row["optimal_changeout"]     = py::cast<int>(opt["optimal_wafers_per_blade"]);
        row["cost_per_wafer_usd"]    = py::cast<double>(opt["cost_per_wafer_usd"]);
        row["tool_life_min"]         = py::cast<double>(taylor["tool_life_min"]);
        results.append(row);
    }
    return results;
}

// ── pybind11 module ──────────────────────────────────────────────────────────
PYBIND11_MODULE(_blade_wear_kernel, m) {
    m.doc() =
        "Physics-based blade wear model for DISCO precision dicing.\n"
        "\n"
        "AP試験 / DISCO engineer topics:\n"
        "  - Archard摩耗則: V = k·F·L/H  (摩耗体積 = 摩耗係数×荷重×距離/硬度)\n"
        "  - Taylor工具寿命方程式: V·T^n = C\n"
        "  - 素材別摩耗比較: SiC(28GPa) vs Si(13GPa) vs GaN(15GPa)\n"
        "  - コスト最適化: 計画交換 vs 予期しない破損のトレードオフ\n"
        "\n"
        "DISCO context:\n"
        "  - 消耗品(ブレード)は売上の約30%。寿命予測→交換最適化→稼働率向上。\n"
        "  - SiC(EV向け)はSiの2倍以上摩耗 → 消耗品単価高いが需要拡大中。";

    m.def("archard_analysis", &archard_analysis,
          py::arg("material")        = "Si",
          py::arg("depth_um")        = 50.0,
          py::arg("feed_mm_s")       = 10.0,
          py::arg("spindle_krpm")    = 30.0,
          py::arg("blade_diam_mm")   = 54.0,
          py::arg("wafer_diam_mm")   = 200.0,
          py::arg("blade_max_wear_mm3") = 0.05,
          "Archard wear analysis: V=k·F·L/H per wafer → wafers_per_blade.\n"
          "Materials: Si, SiC, GaN, Sapphire, Diamond.");

    m.def("taylor_analysis", &taylor_analysis,
          py::arg("material")     = "Si",
          py::arg("spindle_krpm") = 30.0,
          py::arg("blade_diam_mm")= 54.0,
          py::arg("n_taylor")     = 0.35,
          py::arg("C_taylor")     = 120.0,
          "Taylor tool life: V·T^n=C → tool life in minutes and estimated wafers.");

    m.def("optimize_replacement", &optimize_replacement,
          py::arg("wafers_per_blade_mean"),
          py::arg("blade_cost_usd")                 = 85.0,
          py::arg("downtime_cost_per_changeout_usd")= 200.0,
          py::arg("unplanned_penalty_usd")           = 500.0,
          py::arg("weibull_beta")                    = 2.0,
          "Cost-optimal blade replacement interval (planned vs unplanned tradeoff).\n"
          "Returns: optimal_wafers_per_blade, cost_per_wafer_usd, overrun sensitivity.");

    m.def("blade_wear_demo", &blade_wear_demo,
          "Full demo: Si / SiC / GaN / Sapphire comparison — Archard + Taylor + cost.");
}
