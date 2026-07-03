// grind_surface.hpp — ground-surface quality inspection.
//
// After grinding, the die back-surface carries a directional grinding-mark
// texture and a shallow subsurface-damage (SSD) layer. Both matter: strong
// marks and a thick SSD layer weaken the die. From a single grayscale surface
// image this recovers, with no external dependencies:
//
//   * grind-mark severity + direction, via the gradient structure tensor
//     (coherent, strongly oriented texture = pronounced marks);
//   * a roughness proxy (Ra) as the mean absolute high-pass residual;
//   * an SSD index — the fraction of image variance carried by high spatial
//     frequency, a stand-in for the depth/energy of near-surface damage.
//
// It also renders a heat map of local high-frequency energy so damage hotspots
// are visible at a glance.
#pragma once

#include "wproc/image.hpp"

namespace wproc::grind {

struct SurfaceQuality {
    double coherence = 0.0;         // structure-tensor coherence, 0..1
    double mark_angle_deg = 0.0;    // dominant grinding-mark direction, [0,180)
    double mark_severity = 0.0;     // coherence * relative gradient energy, 0..1
    double roughness_ra = 0.0;      // mean |high-pass residual| (gray levels)
    double ssd_index = 0.0;         // high-freq variance / total variance, 0..1
    bool ok = true;                 // within the supplied accept thresholds
};

struct SurfaceSpec {
    double max_mark_severity = 0.35;
    double max_roughness_ra = 12.0;
    double max_ssd_index = 0.25;
};

// Analyze a grayscale surface image. `hp_sigma` sets the Gaussian scale whose
// residual defines the high-pass band (roughness / SSD). The verdict `ok` is
// evaluated against `spec`.
SurfaceQuality analyze_surface(const Image& gray, float hp_sigma = 2.0f,
                               const SurfaceSpec& spec = SurfaceSpec{});

// Heat map (black→red→yellow→white) of local high-frequency energy — the SSD
// hotspot view. `hp_sigma` is the high-pass scale, `pool_sigma` the smoothing
// applied to the squared residual to pool local energy.
ColorImage render_ssd_heatmap(const Image& gray, float hp_sigma = 2.0f,
                              float pool_sigma = 4.0f);

}  // namespace wproc::grind
