#ifndef PIGMENT_PAINTER_MIXER_HPP
#define PIGMENT_PAINTER_MIXER_HPP

#include <array>
#include <vector>

namespace pigment_painter {

std::array<float, 3> mix_srgb(const std::vector<std::array<float, 3>> &colors,
                              const std::vector<float>                &weights);

std::array<float, 3> mix_srgb(const std::vector<std::array<float, 3>> &colors,
                              const std::vector<int>                  &weights);

} // namespace pigment_painter

#endif
