#include "TriangleSetSampling.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace Slic3r {

TriangleSetSamples sample_its_uniform_parallel(size_t samples_count, const indexed_triangle_set &triangle_set) {
    TriangleSetSamples result;
    result.total_area = 0.f;

    if (samples_count == 0 || triangle_set.indices.empty() || triangle_set.vertices.empty())
        return result;

    const std::vector<stl_triangle_vertex_indices> &indices  = triangle_set.indices;
    const std::vector<stl_vertex>                  &vertices = triangle_set.vertices;
    const size_t                                    vertex_count = vertices.size();

    std::vector<double> cumulative_area;
    std::vector<size_t> valid_triangle_indices;
    cumulative_area.reserve(indices.size());
    valid_triangle_indices.reserve(indices.size());

    double total_area = 0.0;
    for (size_t t_idx = 0; t_idx < indices.size(); ++t_idx) {
        const stl_triangle_vertex_indices &tri = indices[t_idx];
        if (tri.x() < 0 || tri.y() < 0 || tri.z() < 0)
            continue;

        const size_t ia = size_t(tri.x());
        const size_t ib = size_t(tri.y());
        const size_t ic = size_t(tri.z());
        if (ia >= vertex_count || ib >= vertex_count || ic >= vertex_count)
            continue;

        const Vec3f &a = vertices[ia];
        const Vec3f &b = vertices[ib];
        const Vec3f &c = vertices[ic];
        if (!std::isfinite(a.x()) || !std::isfinite(a.y()) || !std::isfinite(a.z()) ||
            !std::isfinite(b.x()) || !std::isfinite(b.y()) || !std::isfinite(b.z()) ||
            !std::isfinite(c.x()) || !std::isfinite(c.y()) || !std::isfinite(c.z()))
            continue;

        const double area = double(0.5f * (b - a).cross(c - a).norm());
        if (!std::isfinite(area) || area <= 0.0)
            continue;

        total_area += area;
        cumulative_area.emplace_back(total_area);
        valid_triangle_indices.emplace_back(t_idx);
    }

    if (valid_triangle_indices.empty() || total_area <= 0.0 || !std::isfinite(total_area))
        return result;

    std::mt19937_64 mersenne_engine { 27644437 };
    // random numbers on interval [0, 1)
    std::uniform_real_distribution<double> fdistribution;

    auto get_random = [&fdistribution, &mersenne_engine]() {
        return Vec3d { fdistribution(mersenne_engine), fdistribution(mersenne_engine), fdistribution(mersenne_engine) };
    };

    std::vector<Vec3d> random_samples(samples_count);
    std::generate(random_samples.begin(), random_samples.end(), get_random);

    result.total_area = float(total_area);
    result.positions.resize(samples_count);
    result.normals.resize(samples_count);
    result.triangle_indices.resize(samples_count);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, samples_count),
            [&indices, &vertices, &cumulative_area, &valid_triangle_indices, &total_area, &random_samples, &result](
                    tbb::blocked_range<size_t> r) {
                for (size_t s_idx = r.begin(); s_idx < r.end(); ++s_idx) {
                    const double t_sample = random_samples[s_idx].x() * total_area;
                    const auto   it = std::upper_bound(cumulative_area.begin(), cumulative_area.end(), t_sample);
                    const size_t sampled_idx =
                        (it == cumulative_area.end()) ? (cumulative_area.size() - 1) : size_t(std::distance(cumulative_area.begin(), it));
                    const size_t t_idx = valid_triangle_indices[sampled_idx];

                    const double sq_u = std::sqrt(random_samples[s_idx].y());
                    const double v = random_samples[s_idx].z();

                    const stl_triangle_vertex_indices &tri = indices[t_idx];
                    const Vec3f &A = vertices[size_t(tri.x())];
                    const Vec3f &B = vertices[size_t(tri.y())];
                    const Vec3f &C = vertices[size_t(tri.z())];

                    result.positions[s_idx] = A * (1 - sq_u) + B * (sq_u * (1 - v)) + C * (v * sq_u);
                    Vec3f normal = (B - A).cross(C - B);
                    const float normal_len = normal.norm();
                    if (normal_len > 0.f && std::isfinite(normal_len))
                        normal /= normal_len;
                    else
                        normal = Vec3f(0.f, 0.f, 1.f);
                    result.normals[s_idx] = normal;
                    result.triangle_indices[s_idx] = t_idx;
                }
            });

    return result;
}

}
