#include "Clipper2Utils.hpp"
#include "ClipperUtils.hpp"
#include "ShortestPath.hpp"
#include "libslic3r.h"
#include "clipper2/clipper.h"

#include <algorithm>
#include <iterator>

namespace Slic3r {

Clipper2Lib::Point64 Slic3rPoint_to_Point64(const Slic3r::Point& in)
{
    return Clipper2Lib::Point64(in.x(), in.y());
}

//BBS: FIXME
Slic3r::Polylines Paths64_to_polylines(const Clipper2Lib::Paths64& in)
{
    Slic3r::Polylines out;
    out.reserve(in.size());
    for (const Clipper2Lib::Path64& path64 : in) {
        Slic3r::Points points;
        points.reserve(path64.size());
        for (const Clipper2Lib::Point64& point64 : path64)
            points.emplace_back(std::move(Slic3r::Point(point64.x, point64.y)));
        out.emplace_back(std::move(Slic3r::Polyline(points)));
    }
    return out;
}

//BBS: FIXME
template <typename Container>
Clipper2Lib::Paths64 Slic3rPoints_to_Paths64(const Container& in)
{
    Clipper2Lib::Paths64 out;
    out.reserve(in.size());
    for (const auto& item : in) {
        Clipper2Lib::Path64 path;
        path.reserve(item.size());
        for (const Slic3r::Point& point : item.points)
            path.emplace_back(std::move(Clipper2Lib::Point64(point.x(), point.y())));
        out.emplace_back(std::move(path));
    }
    return out;
}

Clipper2Lib::Path64 Slic3rPolygon_to_Path64(const Slic3r::Polygon& in)
{
    Clipper2Lib::Path64 path;
    path.reserve(in.points.size());
    for (const Slic3r::Point& point : in.points)
        path.emplace_back(Slic3rPoint_to_Point64(point));
    return path;
}

Clipper2Lib::Paths64 Slic3rPolygon_to_Paths64(const Slic3r::Polygon& in)
{
    Clipper2Lib::Paths64 out;
    out.reserve(1);
    Clipper2Lib::Path64 path = Slic3rPolygon_to_Path64(in);
    out.emplace_back(std::move(path));
    return out;
}

Clipper2Lib::Paths64 Slic3rPolylines_to_Paths64(const Slic3r::Polylines& in)
{
    return Slic3rPoints_to_Paths64(in);
}

Slic3r::Points Path64_to_points(const Clipper2Lib::Path64& path64)
{
    Slic3r::Points points;
    points.reserve(path64.size());
    for (const Clipper2Lib::Point64 &point64 : path64)
        points.emplace_back(point64.x, point64.y);
    return points;
}

Slic3r::Polygons Paths64_to_polygons(const Clipper2Lib::Paths64& in)
{
    Slic3r::Polygons out;
    out.reserve(in.size());
    for (const Clipper2Lib::Path64& path64 : in)
        out.emplace_back(Path64_to_points(path64));
    return out;
}

static void PolyTree64_append_expolygon(const Clipper2Lib::PolyPath64& polynode, Slic3r::ExPolygons& expolygons)
{
    if (!polynode.Polygon().empty()) {
        Slic3r::ExPolygon expolygon;
        expolygon.contour.points = Path64_to_points(polynode.Polygon());
        expolygon.holes.reserve(polynode.Count());
        for (size_t i = 0; i < polynode.Count(); ++i) {
            const Clipper2Lib::PolyPath64* child = polynode.Child(i);
            if (!child->Polygon().empty())
                expolygon.holes.emplace_back(Path64_to_points(child->Polygon()));
        }
        expolygons.emplace_back(std::move(expolygon));
    }
    for (size_t i = 0; i < polynode.Count(); ++i) {
        const Clipper2Lib::PolyPath64* child = polynode.Child(i);
        for (size_t j = 0; j < child->Count(); ++j)
            PolyTree64_append_expolygon(*child->Child(j), expolygons);
    }
}

static size_t PolyTree64_count_expolygons(const Clipper2Lib::PolyPath64& polynode)
{
    size_t count = polynode.Polygon().empty() ? 0 : 1;
    for (size_t i = 0; i < polynode.Count(); ++i) {
        const Clipper2Lib::PolyPath64* child = polynode.Child(i);
        for (size_t j = 0; j < child->Count(); ++j)
            count += PolyTree64_count_expolygons(*child->Child(j));
    }
    return count;
}

Slic3r::ExPolygons PolyTree64_to_expolygons(const Clipper2Lib::PolyTree64& polytree)
{
    Slic3r::ExPolygons retval;
    size_t cnt = 0;
    for (size_t i = 0; i < polytree.Count(); ++i)
        cnt += PolyTree64_count_expolygons(*polytree.Child(i));
    retval.reserve(cnt);
    for (size_t i = 0; i < polytree.Count(); ++i)
        PolyTree64_append_expolygon(*polytree.Child(i), retval);
    return retval;
}

Slic3r::ExPolygons PolyTree64_to_expolygons(Clipper2Lib::PolyTree64&& polytree)
{
    return PolyTree64_to_expolygons(static_cast<const Clipper2Lib::PolyTree64&>(polytree));
}

static Slic3r::ExPolygons PolyTreeToExPolygons(Clipper2Lib::PolyTree64&& polytree)
{
    return PolyTree64_to_expolygons(std::move(polytree));
}

void SimplifyPolyTree(const Clipper2Lib::PolyPath64 &polytree, double epsilon, Clipper2Lib::PolyPath64 &result)
{
    for (const auto &child : polytree) {
        Clipper2Lib::PolyPath64 *newchild = result.AddChild(Clipper2Lib::SimplifyPath(child->Polygon(), epsilon));
        SimplifyPolyTree(*child, epsilon, *newchild);
    }
}

Clipper2Lib::Paths64 Slic3rPolygons_to_Paths64(const Slic3r::Polygons &in)
{
    Clipper2Lib::Paths64 out;
    out.reserve(in.size());
    for (const Slic3r::Polygon &poly : in)
        out.emplace_back(Slic3rPolygon_to_Path64(poly));
    return out;
}

Clipper2Lib::Paths64 Slic3rPolygons_to_Paths64(const Slic3r::Polygons &in, bool filter_degenerate)
{
    if (!filter_degenerate)
        return Slic3rPolygons_to_Paths64(in);
    Clipper2Lib::Paths64 out;
    out.reserve(in.size());
    for (const Slic3r::Polygon &poly : in)
        if (poly.points.size() >= 3)
            out.emplace_back(Slic3rPolygon_to_Path64(poly));
    return out;
}

Clipper2Lib::Paths64 Slic3rExPolygon_to_Paths64(const Slic3r::ExPolygon& in)
{
    Clipper2Lib::Paths64 out;
    out.reserve(in.num_contours());
    out.emplace_back(Slic3rPolygon_to_Path64(in.contour));
    for (const Slic3r::Polygon& hole : in.holes)
        out.emplace_back(Slic3rPolygon_to_Path64(hole));
    return out;
}

Clipper2Lib::Paths64 Slic3rExPolygon_to_Paths64(const Slic3r::ExPolygon& in, bool filter_degenerate)
{
    if (!filter_degenerate)
        return Slic3rExPolygon_to_Paths64(in);
    Clipper2Lib::Paths64 out;
    out.reserve(in.num_contours());
    if (in.contour.points.size() >= 3)
        out.emplace_back(Slic3rPolygon_to_Path64(in.contour));
    for (const Slic3r::Polygon& hole : in.holes)
        if (hole.points.size() >= 3)
            out.emplace_back(Slic3rPolygon_to_Path64(hole));
    return out;
}

Clipper2Lib::Paths64 Slic3rExPolygons_to_Paths64(const Slic3r::ExPolygons& in)
{
    Clipper2Lib::Paths64 out;
    out.reserve(number_polygons(in));
    for (const Slic3r::ExPolygon& expolygon : in) {
        for (size_t i = 0; i < expolygon.num_contours(); i++) {
            const auto         &poly = expolygon.contour_or_hole(i);
            out.emplace_back(Slic3rPolygon_to_Path64(poly));
        }
    }
    return out;
}

Clipper2Lib::Paths64 Slic3rExPolygons_to_Paths64(const Slic3r::ExPolygons& in, bool filter_degenerate)
{
    if (!filter_degenerate)
        return Slic3rExPolygons_to_Paths64(in);
    Clipper2Lib::Paths64 out;
    out.reserve(number_polygons(in));
    for (const Slic3r::ExPolygon& expolygon : in) {
        Clipper2Lib::Paths64 expolygon_paths = Slic3rExPolygon_to_Paths64(expolygon, true);
        out.insert(out.end(), expolygon_paths.begin(), expolygon_paths.end());
    }
    return out;
}

Slic3r::Polylines _clipper2_pl_open(Clipper2Lib::ClipType clipType, const Slic3r::Polylines& subject, const Slic3r::Polygons& clip)
{
    Clipper2Lib::Clipper64 c;
    c.AddOpenSubject(Slic3rPoints_to_Paths64(subject));
    c.AddClip(Slic3rPoints_to_Paths64(clip));

    Clipper2Lib::ClipType ct = clipType;
    Clipper2Lib::FillRule fr = Clipper2Lib::FillRule::NonZero;
    Clipper2Lib::Paths64 solution, solution_open;
    c.Execute(ct, fr, solution, solution_open);

    Slic3r::Polylines out;
    out.reserve(solution.size() + solution_open.size());
    polylines_append(out, std::move(Paths64_to_polylines(solution)));
    polylines_append(out, std::move(Paths64_to_polylines(solution_open)));

    return out;
}

Slic3r::Polylines intersection_pl_2(const Slic3r::Polylines& subject, const Slic3r::Polygons& clip)
    { return _clipper2_pl_open(Clipper2Lib::ClipType::Intersection, subject, clip); }
Slic3r::Polylines  diff_pl_2(const Slic3r::Polylines& subject, const Slic3r::Polygons& clip)
    { return _clipper2_pl_open(Clipper2Lib::ClipType::Difference, subject, clip); }

Clipper2Lib::JoinType clipper2_join_type(ClipperLib::JoinType join_type)
{
    switch (join_type) {
    case ClipperLib::jtSquare: return Clipper2Lib::JoinType::Square;
    case ClipperLib::jtRound: return Clipper2Lib::JoinType::Round;
    case ClipperLib::jtMiter: return Clipper2Lib::JoinType::Miter;
    default: return Clipper2Lib::JoinType::Miter;
    }
}

void configure_clipper2_offsetter(Clipper2Lib::ClipperOffset& offsetter, ClipperLib::JoinType join_type, double miter_limit)
{
    if (join_type == ClipperLib::jtRound)
        offsetter.ArcTolerance(miter_limit);
    else
        offsetter.MiterLimit(miter_limit);
}

Clipper2Lib::Paths64 offset_paths_2(const Clipper2Lib::Paths64& paths, double delta, ClipperLib::JoinType join_type, double miter_limit)
{
    if (paths.empty() || delta == 0.)
        return paths;
    Clipper2Lib::ClipperOffset offsetter;
    configure_clipper2_offsetter(offsetter, join_type, miter_limit);
    offsetter.AddPaths(paths, clipper2_join_type(join_type), Clipper2Lib::EndType::Polygon);
    Clipper2Lib::Paths64 out;
    offsetter.Execute(delta, out);
    return out;
}

Slic3r::ExPolygons boolean_ex_2(Clipper2Lib::ClipType clip_type,
                        const Clipper2Lib::Paths64& subject,
                        Clipper2Lib::Paths64 clip,
                        ApplySafetyOffset do_safety_offset)
{
    if (subject.empty() || (clip.empty() && clip_type == Clipper2Lib::ClipType::Intersection))
        return {};
    if (do_safety_offset == ApplySafetyOffset::Yes)
        clip = offset_paths_2(clip, ClipperSafetyOffset, DefaultJoinType, DefaultMiterLimit);
    Clipper2Lib::Clipper64 clipper;
    clipper.AddSubject(subject);
    if (!clip.empty())
        clipper.AddClip(clip);
    Clipper2Lib::PolyTree64 solution;
    clipper.Execute(clip_type, Clipper2Lib::FillRule::NonZero, solution);
    return PolyTree64_to_expolygons(solution);
}

static Clipper2Lib::Paths64 clipper2_safety_offset(const Clipper2Lib::Paths64& paths)
{
    Clipper2Lib::Paths64 out;
    out.reserve(paths.size());
    for (const Clipper2Lib::Path64& path : paths) {
        Clipper2Lib::ClipperOffset offsetter;
        offsetter.MiterLimit(DefaultMiterLimit);
        offsetter.AddPath(path, Clipper2Lib::JoinType::Miter, Clipper2Lib::EndType::Polygon);
        Clipper2Lib::Paths64 out_this;
        const bool ccw = Clipper2Lib::Area(path) > 0.;
        offsetter.Execute(ccw ? ClipperSafetyOffset : -ClipperSafetyOffset, out_this);
        if (!ccw)
            for (Clipper2Lib::Path64& out_path : out_this)
                std::reverse(out_path.begin(), out_path.end());
        out.insert(out.end(), std::make_move_iterator(out_this.begin()), std::make_move_iterator(out_this.end()));
    }
    return out;
}

static Clipper2Lib::Paths64 clipper2_do_paths(Clipper2Lib::ClipType clip_type,
                                              const Clipper2Lib::Paths64& subject,
                                              Clipper2Lib::Paths64 clip,
                                              ApplySafetyOffset do_safety_offset)
{
    if (subject.empty() || (clip.empty() && clip_type == Clipper2Lib::ClipType::Intersection))
        return {};
    if (do_safety_offset == ApplySafetyOffset::Yes)
        clip = clipper2_safety_offset(clip);
    Clipper2Lib::Clipper64 c;
    c.AddSubject(subject);
    if (!clip.empty())
        c.AddClip(clip);
    Clipper2Lib::Paths64 solution;
    c.Execute(clip_type, Clipper2Lib::FillRule::NonZero, solution);
    return solution;
}

static Slic3r::ExPolygons clipper2_do_expolygons(Clipper2Lib::ClipType clip_type,
                                         const Clipper2Lib::Paths64& subject,
                                         Clipper2Lib::Paths64 clip,
                                         ApplySafetyOffset do_safety_offset)
{
    if (subject.empty() || (clip.empty() && clip_type == Clipper2Lib::ClipType::Intersection))
        return {};
    if (do_safety_offset == ApplySafetyOffset::Yes)
        clip = clipper2_safety_offset(clip);
    Clipper2Lib::Clipper64 c;
    c.AddSubject(subject);
    if (!clip.empty())
        c.AddClip(clip);
    Clipper2Lib::PolyTree64 solution;
    c.Execute(clip_type, Clipper2Lib::FillRule::NonZero, solution);
    return PolyTreeToExPolygons(std::move(solution));
}

static Slic3r::Polygons _clipper2(Clipper2Lib::ClipType clip_type,
                          const Clipper2Lib::Paths64& subject,
                          Clipper2Lib::Paths64 clip,
                          ApplySafetyOffset do_safety_offset)
{
    return Paths64_to_polygons(clipper2_do_paths(clip_type, subject, std::move(clip), do_safety_offset));
}

static Slic3r::ExPolygons _clipper2_ex(Clipper2Lib::ClipType clip_type,
                               const Clipper2Lib::Paths64& subject,
                               Clipper2Lib::Paths64 clip,
                               ApplySafetyOffset do_safety_offset)
{
    return clipper2_do_expolygons(clip_type, subject, std::move(clip), do_safety_offset);
}

Slic3r::Polygons diff_2(const Slic3r::Polygon& subject, const Slic3r::Polygon& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2(Clipper2Lib::ClipType::Difference, Slic3rPolygon_to_Paths64(subject), Slic3rPolygon_to_Paths64(clip), do_safety_offset); }
Slic3r::Polygons diff_2(const Slic3r::Polygons& subject, const Slic3r::Polygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2(Clipper2Lib::ClipType::Difference, Slic3rPolygons_to_Paths64(subject), Slic3rPolygons_to_Paths64(clip), do_safety_offset); }
Slic3r::Polygons diff_2(const Slic3r::Polygons& subject, const Slic3r::ExPolygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2(Clipper2Lib::ClipType::Difference, Slic3rPolygons_to_Paths64(subject), Slic3rExPolygons_to_Paths64(clip), do_safety_offset); }
Slic3r::Polygons diff_2(const Slic3r::ExPolygons& subject, const Slic3r::Polygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2(Clipper2Lib::ClipType::Difference, Slic3rExPolygons_to_Paths64(subject), Slic3rPolygons_to_Paths64(clip), do_safety_offset); }
Slic3r::Polygons diff_2(const Slic3r::ExPolygons& subject, const Slic3r::ExPolygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2(Clipper2Lib::ClipType::Difference, Slic3rExPolygons_to_Paths64(subject), Slic3rExPolygons_to_Paths64(clip), do_safety_offset); }

Slic3r::ExPolygons diff_ex_2(const Slic3r::Polygon& subject, const Slic3r::Polygon& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Difference, Slic3rPolygon_to_Paths64(subject), Slic3rPolygon_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons diff_ex_2(const Slic3r::Polygon& subject, const Slic3r::Polygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Difference, Slic3rPolygon_to_Paths64(subject), Slic3rPolygons_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons diff_ex_2(const Slic3r::Polygon& subject, const Slic3r::ExPolygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Difference, Slic3rPolygon_to_Paths64(subject), Slic3rExPolygons_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons diff_ex_2(const Slic3r::Polygons& subject, const Slic3r::Polygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Difference, Slic3rPolygons_to_Paths64(subject), Slic3rPolygons_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons diff_ex_2(const Slic3r::Polygons& subject, const Slic3r::ExPolygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Difference, Slic3rPolygons_to_Paths64(subject), Slic3rExPolygons_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons diff_ex_2(const Slic3r::ExPolygon& subject, const Slic3r::Polygon& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Difference, Slic3rExPolygon_to_Paths64(subject), Slic3rPolygon_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons diff_ex_2(const Slic3r::ExPolygon& subject, const Slic3r::Polygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Difference, Slic3rExPolygon_to_Paths64(subject), Slic3rPolygons_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons diff_ex_2(const Slic3r::ExPolygon& subject, const Slic3r::ExPolygon& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Difference, Slic3rExPolygon_to_Paths64(subject), Slic3rExPolygon_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons diff_ex_2(const Slic3r::ExPolygon& subject, const Slic3r::ExPolygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Difference, Slic3rExPolygon_to_Paths64(subject), Slic3rExPolygons_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons diff_ex_2(const Slic3r::ExPolygons& subject, const Slic3r::Polygon& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Difference, Slic3rExPolygons_to_Paths64(subject), Slic3rPolygon_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons diff_ex_2(const Slic3r::ExPolygons& subject, const Slic3r::Polygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Difference, Slic3rExPolygons_to_Paths64(subject), Slic3rPolygons_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons diff_ex_2(const Slic3r::ExPolygons& subject, const Slic3r::ExPolygon& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Difference, Slic3rExPolygons_to_Paths64(subject), Slic3rExPolygon_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons diff_ex_2(const Slic3r::ExPolygons& subject, const Slic3r::ExPolygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Difference, Slic3rExPolygons_to_Paths64(subject), Slic3rExPolygons_to_Paths64(clip), do_safety_offset); }

Slic3r::Polygons intersection_2(const Slic3r::Polygon& subject, const Slic3r::Polygon& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2(Clipper2Lib::ClipType::Intersection, Slic3rPolygon_to_Paths64(subject), Slic3rPolygon_to_Paths64(clip), do_safety_offset); }
Slic3r::Polygons intersection_2(const Slic3r::Polygons& subject, const Slic3r::Polygon& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2(Clipper2Lib::ClipType::Intersection, Slic3rPolygons_to_Paths64(subject), Slic3rPolygon_to_Paths64(clip), do_safety_offset); }
Slic3r::Polygons intersection_2(const Slic3r::Polygons& subject, const Slic3r::Polygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2(Clipper2Lib::ClipType::Intersection, Slic3rPolygons_to_Paths64(subject), Slic3rPolygons_to_Paths64(clip), do_safety_offset); }
Slic3r::Polygons intersection_2(const Slic3r::Polygons& subject, const Slic3r::ExPolygon& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2(Clipper2Lib::ClipType::Intersection, Slic3rPolygons_to_Paths64(subject), Slic3rExPolygon_to_Paths64(clip), do_safety_offset); }
Slic3r::Polygons intersection_2(const Slic3r::ExPolygon& subject, const Slic3r::ExPolygon& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2(Clipper2Lib::ClipType::Intersection, Slic3rExPolygon_to_Paths64(subject), Slic3rExPolygon_to_Paths64(clip), do_safety_offset); }
Slic3r::Polygons intersection_2(const Slic3r::ExPolygons& subject, const Slic3r::Polygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2(Clipper2Lib::ClipType::Intersection, Slic3rExPolygons_to_Paths64(subject), Slic3rPolygons_to_Paths64(clip), do_safety_offset); }
Slic3r::Polygons intersection_2(const Slic3r::ExPolygons& subject, const Slic3r::ExPolygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2(Clipper2Lib::ClipType::Intersection, Slic3rExPolygons_to_Paths64(subject), Slic3rExPolygons_to_Paths64(clip), do_safety_offset); }

Slic3r::ExPolygons intersection_ex_2(const Slic3r::Polygon& subject, const Slic3r::Polygon& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Intersection, Slic3rPolygon_to_Paths64(subject), Slic3rPolygon_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons intersection_ex_2(const Slic3r::Polygons& subject, const Slic3r::Polygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Intersection, Slic3rPolygons_to_Paths64(subject), Slic3rPolygons_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons intersection_ex_2(const Slic3r::Polygons& subject, const Slic3r::ExPolygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Intersection, Slic3rPolygons_to_Paths64(subject), Slic3rExPolygons_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons intersection_ex_2(const Slic3r::ExPolygon& subject, const Slic3r::Polygon& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Intersection, Slic3rExPolygon_to_Paths64(subject), Slic3rPolygon_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons intersection_ex_2(const Slic3r::ExPolygon& subject, const Slic3r::Polygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Intersection, Slic3rExPolygon_to_Paths64(subject), Slic3rPolygons_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons intersection_ex_2(const Slic3r::ExPolygon& subject, const Slic3r::ExPolygon& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Intersection, Slic3rExPolygon_to_Paths64(subject), Slic3rExPolygon_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons intersection_ex_2(const Slic3r::ExPolygon& subject, const Slic3r::ExPolygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Intersection, Slic3rExPolygon_to_Paths64(subject), Slic3rExPolygons_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons intersection_ex_2(const Slic3r::ExPolygons& subject, const Slic3r::Polygon& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Intersection, Slic3rExPolygons_to_Paths64(subject), Slic3rPolygon_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons intersection_ex_2(const Slic3r::ExPolygons& subject, const Slic3r::Polygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Intersection, Slic3rExPolygons_to_Paths64(subject), Slic3rPolygons_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons intersection_ex_2(const Slic3r::ExPolygons& subject, const Slic3r::ExPolygon& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Intersection, Slic3rExPolygons_to_Paths64(subject), Slic3rExPolygon_to_Paths64(clip), do_safety_offset); }
Slic3r::ExPolygons intersection_ex_2(const Slic3r::ExPolygons& subject, const Slic3r::ExPolygons& clip, ApplySafetyOffset do_safety_offset)
    { return _clipper2_ex(Clipper2Lib::ClipType::Intersection, Slic3rExPolygons_to_Paths64(subject), Slic3rExPolygons_to_Paths64(clip), do_safety_offset); }

Slic3r::ExPolygons union_ex_2(const Slic3r::Polygons& polygons)
{
    return union_ex_2(polygons, false);
}

Slic3r::ExPolygons union_ex_2(const Slic3r::Polygons& polygons, bool filter_degenerate)
{
    return boolean_ex_2(Clipper2Lib::ClipType::Union, Slic3rPolygons_to_Paths64(polygons, filter_degenerate), {}, ApplySafetyOffset::No);
}

Slic3r::ExPolygons union_ex_2(const Slic3r::ExPolygons &expolygons)
{
    return union_ex_2(expolygons, false);
}

Slic3r::ExPolygons union_ex_2(const Slic3r::ExPolygons &expolygons, bool filter_degenerate)
{
    return boolean_ex_2(Clipper2Lib::ClipType::Union, Slic3rExPolygons_to_Paths64(expolygons, filter_degenerate), {}, ApplySafetyOffset::No);
}

Slic3r::ExPolygons offset_ex_2(const Clipper2Lib::Paths64 &paths, double delta, ClipperLib::JoinType join_type, double miter_limit)
{
    if (paths.empty())
        return {};
    if (delta == 0.)
        return boolean_ex_2(Clipper2Lib::ClipType::Union, paths, {}, ApplySafetyOffset::No);
    Clipper2Lib::ClipperOffset offsetter;
    configure_clipper2_offsetter(offsetter, join_type, miter_limit);
    offsetter.AddPaths(paths, clipper2_join_type(join_type), Clipper2Lib::EndType::Polygon);
    Clipper2Lib::PolyTree64 solution;
    offsetter.Execute(delta, solution);
    return PolyTree64_to_expolygons(solution);
}

Slic3r::ExPolygons offset_ex_2(const Slic3r::ExPolygon &expolygon, double delta, ClipperLib::JoinType join_type, double miter_limit)
{
    return offset_ex_2(Slic3rExPolygon_to_Paths64(expolygon, true), delta, join_type, miter_limit);
}

Slic3r::ExPolygons offset_ex_2(const Slic3r::ExPolygons &expolygons, double delta, ClipperLib::JoinType join_type, double miter_limit)
{
    return offset_ex_2(Slic3rExPolygons_to_Paths64(expolygons, true), delta, join_type, miter_limit);
}

// 对 ExPolygons 进行偏移
Slic3r::ExPolygons offset_ex_2(const Slic3r::ExPolygons &expolygons, double delta)
{
    Clipper2Lib::Paths64 subject = Slic3rExPolygons_to_Paths64(expolygons);
    Clipper2Lib::ClipperOffset offsetter;
    offsetter.AddPaths(subject, Clipper2Lib::JoinType::Round, Clipper2Lib::EndType::Polygon);
    Clipper2Lib::PolyPath64 polytree;
    offsetter.Execute(delta, polytree);
    Slic3r::ExPolygons results = PolyTreeToExPolygons(std::move(polytree));

    return results;
}

Slic3r::ExPolygons offset2_ex_2(const Slic3r::ExPolygons& expolygons, double delta1, double delta2)
{
    // 1st offset
    Clipper2Lib::Paths64       subject = Slic3rExPolygons_to_Paths64(expolygons);
    Clipper2Lib::ClipperOffset offsetter;
    offsetter.AddPaths(subject, Clipper2Lib::JoinType::Round, Clipper2Lib::EndType::Polygon);
    Clipper2Lib::PolyPath64 polytree;
    offsetter.Execute(delta1, polytree);

    // simplify the result
    Clipper2Lib::PolyPath64 polytree2;
    SimplifyPolyTree(polytree, SCALED_EPSILON, polytree2);

    // 2nd offset
    offsetter.Clear();
    offsetter.AddPaths(Clipper2Lib::PolyTreeToPaths64(polytree2), Clipper2Lib::JoinType::Round, Clipper2Lib::EndType::Polygon);
    polytree.Clear();
    offsetter.Execute(delta2, polytree);

    // convert back to expolygons
    Slic3r::ExPolygons results = PolyTreeToExPolygons(std::move(polytree));

    return results;
}

Slic3r::ExPolygons offset2_ex_2(const Slic3r::ExPolygons& expolygons, double delta1, double delta2, ClipperLib::JoinType join_type, double miter_limit)
{
    return offset_ex_2(offset_ex_2(expolygons, delta1, join_type, miter_limit), delta2, join_type, miter_limit);
}

Slic3r::ExPolygons closing_ex_2(const Slic3r::ExPolygons& expolygons, double delta, ClipperLib::JoinType join_type, double miter_limit)
{
    return offset2_ex_2(expolygons, delta, -delta, join_type, miter_limit);
}

static void append_chained_polygons_2(const Clipper2Lib::PolyPath64& node, Slic3r::Polygons& polygons)
{
    Slic3r::Points ordering_points;
    std::vector<const Clipper2Lib::PolyPath64*> children;
    ordering_points.reserve(node.Count());
    children.reserve(node.Count());
    for (size_t child_idx = 0; child_idx < node.Count(); ++child_idx) {
        const Clipper2Lib::PolyPath64* child = node.Child(child_idx);
        if (child->Polygon().empty())
            continue;
        children.emplace_back(child);
        const Clipper2Lib::Point64& point = child->Polygon().front();
        ordering_points.emplace_back(point.x, point.y);
    }
    if (children.empty())
        return;
    std::vector<size_t> order = chain_points(ordering_points);
    for (size_t order_idx : order) {
        const Clipper2Lib::PolyPath64* child = children[order_idx];
        Slic3r::Polygon polygon(Path64_to_points(child->Polygon()));
        if (child->IsHole())
            polygon.reverse();
        polygons.emplace_back(std::move(polygon));
        append_chained_polygons_2(*child, polygons);
    }
}

Slic3r::Polygons union_pt_chained_outside_in_2(const Slic3r::Polygons& subject)
{
    if (subject.empty())
        return {};
    Clipper2Lib::Paths64 paths = Slic3rPolygons_to_Paths64(subject, true);
    if (paths.empty())
        return {};
    Clipper2Lib::Clipper64 clipper;
    clipper.AddSubject(paths);
    Clipper2Lib::PolyTree64 solution;
    clipper.Execute(Clipper2Lib::ClipType::Union, Clipper2Lib::FillRule::EvenOdd, solution);
    Slic3r::Polygons polygons;
    append_chained_polygons_2(solution, polygons);
    return polygons;
}

}
