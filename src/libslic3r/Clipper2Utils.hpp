#ifndef slic3r_Clipper2Utils_hpp_
#define slic3r_Clipper2Utils_hpp_

#include "ExPolygon.hpp"
#include "Polygon.hpp"
#include "Polyline.hpp"
#include "clipper.hpp"
#include "clipper2/clipper.h"

namespace Slic3r {

enum class ApplySafetyOffset;

Clipper2Lib::Point64 Slic3rPoint_to_Point64(const Slic3r::Point& in);
Clipper2Lib::Path64  Slic3rPolygon_to_Path64(const Slic3r::Polygon& in);
Clipper2Lib::Paths64 Slic3rPolygon_to_Paths64(const Slic3r::Polygon& in);
Clipper2Lib::Paths64 Slic3rPolylines_to_Paths64(const Slic3r::Polylines& in);
Clipper2Lib::Paths64 Slic3rPolygons_to_Paths64(const Slic3r::Polygons& in);
Clipper2Lib::Paths64 Slic3rPolygons_to_Paths64(const Slic3r::Polygons& in, bool filter_degenerate);
Clipper2Lib::Paths64 Slic3rExPolygon_to_Paths64(const Slic3r::ExPolygon& in);
Clipper2Lib::Paths64 Slic3rExPolygon_to_Paths64(const Slic3r::ExPolygon& in, bool filter_degenerate);
Clipper2Lib::Paths64 Slic3rExPolygons_to_Paths64(const Slic3r::ExPolygons& in);
Clipper2Lib::Paths64 Slic3rExPolygons_to_Paths64(const Slic3r::ExPolygons& in, bool filter_degenerate);
Slic3r::Points     Path64_to_points(const Clipper2Lib::Path64& in);
Slic3r::Polylines  Paths64_to_polylines(const Clipper2Lib::Paths64& in);
Slic3r::Polygons   Paths64_to_polygons(const Clipper2Lib::Paths64& in);
Slic3r::ExPolygons PolyTree64_to_expolygons(const Clipper2Lib::PolyTree64& in);
Slic3r::ExPolygons PolyTree64_to_expolygons(Clipper2Lib::PolyTree64&& in);
Clipper2Lib::JoinType clipper2_join_type(ClipperLib::JoinType join_type);
void                  configure_clipper2_offsetter(Clipper2Lib::ClipperOffset& offsetter, ClipperLib::JoinType join_type, double miter_limit);
Clipper2Lib::Paths64  offset_paths_2(const Clipper2Lib::Paths64& paths, double delta, ClipperLib::JoinType join_type, double miter_limit);
Slic3r::ExPolygons    boolean_ex_2(Clipper2Lib::ClipType clip_type, const Clipper2Lib::Paths64& subject, Clipper2Lib::Paths64 clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::Polylines  intersection_pl_2(const Slic3r::Polylines& subject, const Slic3r::Polygons& clip);
Slic3r::Polylines  diff_pl_2(const Slic3r::Polylines& subject, const Slic3r::Polygons& clip);
Slic3r::Polygons   diff_2(const Slic3r::Polygon &subject, const Slic3r::Polygon &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::Polygons   diff_2(const Slic3r::Polygons &subject, const Slic3r::Polygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::Polygons   diff_2(const Slic3r::Polygons &subject, const Slic3r::ExPolygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::Polygons   diff_2(const Slic3r::ExPolygons &subject, const Slic3r::Polygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::Polygons   diff_2(const Slic3r::ExPolygons &subject, const Slic3r::ExPolygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons diff_ex_2(const Slic3r::Polygon &subject, const Slic3r::Polygon &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons diff_ex_2(const Slic3r::Polygon &subject, const Slic3r::Polygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons diff_ex_2(const Slic3r::Polygon &subject, const Slic3r::ExPolygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons diff_ex_2(const Slic3r::Polygons &subject, const Slic3r::Polygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons diff_ex_2(const Slic3r::Polygons &subject, const Slic3r::ExPolygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons diff_ex_2(const Slic3r::ExPolygon &subject, const Slic3r::Polygon &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons diff_ex_2(const Slic3r::ExPolygon &subject, const Slic3r::Polygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons diff_ex_2(const Slic3r::ExPolygon &subject, const Slic3r::ExPolygon &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons diff_ex_2(const Slic3r::ExPolygon &subject, const Slic3r::ExPolygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons diff_ex_2(const Slic3r::ExPolygons &subject, const Slic3r::Polygon &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons diff_ex_2(const Slic3r::ExPolygons &subject, const Slic3r::Polygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons diff_ex_2(const Slic3r::ExPolygons &subject, const Slic3r::ExPolygon &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons diff_ex_2(const Slic3r::ExPolygons &subject, const Slic3r::ExPolygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::Polygons   intersection_2(const Slic3r::Polygon &subject, const Slic3r::Polygon &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::Polygons   intersection_2(const Slic3r::Polygons &subject, const Slic3r::Polygon &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::Polygons   intersection_2(const Slic3r::Polygons &subject, const Slic3r::Polygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::Polygons   intersection_2(const Slic3r::Polygons &subject, const Slic3r::ExPolygon &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::Polygons   intersection_2(const Slic3r::ExPolygon &subject, const Slic3r::ExPolygon &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::Polygons   intersection_2(const Slic3r::ExPolygons &subject, const Slic3r::Polygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::Polygons   intersection_2(const Slic3r::ExPolygons &subject, const Slic3r::ExPolygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons intersection_ex_2(const Slic3r::Polygon &subject, const Slic3r::Polygon &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons intersection_ex_2(const Slic3r::Polygons &subject, const Slic3r::Polygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons intersection_ex_2(const Slic3r::Polygons &subject, const Slic3r::ExPolygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons intersection_ex_2(const Slic3r::ExPolygon &subject, const Slic3r::Polygon &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons intersection_ex_2(const Slic3r::ExPolygon &subject, const Slic3r::Polygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons intersection_ex_2(const Slic3r::ExPolygon &subject, const Slic3r::ExPolygon &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons intersection_ex_2(const Slic3r::ExPolygon &subject, const Slic3r::ExPolygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons intersection_ex_2(const Slic3r::ExPolygons &subject, const Slic3r::Polygon &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons intersection_ex_2(const Slic3r::ExPolygons &subject, const Slic3r::Polygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons intersection_ex_2(const Slic3r::ExPolygons &subject, const Slic3r::ExPolygon &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons intersection_ex_2(const Slic3r::ExPolygons &subject, const Slic3r::ExPolygons &clip, ApplySafetyOffset do_safety_offset = ApplySafetyOffset{});
Slic3r::ExPolygons         union_ex_2(const Slic3r::Polygons &expolygons);
Slic3r::ExPolygons         union_ex_2(const Slic3r::Polygons &expolygons, bool filter_degenerate);
Slic3r::ExPolygons         union_ex_2(const Slic3r::ExPolygons &expolygons);
Slic3r::ExPolygons         union_ex_2(const Slic3r::ExPolygons &expolygons, bool filter_degenerate);
Slic3r::ExPolygons         offset_ex_2(const Clipper2Lib::Paths64 &paths, double delta, ClipperLib::JoinType join_type, double miter_limit);
Slic3r::ExPolygons         offset_ex_2(const Slic3r::ExPolygon &expolygon, double delta, ClipperLib::JoinType join_type, double miter_limit);
Slic3r::ExPolygons         offset_ex_2(const Slic3r::ExPolygons &expolygons, double delta);
Slic3r::ExPolygons         offset_ex_2(const Slic3r::ExPolygons &expolygons, double delta, ClipperLib::JoinType join_type, double miter_limit);
Slic3r::ExPolygons         offset2_ex_2(const Slic3r::ExPolygons &expolygons, double delta1, double delta2);
Slic3r::ExPolygons         offset2_ex_2(const Slic3r::ExPolygons &expolygons, double delta1, double delta2, ClipperLib::JoinType join_type, double miter_limit);
Slic3r::ExPolygons         closing_ex_2(const Slic3r::ExPolygons &expolygons, double delta, ClipperLib::JoinType join_type, double miter_limit);
Slic3r::Polygons           union_pt_chained_outside_in_2(const Slic3r::Polygons &subject);
}

#endif
