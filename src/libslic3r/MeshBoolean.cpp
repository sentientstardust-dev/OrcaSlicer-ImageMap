#include "Exception.hpp"
#include "MeshBoolean.hpp"
#include "libslic3r/MeshSplitImpl.hpp"
#include "libslic3r/TriangleMesh.hpp"
#include "libslic3r/TryCatchSignal.hpp"
#include "libslic3r/format.hpp"
#undef PI

#include <boost/next_prior.hpp>
#include "boost/log/trivial.hpp"
// Include igl first. It defines "L" macro which then clashes with our localization
#include <igl/copyleft/cgal/mesh_boolean.h>
#undef L

// CGAL headers
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Exact_integer.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Cartesian_converter.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
// BBS: for segment
#include <CGAL/mesh_segmentation.h>
#include <CGAL/property_map.h>
#include <CGAL/boost/graph/copy_face_graph.h>
#include <CGAL/boost/graph/Face_filtered_graph.h>
// BBS: for boolean using mcut
#include "mcut/include/mcut/mcut.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace Slic3r {
namespace MeshBoolean {

using MapMatrixXfUnaligned = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::DontAlign>>;
using MapMatrixXiUnaligned = Eigen::Map<const Eigen::Matrix<int,   Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::DontAlign>>;

TriangleMesh eigen_to_triangle_mesh(const EigenMesh &emesh)
{
    auto &VC = emesh.first; auto &FC = emesh.second;

    indexed_triangle_set its;
    its.vertices.reserve(size_t(VC.rows()));
    its.indices.reserve(size_t(FC.rows()));

    for (Eigen::Index i = 0; i < VC.rows(); ++i)
        its.vertices.emplace_back(VC.row(i).cast<float>());

    for (Eigen::Index i = 0; i < FC.rows(); ++i)
        its.indices.emplace_back(FC.row(i));

    return TriangleMesh { std::move(its) };
}

EigenMesh triangle_mesh_to_eigen(const TriangleMesh &mesh)
{
    EigenMesh emesh;
    emesh.first = MapMatrixXfUnaligned(mesh.its.vertices.front().data(),
                                       Eigen::Index(mesh.its.vertices.size()),
                                       3).cast<double>();

    emesh.second = MapMatrixXiUnaligned(mesh.its.indices.front().data(),
                                        Eigen::Index(mesh.its.indices.size()),
                                        3);
    return emesh;
}

void minus(EigenMesh &A, const EigenMesh &B)
{
    auto &[VA, FA] = A;
    auto &[VB, FB] = B;

    Eigen::MatrixXd VC;
    Eigen::MatrixXi FC;
    igl::MeshBooleanType boolean_type(igl::MESH_BOOLEAN_TYPE_MINUS);
    igl::copyleft::cgal::mesh_boolean(VA, FA, VB, FB, boolean_type, VC, FC);

    VA = std::move(VC); FA = std::move(FC);
}

void minus(TriangleMesh& A, const TriangleMesh& B)
{
    EigenMesh eA = triangle_mesh_to_eigen(A);
    minus(eA, triangle_mesh_to_eigen(B));
    A = eigen_to_triangle_mesh(eA);
}

void self_union(EigenMesh &A)
{
    EigenMesh result;
    auto &[V, F] = A;
    auto &[VC, FC] = result;

    igl::MeshBooleanType boolean_type(igl::MESH_BOOLEAN_TYPE_UNION);
    igl::copyleft::cgal::mesh_boolean(V, F, Eigen::MatrixXd(), Eigen::MatrixXi(), boolean_type, VC, FC);

    A = std::move(result);
}

void self_union(TriangleMesh& mesh)
{
    auto eM = triangle_mesh_to_eigen(mesh);
    self_union(eM);
    mesh = eigen_to_triangle_mesh(eM);
}

namespace cgal {

namespace CGALProc    = CGAL::Polygon_mesh_processing;
namespace CGALParams  = CGAL::Polygon_mesh_processing::parameters;

using EpecKernel = CGAL::Exact_predicates_exact_constructions_kernel;
using EpicKernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using _EpicMesh = CGAL::Surface_mesh<EpicKernel::Point_3>;
using _EpecMesh = CGAL::Surface_mesh<EpecKernel::Point_3>;

struct CGALMesh {
    _EpicMesh m;
    CGALMesh() = default;
    CGALMesh(const _EpicMesh& _m) :m(_m) {}
};

void save_CGALMesh(const std::string& fname, const CGALMesh& cgal_mesh) {
    std::ofstream os(fname);
    os << cgal_mesh.m;
    os.close();
}
// /////////////////////////////////////////////////////////////////////////////
// Converions from and to CGAL mesh
// /////////////////////////////////////////////////////////////////////////////

template<class _Mesh> void triangle_mesh_to_cgal(const TriangleMesh& M, _Mesh& out)
{
    using Index3 = std::array<size_t, 3>;

    if (M.empty()) return;

    std::vector<typename _Mesh::Point> points;
    std::vector<Index3> indices;
    points.reserve(M.its.vertices.size());
    indices.reserve(M.its.indices.size());
    for (auto& v : M.its.vertices) points.emplace_back(v.x(), v.y(), v.z());
    for (auto& _f : M.its.indices) {
        auto f = _f.cast<size_t>();
        indices.emplace_back(Index3{ f(0), f(1), f(2) });
    }

    CGALProc::orient_polygon_soup(points, indices);
    CGALProc::polygon_soup_to_polygon_mesh(points, indices, out);

    // Number the faces because 'orient_to_bound_a_volume' needs a face <--> index map
    unsigned index = 0;
    for (auto face : out.faces()) face = CGAL::SM_Face_index(index++);

    if (CGAL::is_closed(out))
        CGALProc::orient_to_bound_a_volume(out);
    else
        throw Slic3r::RuntimeError("Mesh not watertight");
}

template<class _Mesh>
void triangle_mesh_to_cgal(const std::vector<stl_vertex> &                 V,
                           const std::vector<stl_triangle_vertex_indices> &F,
                           _Mesh &out)
{
    if (F.empty()) return;

    size_t vertices_count = V.size();
    size_t edges_count    = (F.size()* 3) / 2;
    size_t faces_count    = F.size();
    out.reserve(vertices_count, edges_count, faces_count);

    for (auto &v : V)
        out.add_vertex(typename _Mesh::Point{v.x(), v.y(), v.z()});

    using VI = typename _Mesh::Vertex_index;
    for (auto &f : F)
        out.add_face(VI(f(0)), VI(f(1)), VI(f(2)));
}

inline Vec3f to_vec3f(const _EpicMesh::Point& v)
{
    return { float(v.x()), float(v.y()), float(v.z()) };
}

inline Vec3f to_vec3f(const _EpecMesh::Point& v)
{
    CGAL::Cartesian_converter<EpecKernel, EpicKernel> cvt;
    auto iv = cvt(v);
    return { float(iv.x()), float(iv.y()), float(iv.z()) };
}

template<class _Mesh>
indexed_triangle_set cgal_to_indexed_triangle_set(const _Mesh &cgalmesh)
{
    indexed_triangle_set its;
    its.vertices.reserve(cgalmesh.num_vertices());
    its.indices.reserve(cgalmesh.num_faces());

    const auto &faces = cgalmesh.faces();
    const auto &vertices = cgalmesh.vertices();
    int vsize = int(vertices.size());

    for (const auto &vi : vertices) {
        auto &v = cgalmesh.point(vi); // Don't ask...
        its.vertices.emplace_back(to_vec3f(v));
    }

    for (const auto &face : faces) {
        auto vtc = cgalmesh.vertices_around_face(cgalmesh.halfedge(face));

        int i = 0;
        Vec3i32 facet;
        for (auto v : vtc) {
            int iv = v;
            if (i > 2 || iv < 0 || iv >= vsize) { i = 0; break; }
            facet(i++) = iv;
        }

        if (i == 3)
            its.indices.emplace_back(facet);
    }

    return its;
}

template<class _Mesh> TriangleMesh cgal_to_triangle_mesh(const _Mesh &cgalmesh)
{
    indexed_triangle_set its = cgal_to_indexed_triangle_set(cgalmesh);
    return TriangleMesh(std::move(its));
}

std::unique_ptr<CGALMesh, CGALMeshDeleter>
triangle_mesh_to_cgal(const std::vector<stl_vertex> &V,
                      const std::vector<stl_triangle_vertex_indices> &F)
{
    std::unique_ptr<CGALMesh, CGALMeshDeleter> out(new CGALMesh{});
    triangle_mesh_to_cgal(V, F, out->m);
    return out;
}

TriangleMesh cgal_to_triangle_mesh(const CGALMesh &cgalmesh)
{
    return TriangleMesh{cgal_to_indexed_triangle_set(cgalmesh.m)};
}

indexed_triangle_set cgal_to_indexed_triangle_set(const CGALMesh &cgalmesh)
{
    return cgal_to_indexed_triangle_set(cgalmesh.m);
}

// /////////////////////////////////////////////////////////////////////////////
// Boolean operations for CGAL meshes
// /////////////////////////////////////////////////////////////////////////////

static bool _cgal_diff(CGALMesh &A, CGALMesh &B, CGALMesh &R)
{
    const auto &p = CGALParams::throw_on_self_intersection(true);
    return CGALProc::corefine_and_compute_difference(A.m, B.m, R.m, p, p);
}

static bool _cgal_union(CGALMesh &A, CGALMesh &B, CGALMesh &R)
{
    const auto &p = CGALParams::throw_on_self_intersection(true);
    return CGALProc::corefine_and_compute_union(A.m, B.m, R.m, p, p);
}

static bool _cgal_intersection(CGALMesh &A, CGALMesh &B, CGALMesh &R)
{
    const auto &p = CGALParams::throw_on_self_intersection(true);
    return CGALProc::corefine_and_compute_intersection(A.m, B.m, R.m, p, p);
}

template<class Op> void _cgal_do(Op &&op, CGALMesh &A, CGALMesh &B)
{
    bool success = false;
    bool hw_fail = false;
    try {
        CGALMesh result;
        try_catch_signal({SIGSEGV, SIGFPE}, [&success, &A, &B, &result, &op] {
            success = op(A, B, result);
        }, [&] { hw_fail = true; });
        A = std::move(result);      // In-place operation does not work
    } catch (...) {
        success = false;
    }

    if (hw_fail)
        throw Slic3r::HardCrash("CGAL mesh boolean operation crashed.");

    if (! success)
        throw Slic3r::RuntimeError("CGAL mesh boolean operation failed.");
}

void minus(CGALMesh &A, CGALMesh &B) { _cgal_do(_cgal_diff, A, B); }
void plus(CGALMesh &A, CGALMesh &B) { _cgal_do(_cgal_union, A, B); }
void intersect(CGALMesh &A, CGALMesh &B) { _cgal_do(_cgal_intersection, A, B); }
bool does_self_intersect(const CGALMesh &mesh) { return CGALProc::does_self_intersect(mesh.m); }
// BBS
void segment(CGALMesh& src, std::vector<CGALMesh>& dst, double smoothing_alpha = 0.5, int segment_number=5)
{
    typedef boost::graph_traits<_EpicMesh>::face_descriptor face_descriptor;
    typedef _EpicMesh::Property_map<face_descriptor, double> Facet_double_map;
    typedef CGAL::Face_filtered_graph<_EpicMesh> Filtered_graph;

    _EpicMesh mesh = src.m;
    Facet_double_map sdf_property_map;

    sdf_property_map = mesh.add_property_map<face_descriptor, double>("f:sdf").first;

    CGAL::sdf_values(mesh, sdf_property_map);

    // create a property-map for segment-ids
    typedef _EpicMesh::Property_map<face_descriptor, std::size_t> Facet_int_map;
    Facet_int_map segment_property_map = mesh.add_property_map<face_descriptor, std::size_t>("f:sid").first;;
    // segment the mesh using default parameters for number of levels, and smoothing lambda
    // Any other scalar values can be used instead of using SDF values computed using the CGAL function
    std::size_t number_of_segments = CGAL::segmentation_from_sdf_values(mesh, sdf_property_map, segment_property_map, segment_number, smoothing_alpha);
    //print area of each segment and then put it in a Mesh and print it in an OFF file
    Filtered_graph segment_mesh(mesh);
    _EpicMesh mesh_merged;
    for (std::size_t id = 0; id < number_of_segments; ++id)
    {
        segment_mesh.set_selected_faces(id, segment_property_map);
        //std::cout << "Segment " << id << "'s area is : " << CGAL::Polygon_mesh_processing::area(segment_mesh) << std::endl;
        _EpicMesh out;
        CGAL::copy_face_graph(segment_mesh, out);

        // save_CGALMesh("out.off", out);

        // fill holes
        typedef boost::graph_traits<_EpicMesh>::halfedge_descriptor      halfedge_descriptor;
        typedef boost::graph_traits<_EpicMesh>::vertex_descriptor        vertex_descriptor;
        std::vector<halfedge_descriptor> border_cycles;
        CGAL::Polygon_mesh_processing::extract_boundary_cycles(out, std::back_inserter(border_cycles));
        for (halfedge_descriptor h : border_cycles)
        {
            std::vector<face_descriptor>  patch_facets;
#if 0
            std::vector<vertex_descriptor> patch_vertices;
            CGAL::Polygon_mesh_processing::triangulate_and_refine_hole(out, h, std::back_inserter(patch_facets),
                std::back_inserter(patch_vertices));
            std::cout << "* Number of facets in constructed patch: " << patch_facets.size() << std::endl;
            std::cout << "  Number of vertices in constructed patch: " << patch_vertices.size() << std::endl;
#else
            CGAL::Polygon_mesh_processing::triangulate_hole(out, h, CGAL::parameters::default_values().face_output_iterator(std::back_inserter(patch_facets)));
#endif
        }

        //if (id > 2) {
        //    mesh_merged.join(out);
        //}
        //else
        {
            dst.emplace_back(std::move(CGALMesh(out)));
        }
    }
    //if (mesh_merged.is_empty() == false) {
    //    CGAL::Polygon_mesh_processing::stitch_borders(mesh_merged);
    //    dst.emplace_back(std::move(CGALMesh(mesh_merged)));
    //}
}

std::vector<TriangleMesh> segment(const TriangleMesh& src, double smoothing_alpha, int segment_number)
{
    CGALMesh in_cgal_mesh;
    MeshBoolean::cgal::triangle_mesh_to_cgal(src, in_cgal_mesh.m);
    std::vector<CGALMesh> out_cgal_meshes;
    segment(in_cgal_mesh, out_cgal_meshes, smoothing_alpha, segment_number);

    std::vector<TriangleMesh> out_meshes;
    for (auto& outf_cgal_mesh: out_cgal_meshes)
    {
        out_meshes.emplace_back(std::move(cgal_to_triangle_mesh(outf_cgal_mesh.m)));
    }

    return out_meshes;
}

void merge(std::vector<_EpicMesh>& srcs, _EpicMesh& dst)
{
    _EpicMesh mesh_merged;
    for (size_t i = 0; i < srcs.size(); i++)
    {
        mesh_merged.join(srcs[i]);
    }
    if (mesh_merged.is_empty() == false) {
        CGAL::Polygon_mesh_processing::stitch_borders(mesh_merged);
        dst = std::move(mesh_merged);
    }
}

TriangleMesh merge(std::vector<TriangleMesh> meshes)
{
    std::vector<_EpicMesh> srcs(meshes.size());
    for (size_t i = 0; i < meshes.size(); i++)
    {
        MeshBoolean::cgal::triangle_mesh_to_cgal(meshes[i], srcs[i]);
    }
    _EpicMesh dst;
    merge(srcs, dst);
    return cgal_to_triangle_mesh(dst);
}

template<class Op> void _mesh_boolean_do(Op &&op, indexed_triangle_set &A, const indexed_triangle_set &B)
{
    CGALMesh meshA;
    CGALMesh meshB;
    triangle_mesh_to_cgal(A.vertices, A.indices, meshA.m);
    triangle_mesh_to_cgal(B.vertices, B.indices, meshB.m);

    _cgal_do(op, meshA, meshB);

    A = cgal_to_indexed_triangle_set(meshA.m);
}

template<class Op> void _mesh_boolean_do(Op &&op, TriangleMesh &A, const TriangleMesh &B)
{
    CGALMesh meshA;
    CGALMesh meshB;
    triangle_mesh_to_cgal(A.its.vertices, A.its.indices, meshA.m);
    triangle_mesh_to_cgal(B.its.vertices, B.its.indices, meshB.m);

    _cgal_do(op, meshA, meshB);

    A = cgal_to_triangle_mesh(meshA);
}

void minus(TriangleMesh &A, const TriangleMesh &B)
{
    _mesh_boolean_do(_cgal_diff, A, B);
}

void plus(TriangleMesh &A, const TriangleMesh &B)
{
    _mesh_boolean_do(_cgal_union, A, B);
}

void intersect(TriangleMesh &A, const TriangleMesh &B)
{
    _mesh_boolean_do(_cgal_intersection, A, B);
}

void minus(indexed_triangle_set &A, const indexed_triangle_set &B)
{
    _mesh_boolean_do(_cgal_diff, A, B);
}

void plus(indexed_triangle_set &A, const indexed_triangle_set &B)
{
    _mesh_boolean_do(_cgal_union, A, B);
}

void intersect(indexed_triangle_set &A, const indexed_triangle_set &B)
{
    _mesh_boolean_do(_cgal_intersection, A, B);
}

bool does_self_intersect(const TriangleMesh &mesh)
{
    CGALMesh cgalm;
    triangle_mesh_to_cgal(mesh.its.vertices, mesh.its.indices, cgalm.m);
    return CGALProc::does_self_intersect(cgalm.m);
}

void CGALMeshDeleter::operator()(CGALMesh *ptr) { delete ptr; }

bool does_bound_a_volume(const CGALMesh &mesh)
{
    return CGAL::is_closed(mesh.m) && CGALProc::does_bound_a_volume(mesh.m);
}

bool empty(const CGALMesh &mesh)
{
    return mesh.m.is_empty();
}

bool repair(TriangleMesh& mesh, RepairedMeshErrors* repaired_errors, std::string* error)
{
    using namespace CGAL;
    namespace PMP = CGAL::Polygon_mesh_processing;

    if (mesh.empty())
        return true;

    try {
        // 1) Convert to polygon soup
        std::vector<_EpicMesh::Point>         points;
        std::vector<std::vector<std::size_t>> polygons;

        points.reserve(mesh.its.vertices.size());
        polygons.reserve(mesh.its.indices.size());

        for (const auto& v : mesh.its.vertices)
            points.emplace_back(v.x(), v.y(), v.z());

        for (const auto& f : mesh.its.indices)
            polygons.push_back({size_t(f[0]), size_t(f[1]), size_t(f[2])});

        // 2) Aggressive soup cleanup
        PMP::repair_polygon_soup(points, polygons);

        // 3) Convert soup → mesh
        _EpicMesh cgal_mesh;
        PMP::polygon_soup_to_polygon_mesh(points, polygons, cgal_mesh);

        // 4) Remove degenerate geometry
        PMP::remove_degenerate_faces(cgal_mesh);
        PMP::remove_isolated_vertices(cgal_mesh);

        // 5) Fix remaining non-manifold vertices
        PMP::duplicate_non_manifold_vertices(cgal_mesh);

        // 6) Boolean union (keeps only outer shell)
        _EpicMesh tmp;
        if (PMP::corefine_and_compute_union(cgal_mesh, cgal_mesh, tmp)) {
            cgal_mesh = std::move(tmp);
        }
        // If it fails, continue anyway with previous mesh

        // 7) Fill holes
        if (!CGAL::is_closed(cgal_mesh)) {
            using halfedge_descriptor = boost::graph_traits<_EpicMesh>::halfedge_descriptor;

            std::vector<halfedge_descriptor> borders;
            PMP::extract_boundary_cycles(cgal_mesh, std::back_inserter(borders));

            for (halfedge_descriptor h : borders) {
                PMP::triangulate_and_refine_hole(cgal_mesh, h);
            }
        }

        // 8) Final validity check
        if (!CGAL::is_closed(cgal_mesh)) {
            if (error)
                *error = "Repair failed: mesh still open after hole filling.";
            return false;
        }

        // 9) Ensure outward orientation
        if (!PMP::does_bound_a_volume(cgal_mesh))
            PMP::orient_to_bound_a_volume(cgal_mesh);

        // 10) Convert back
        indexed_triangle_set its = cgal_to_indexed_triangle_set(cgal_mesh);

        RepairedMeshErrors errs{};
        errs.facets_removed = 0;
        errs.edges_fixed    = 0;

        mesh = TriangleMesh(std::move(its), errs);

        if (repaired_errors)
            *repaired_errors = errs;

        return true;
    } catch (const std::exception& e) {
        if (error)
            *error = e.what();
        return false;
    }
}

CGALMeshPtr clone(const CGALMesh &m)
{
    return CGALMeshPtr{new CGALMesh{m}};
}

} // namespace cgal


namespace mcut {
/* BBS: MusangKing
 * mcut mesh array format for Boolean Opts calculation
 */
struct McutMesh
{
    // variables for mesh data in a format suited for mcut
    std::vector<uint32_t> faceSizesArray;
    std::vector<uint32_t> faceIndicesArray;
    std::vector<double>   vertexCoordsArray;
};
void McutMeshDeleter::operator()(McutMesh *ptr) { delete ptr; }

bool empty(const McutMesh &mesh) { return mesh.vertexCoordsArray.empty() || mesh.faceIndicesArray.empty(); }

static Vec3f mcut_vertex(const McutMesh &mesh, uint32_t vertex_idx)
{
    const size_t offset = size_t(vertex_idx) * 3;
    if (offset + 2 >= mesh.vertexCoordsArray.size())
        return Vec3f::Zero();
    return Vec3f(float(mesh.vertexCoordsArray[offset + 0]), float(mesh.vertexCoordsArray[offset + 1]), float(mesh.vertexCoordsArray[offset + 2]));
}

static size_t mcut_face_offset(const McutMesh &mesh, size_t face_idx)
{
    size_t offset = 0;
    for (size_t idx = 0; idx < face_idx && idx < mesh.faceSizesArray.size(); ++idx)
        offset += size_t(mesh.faceSizesArray[idx]);
    return offset;
}

static bool mcut_face_vertices(const McutMesh &mesh, size_t face_idx, std::array<Vec3f, 3> &vertices)
{
    if (face_idx >= mesh.faceSizesArray.size() || mesh.faceSizesArray[face_idx] != 3)
        return false;
    const size_t offset = mcut_face_offset(mesh, face_idx);
    if (offset + 2 >= mesh.faceIndicesArray.size())
        return false;
    vertices = {
        mcut_vertex(mesh, mesh.faceIndicesArray[offset + 0]),
        mcut_vertex(mesh, mesh.faceIndicesArray[offset + 1]),
        mcut_vertex(mesh, mesh.faceIndicesArray[offset + 2])
    };
    return true;
}

static Vec3f normalized_nonnegative_barycentric_mcut(Vec3f weights)
{
    weights.x() = std::max(weights.x(), 0.f);
    weights.y() = std::max(weights.y(), 0.f);
    weights.z() = std::max(weights.z(), 0.f);
    const float sum = weights.x() + weights.y() + weights.z();
    if (sum <= 1e-6f)
        return Vec3f(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);
    weights /= sum;
    return weights;
}

static Vec3f barycentric_weights_3d_mcut(const Vec3f &point, const std::array<Vec3f, 3> &vertices)
{
    const Vec3f v0 = vertices[1] - vertices[0];
    const Vec3f v1 = vertices[2] - vertices[0];
    const Vec3f v2 = point - vertices[0];
    const float d00 = v0.dot(v0);
    const float d01 = v0.dot(v1);
    const float d11 = v1.dot(v1);
    const float d20 = v2.dot(v0);
    const float d21 = v2.dot(v1);
    const float denom = d00 * d11 - d01 * d01;
    if (!std::isfinite(denom) || std::abs(denom) <= 1e-6f)
        return Vec3f(1.f / 3.f, 1.f / 3.f, 1.f / 3.f);
    const float v = (d11 * d20 - d01 * d21) / denom;
    const float w = (d00 * d21 - d01 * d20) / denom;
    return Vec3f(1.f - v - w, v, w);
}

static Vec3f interpolate_source_barycentric(const MeshFaceProvenance &provenance, const Vec3f &weights)
{
    return normalized_nonnegative_barycentric_mcut(provenance.source_barycentric[0] * weights.x() +
                                                   provenance.source_barycentric[1] * weights.y() +
                                                   provenance.source_barycentric[2] * weights.z());
}

std::vector<MeshFaceProvenance> identity_provenance(const indexed_triangle_set &its, size_t source_index)
{
    std::vector<MeshFaceProvenance> provenance(its.indices.size());
    for (size_t tri_idx = 0; tri_idx < provenance.size(); ++tri_idx) {
        MeshFaceProvenance &entry = provenance[tri_idx];
        entry.valid = true;
        entry.source_index = source_index;
        entry.source_triangle = tri_idx;
        entry.source_barycentric = {
            Vec3f(1.f, 0.f, 0.f),
            Vec3f(0.f, 1.f, 0.f),
            Vec3f(0.f, 0.f, 1.f)
        };
    }
    return provenance;
}

struct ProvenancedIts
{
    indexed_triangle_set its;
    std::vector<MeshFaceProvenance> provenance;
};

enum class McutBooleanStatus
{
    Success,
    NoOutput,
    Failure
};

static bool its_bounds_overlap_mcut(const indexed_triangle_set &a, const indexed_triangle_set &b)
{
    if (a.vertices.empty() || b.vertices.empty())
        return false;

    Vec3f a_min = Vec3f::Constant(std::numeric_limits<float>::max());
    Vec3f a_max = Vec3f::Constant(std::numeric_limits<float>::lowest());
    for (const Vec3f &v : a.vertices) {
        a_min = a_min.cwiseMin(v);
        a_max = a_max.cwiseMax(v);
    }

    Vec3f b_min = Vec3f::Constant(std::numeric_limits<float>::max());
    Vec3f b_max = Vec3f::Constant(std::numeric_limits<float>::lowest());
    for (const Vec3f &v : b.vertices) {
        b_min = b_min.cwiseMin(v);
        b_max = b_max.cwiseMax(v);
    }

    constexpr float eps = 1e-5f;
    return a_min.x() <= b_max.x() + eps && a_max.x() + eps >= b_min.x() &&
           a_min.y() <= b_max.y() + eps && a_max.y() + eps >= b_min.y() &&
           a_min.z() <= b_max.z() + eps && a_max.z() + eps >= b_min.z();
}

static std::vector<ProvenancedIts> split_with_provenance(const indexed_triangle_set &its, const std::vector<MeshFaceProvenance> &provenance)
{
    std::vector<ProvenancedIts> ret;
    if (provenance.size() != its.indices.size())
        return ret;

    struct VertexConv {
        size_t part_id = std::numeric_limits<size_t>::max();
        size_t vertex_image = 0;
    };
    std::vector<VertexConv> vidx_conv(its.vertices.size());
    meshsplit_detail::NeighborVisitor visitor(its, meshsplit_detail::ItsWithNeighborsIndex_<indexed_triangle_set>::get_index(its));

    std::vector<size_t> facets;
    for (size_t part_id = 0;; ++part_id) {
        facets.clear();
        visitor.visit([&facets](size_t idx) { facets.emplace_back(idx); return true; });
        if (facets.empty())
            break;

        ProvenancedIts part;
        part.its.indices.reserve(facets.size());
        part.its.vertices.reserve(std::min(facets.size() * 3, its.vertices.size()));
        part.provenance.reserve(facets.size());

        for (size_t face_id : facets) {
            const auto &face = its.indices[face_id];
            Vec3i32 new_face;
            for (size_t v = 0; v < 3; ++v) {
                auto vi = face(v);
                if (vidx_conv[vi].part_id != part_id) {
                    vidx_conv[vi] = { part_id, part.its.vertices.size() };
                    part.its.vertices.emplace_back(its.vertices[size_t(vi)]);
                }
                new_face(v) = int(vidx_conv[vi].vertex_image);
            }
            part.its.indices.emplace_back(new_face);
            part.provenance.emplace_back(provenance[face_id]);
        }

        ret.emplace_back(std::move(part));
    }

    return ret;
}

static void append_with_provenance(indexed_triangle_set &dst,
                                   std::vector<MeshFaceProvenance> &dst_provenance,
                                   const indexed_triangle_set &src,
                                   const std::vector<MeshFaceProvenance> &src_provenance)
{
    if (src.indices.size() != src_provenance.size())
        return;
    const int vertex_offset = int(dst.vertices.size());
    dst.vertices.insert(dst.vertices.end(), src.vertices.begin(), src.vertices.end());
    for (const Vec3i32 &face : src.indices)
        dst.indices.emplace_back(Vec3i32(face.x() + vertex_offset, face.y() + vertex_offset, face.z() + vertex_offset));
    dst_provenance.insert(dst_provenance.end(), src_provenance.begin(), src_provenance.end());
}

static void merge_mcut_meshes_with_provenance(McutMesh &srcMesh,
                                              std::vector<MeshFaceProvenance> &src_provenance,
                                              const McutMesh &cutMesh,
                                              const std::vector<MeshFaceProvenance> &cut_provenance)
{
    TriangleMesh tri_src = mcut_to_triangle_mesh(srcMesh);
    TriangleMesh tri_cut = mcut_to_triangle_mesh(cutMesh);
    indexed_triangle_set all_its;
    std::vector<MeshFaceProvenance> all_provenance;
    append_with_provenance(all_its, all_provenance, tri_src.its, src_provenance);
    append_with_provenance(all_its, all_provenance, tri_cut.its, cut_provenance);
    srcMesh = *triangle_mesh_to_mcut(all_its);
    src_provenance = std::move(all_provenance);
}

static bool mapped_output_face_provenance(const McutMesh &srcMesh,
                                          const std::vector<MeshFaceProvenance> &src_provenance,
                                          const McutMesh &cutMesh,
                                          const std::vector<MeshFaceProvenance> &cut_provenance,
                                          uint32_t face_map,
                                          const std::vector<double> &ccVertices,
                                          const std::vector<uint32_t> &ccFaceIndices,
                                          uint32_t face_offset,
                                          int vertex_index_offset,
                                          MeshFaceProvenance &out)
{
    const uint32_t src_face_count = uint32_t(srcMesh.faceSizesArray.size());
    const bool from_cut = face_map >= src_face_count;
    const McutMesh &input_mesh = from_cut ? cutMesh : srcMesh;
    const std::vector<MeshFaceProvenance> &input_provenance = from_cut ? cut_provenance : src_provenance;
    const size_t input_face = size_t(from_cut ? face_map - src_face_count : face_map);
    if (input_face >= input_provenance.size() || !input_provenance[input_face].valid)
        return false;

    std::array<Vec3f, 3> input_vertices;
    if (!mcut_face_vertices(input_mesh, input_face, input_vertices))
        return false;

    const MeshFaceProvenance &input = input_provenance[input_face];
    out.valid = true;
    out.source_index = input.source_index;
    out.source_triangle = input.source_triangle;
    for (size_t corner = 0; corner < 3; ++corner) {
        const uint32_t global_vertex_idx = ccFaceIndices[size_t(face_offset) + corner];
        if (int(global_vertex_idx) < vertex_index_offset)
            return false;
        const size_t local_vertex_idx = size_t(int(global_vertex_idx) - vertex_index_offset);
        const size_t vertex_offset = local_vertex_idx * 3;
        if (vertex_offset + 2 >= ccVertices.size())
            return false;
        const Vec3f point = Vec3f(float(ccVertices[vertex_offset + 0]),
                                  float(ccVertices[vertex_offset + 1]),
                                  float(ccVertices[vertex_offset + 2]));
        out.source_barycentric[corner] = interpolate_source_barycentric(input, barycentric_weights_3d_mcut(point, input_vertices));
    }
    return true;
}

void triangle_mesh_to_mcut(const TriangleMesh &src_mesh, McutMesh &srcMesh, const Transform3d &src_nm = Transform3d::Identity())
{
    // vertices precision convention and copy
    srcMesh.vertexCoordsArray.reserve(src_mesh.its.vertices.size() * 3);
    for (int i = 0; i < src_mesh.its.vertices.size(); ++i) {
        const Vec3d v = src_nm * src_mesh.its.vertices[i].cast<double>();
        srcMesh.vertexCoordsArray.push_back(v[0]);
        srcMesh.vertexCoordsArray.push_back(v[1]);
        srcMesh.vertexCoordsArray.push_back(v[2]);
    }

    // faces copy
    srcMesh.faceIndicesArray.reserve(src_mesh.its.indices.size() * 3);
    srcMesh.faceSizesArray.reserve(src_mesh.its.indices.size());
    for (int i = 0; i < src_mesh.its.indices.size(); ++i) {
        const int &f0 = src_mesh.its.indices[i][0];
        const int &f1 = src_mesh.its.indices[i][1];
        const int &f2 = src_mesh.its.indices[i][2];
        srcMesh.faceIndicesArray.push_back(f0);
        srcMesh.faceIndicesArray.push_back(f1);
        srcMesh.faceIndicesArray.push_back(f2);

        srcMesh.faceSizesArray.push_back((uint32_t) 3);
    }
}

McutMeshPtr triangle_mesh_to_mcut(const indexed_triangle_set &M)
{
    std::unique_ptr<McutMesh, McutMeshDeleter> out(new McutMesh{});
    TriangleMesh                               trimesh(M);
    triangle_mesh_to_mcut(trimesh, *out.get());
    return out;
}

TriangleMesh mcut_to_triangle_mesh(const McutMesh &mcutmesh)
{
    uint32_t ccVertexCount = mcutmesh.vertexCoordsArray.size() / 3;
    auto    &ccVertices    = mcutmesh.vertexCoordsArray;
    auto    &ccFaceIndices = mcutmesh.faceIndicesArray;
    auto    &faceSizes     = mcutmesh.faceSizesArray;
    uint32_t ccFaceCount   = faceSizes.size();
    // rearrange vertices/faces and save into result mesh
    std::vector<Vec3f> vertices(ccVertexCount);
    for (uint32_t i = 0; i < ccVertexCount; i++) {
        vertices[i][0] = (float) ccVertices[(uint64_t) i * 3 + 0];
        vertices[i][1] = (float) ccVertices[(uint64_t) i * 3 + 1];
        vertices[i][2] = (float) ccVertices[(uint64_t) i * 3 + 2];
    }

    // output faces
    int faceVertexOffsetBase = 0;

    // for each face in CC
    std::vector<Vec3i32> faces(ccFaceCount);
    for (uint32_t f = 0; f < ccFaceCount; ++f) {
        int faceSize = faceSizes.at(f);

        // for each vertex in face
        for (int v = 0; v < faceSize; v++) { faces[f][v] = ccFaceIndices[(uint64_t) faceVertexOffsetBase + v]; }
        faceVertexOffsetBase += faceSize;
    }

    TriangleMesh out(vertices, faces);
    return out;
}

void merge_mcut_meshes(McutMesh& src, const McutMesh& cut) {
    indexed_triangle_set all_its;
    TriangleMesh tri_src = mcut_to_triangle_mesh(src);
    TriangleMesh tri_cut = mcut_to_triangle_mesh(cut);
    its_merge(all_its, tri_src.its);
    its_merge(all_its, tri_cut.its);
    src = *triangle_mesh_to_mcut(all_its);
    }

MCAPI_ATTR void MCAPI_CALL mcDebugOutput(McDebugSource source,
    McDebugType type,
    unsigned int id,
    McDebugSeverity severity,
    size_t length,
    const char* message,
    const void* userParam)
{
    BOOST_LOG_TRIVIAL(debug)<<Slic3r::format("mcut mcDebugOutput message ( %d ): %s ", id, message);

    switch (source) {
        case MC_DEBUG_SOURCE_API:
            BOOST_LOG_TRIVIAL(debug)<<("Source: API");
            break;
        case MC_DEBUG_SOURCE_KERNEL:
            BOOST_LOG_TRIVIAL(debug)<<("Source: Kernel");
            break;
        }

    switch (type) {
        case MC_DEBUG_TYPE_ERROR:
            BOOST_LOG_TRIVIAL(debug)<<("Type: Error");
            break;
        case MC_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
            BOOST_LOG_TRIVIAL(debug)<<("Type: Deprecated Behaviour");
            break;
        case MC_DEBUG_TYPE_OTHER:
            BOOST_LOG_TRIVIAL(debug)<<("Type: Other");
            break;
        }

    switch (severity) {
        case MC_DEBUG_SEVERITY_HIGH:
            BOOST_LOG_TRIVIAL(debug)<<("Severity: high");
            break;
        case MC_DEBUG_SEVERITY_MEDIUM:
            BOOST_LOG_TRIVIAL(debug)<<("Severity: medium");
            break;
        case MC_DEBUG_SEVERITY_LOW:
            BOOST_LOG_TRIVIAL(debug)<<("Severity: low");
            break;
        case MC_DEBUG_SEVERITY_NOTIFICATION:
            BOOST_LOG_TRIVIAL(debug)<<("Severity: notification");
            break;
        }
}


bool do_boolean_single(McutMesh &srcMesh, const McutMesh &cutMesh, const std::string &boolean_opts)
{
    // create context
    McContext context = MC_NULL_HANDLE;
    McResult  err     = mcCreateContext(&context, 0);
    // add debug callback according to https://cutdigital.github.io/mcut.site/tutorials/debugging/
    mcDebugMessageCallback(context, mcDebugOutput, nullptr);
    mcDebugMessageControl(
        context,
        MC_DEBUG_SOURCE_ALL,
        MC_DEBUG_TYPE_ERROR,
        MC_DEBUG_SEVERITY_MEDIUM,
        true);
    // We can either let MCUT compute all possible meshes (including patches etc.), or we can
    // constrain the library to compute exactly the boolean op mesh we want. This 'constrained' case
    // is done with the following flags.
    // NOTE#1: you can extend these flags by bitwise ORing with additional flags (see `McDispatchFlags' in mcut.h)
    // NOTE#2: below order of columns MATTERS
    const std::map<std::string, McFlags> booleanOpts = {
        {"A_NOT_B", MC_DISPATCH_FILTER_FRAGMENT_SEALING_INSIDE | MC_DISPATCH_FILTER_FRAGMENT_LOCATION_ABOVE},
        {"B_NOT_A", MC_DISPATCH_FILTER_FRAGMENT_SEALING_OUTSIDE | MC_DISPATCH_FILTER_FRAGMENT_LOCATION_BELOW},
        {"UNION", MC_DISPATCH_FILTER_FRAGMENT_SEALING_OUTSIDE | MC_DISPATCH_FILTER_FRAGMENT_LOCATION_ABOVE},
        {"INTERSECTION", MC_DISPATCH_FILTER_FRAGMENT_SEALING_INSIDE | MC_DISPATCH_FILTER_FRAGMENT_LOCATION_BELOW},
    };

    std::map<std::string, McFlags>::const_iterator it          = booleanOpts.find(boolean_opts);
    McFlags                                        boolOpFlags = it->second;

    if (srcMesh.vertexCoordsArray.empty() && (boolean_opts == "UNION" || boolean_opts == "B_NOT_A")) {
        srcMesh = cutMesh;
        mcReleaseContext(context);
        return true;
    }

    err = mcDispatch(context,
                     MC_DISPATCH_VERTEX_ARRAY_DOUBLE |          // vertices are in array of doubles
                         MC_DISPATCH_ENFORCE_GENERAL_POSITION | // perturb if necessary
                         boolOpFlags,                           // filter flags which specify the type of output we want
                     // source mesh
                     reinterpret_cast<const void *>(srcMesh.vertexCoordsArray.data()), reinterpret_cast<const uint32_t *>(srcMesh.faceIndicesArray.data()),
                     srcMesh.faceSizesArray.data(), static_cast<uint32_t>(srcMesh.vertexCoordsArray.size() / 3), static_cast<uint32_t>(srcMesh.faceSizesArray.size()),
                     // cut mesh
                     reinterpret_cast<const void *>(cutMesh.vertexCoordsArray.data()), cutMesh.faceIndicesArray.data(), cutMesh.faceSizesArray.data(),
                     static_cast<uint32_t>(cutMesh.vertexCoordsArray.size() / 3), static_cast<uint32_t>(cutMesh.faceSizesArray.size()));
    if (err != MC_NO_ERROR) {
        BOOST_LOG_TRIVIAL(debug) << "MCUT mcDispatch fails! err=" << err;
        mcReleaseContext(context);
        if (boolean_opts == "UNION") {
            merge_mcut_meshes(srcMesh, cutMesh);
            return true;
        }
        return false;
    }

    // query the number of available connected component
    uint32_t numConnComps;
    err = mcGetConnectedComponents(context, MC_CONNECTED_COMPONENT_TYPE_FRAGMENT, 0, NULL, &numConnComps);
    if (err != MC_NO_ERROR || numConnComps==0) {
        BOOST_LOG_TRIVIAL(debug) << "MCUT mcGetConnectedComponents fails! err=" << err << ", numConnComps" << numConnComps;
        mcReleaseContext(context);
        if (numConnComps == 0 && boolean_opts == "UNION") {
            merge_mcut_meshes(srcMesh, cutMesh);
            return true;
        }
        return false;
    }

    std::vector<McConnectedComponent> connectedComponents(numConnComps, MC_NULL_HANDLE);
    err = mcGetConnectedComponents(context, MC_CONNECTED_COMPONENT_TYPE_FRAGMENT, (uint32_t) connectedComponents.size(), connectedComponents.data(), NULL);

    McutMesh outMesh;
    int N_vertices = 0;
    // traversal of all connected components
    for (int n = 0; n < numConnComps; ++n) {
        // query the data of each connected component from MCUT
        McConnectedComponent connComp = connectedComponents[n];

        // query the vertices
        McSize numBytes                  = 0;
        err                               = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_VERTEX_DOUBLE, 0, NULL, &numBytes);
        uint32_t            ccVertexCount = (uint32_t) (numBytes / (sizeof(double) * 3));
        std::vector<double> ccVertices((uint64_t) ccVertexCount * 3u, 0);
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_VERTEX_DOUBLE, numBytes, (void *) ccVertices.data(), NULL);

        // query the faces
        numBytes = 0;
        err      = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_TRIANGULATION, 0, NULL, &numBytes);
        std::vector<uint32_t> ccFaceIndices(numBytes / sizeof(uint32_t), 0);
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_TRIANGULATION, numBytes, ccFaceIndices.data(), NULL);
        std::vector<uint32_t> faceSizes(ccFaceIndices.size() / 3, 3);

        const uint32_t ccFaceCount = static_cast<uint32_t>(faceSizes.size());

        // Here we show, how to know when connected components, pertain particular boolean operations.
        McPatchLocation patchLocation = (McPatchLocation) 0;
        err                           = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_PATCH_LOCATION, sizeof(McPatchLocation), &patchLocation, NULL);

        McFragmentLocation fragmentLocation = (McFragmentLocation) 0;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FRAGMENT_LOCATION, sizeof(McFragmentLocation), &fragmentLocation, NULL);

        outMesh.vertexCoordsArray.insert(outMesh.vertexCoordsArray.end(), ccVertices.begin(), ccVertices.end());

        // add offset to face index
        for (size_t i = 0; i < ccFaceIndices.size(); i++) {
            ccFaceIndices[i] += N_vertices;
        }

        int faceVertexOffsetBase = 0;

        // for each face in CC
        std::vector<Vec3i32> faces(ccFaceCount);
        for (uint32_t f = 0; f < ccFaceCount; ++f) {
            bool reverseWindingOrder = (fragmentLocation == MC_FRAGMENT_LOCATION_BELOW) && (patchLocation == MC_PATCH_LOCATION_OUTSIDE);
            int  faceSize            = faceSizes.at(f);
            if (reverseWindingOrder) {
                std::vector<uint32_t> faceIndex(faceSize);
                // for each vertex in face
                for (int v = faceSize - 1; v >= 0; v--) { faceIndex[v] = ccFaceIndices[(uint64_t) faceVertexOffsetBase + v]; }
                std::copy(faceIndex.begin(), faceIndex.end(), ccFaceIndices.begin() + faceVertexOffsetBase);
            }
            faceVertexOffsetBase += faceSize;
        }

        outMesh.faceIndicesArray.insert(outMesh.faceIndicesArray.end(), ccFaceIndices.begin(), ccFaceIndices.end());
        outMesh.faceSizesArray.insert(outMesh.faceSizesArray.end(), faceSizes.begin(), faceSizes.end());

        N_vertices += ccVertexCount;
    }

    // free connected component data
    err = mcReleaseConnectedComponents(context, 0, NULL);
    // destroy context
    err = mcReleaseContext(context);

    srcMesh = outMesh;

    return true;
}

static bool do_boolean_single_with_cgal_fallback(McutMesh &srcMesh, const McutMesh &cutMesh, const std::string &boolean_opts)
{
    McutMesh original = srcMesh;
    if (boolean_opts != "INTERSECTION" && do_boolean_single(srcMesh, cutMesh, boolean_opts))
        return true;

    TriangleMesh tri_src = mcut_to_triangle_mesh(original);
    TriangleMesh tri_cut = mcut_to_triangle_mesh(cutMesh);
    try {
        if (boolean_opts == "UNION")
            MeshBoolean::cgal::plus(tri_src, tri_cut);
        else if (boolean_opts == "A_NOT_B")
            MeshBoolean::cgal::minus(tri_src, tri_cut);
        else if (boolean_opts == "INTERSECTION")
            MeshBoolean::cgal::intersect(tri_src, tri_cut);
        else {
            srcMesh = std::move(original);
            return false;
        }
    } catch (...) {
        srcMesh = std::move(original);
        return false;
    }

    srcMesh = *triangle_mesh_to_mcut(tri_src.its);
    return true;
}

static McutBooleanStatus do_boolean_single_with_igl_provenance(McutMesh &srcMesh,
                                                               std::vector<MeshFaceProvenance> &src_provenance,
                                                               const McutMesh &cutMesh,
                                                               const std::vector<MeshFaceProvenance> &cut_provenance,
                                                               const std::string &boolean_opts)
{
    TriangleMesh tri_src = mcut_to_triangle_mesh(srcMesh);
    TriangleMesh tri_cut = mcut_to_triangle_mesh(cutMesh);
    if (tri_src.its.indices.size() != src_provenance.size() || tri_cut.its.indices.size() != cut_provenance.size())
        return McutBooleanStatus::Failure;
    if (tri_src.empty()) {
        if (boolean_opts == "UNION") {
            srcMesh = cutMesh;
            src_provenance = cut_provenance;
        } else
            src_provenance.clear();
        return McutBooleanStatus::Success;
    }
    if (tri_cut.empty()) {
        if (boolean_opts == "INTERSECTION") {
            srcMesh = *triangle_mesh_to_mcut(indexed_triangle_set{});
            src_provenance.clear();
        }
        return McutBooleanStatus::Success;
    }

    igl::MeshBooleanType boolean_type;
    if (boolean_opts == "UNION")
        boolean_type = igl::MESH_BOOLEAN_TYPE_UNION;
    else if (boolean_opts == "A_NOT_B")
        boolean_type = igl::MESH_BOOLEAN_TYPE_MINUS;
    else if (boolean_opts == "INTERSECTION")
        boolean_type = igl::MESH_BOOLEAN_TYPE_INTERSECT;
    else
        return McutBooleanStatus::Failure;

    EigenMesh src_eigen = triangle_mesh_to_eigen(tri_src);
    EigenMesh cut_eigen = triangle_mesh_to_eigen(tri_cut);
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;
    Eigen::VectorXi face_birth;
    try {
        igl::copyleft::cgal::mesh_boolean(src_eigen.first,
                                           src_eigen.second,
                                           cut_eigen.first,
                                           cut_eigen.second,
                                           boolean_type,
                                           vertices,
                                           faces,
                                           face_birth);
    } catch (...) {
        return McutBooleanStatus::Failure;
    }

    if (faces.rows() != face_birth.rows())
        return McutBooleanStatus::Failure;

    TriangleMesh out_mesh = eigen_to_triangle_mesh({ std::move(vertices), std::move(faces) });
    std::vector<MeshFaceProvenance> out_provenance;
    out_provenance.reserve(out_mesh.its.indices.size());
    const int src_face_count = int(tri_src.its.indices.size());
    for (size_t face_idx = 0; face_idx < out_mesh.its.indices.size(); ++face_idx) {
        const int birth = face_birth[int(face_idx)];
        if (birth < 0)
            return McutBooleanStatus::Failure;
        const bool from_cut = birth >= src_face_count;
        const indexed_triangle_set &input_its = from_cut ? tri_cut.its : tri_src.its;
        const std::vector<MeshFaceProvenance> &input_provenance = from_cut ? cut_provenance : src_provenance;
        const size_t input_face = size_t(from_cut ? birth - src_face_count : birth);
        if (input_face >= input_its.indices.size() || input_face >= input_provenance.size() || !input_provenance[input_face].valid)
            return McutBooleanStatus::Failure;

        const Vec3i32 &input_face_indices = input_its.indices[input_face];
        std::array<Vec3f, 3> input_vertices = {
            input_its.vertices[size_t(input_face_indices.x())],
            input_its.vertices[size_t(input_face_indices.y())],
            input_its.vertices[size_t(input_face_indices.z())]
        };
        const MeshFaceProvenance &input = input_provenance[input_face];
        MeshFaceProvenance output;
        output.valid = true;
        output.source_index = input.source_index;
        output.source_triangle = input.source_triangle;
        const Vec3i32 &output_face_indices = out_mesh.its.indices[face_idx];
        for (size_t corner = 0; corner < 3; ++corner) {
            const Vec3f &point = out_mesh.its.vertices[size_t(output_face_indices[int(corner)])];
            output.source_barycentric[corner] = interpolate_source_barycentric(input, barycentric_weights_3d_mcut(point, input_vertices));
        }
        out_provenance.emplace_back(output);
    }

    srcMesh = *triangle_mesh_to_mcut(out_mesh.its);
    src_provenance = std::move(out_provenance);
    return McutBooleanStatus::Success;
}

static McutBooleanStatus do_boolean_single_with_provenance(McutMesh &srcMesh,
                                                           std::vector<MeshFaceProvenance> &src_provenance,
                                                           const McutMesh &cutMesh,
                                                           const std::vector<MeshFaceProvenance> &cut_provenance,
                                                           const std::string &boolean_opts)
{
    if (src_provenance.size() != srcMesh.faceSizesArray.size() || cut_provenance.size() != cutMesh.faceSizesArray.size())
        return McutBooleanStatus::Failure;
    if (boolean_opts == "INTERSECTION")
        return do_boolean_single_with_igl_provenance(srcMesh, src_provenance, cutMesh, cut_provenance, boolean_opts);

    McContext context = MC_NULL_HANDLE;
    McResult err = mcCreateContext(&context, 0);
    if (err != MC_NO_ERROR)
        return McutBooleanStatus::Failure;
    mcDebugMessageCallback(context, mcDebugOutput, nullptr);
    mcDebugMessageControl(context, MC_DEBUG_SOURCE_ALL, MC_DEBUG_TYPE_ERROR, MC_DEBUG_SEVERITY_MEDIUM, true);

    const std::map<std::string, McFlags> booleanOpts = {
        {"A_NOT_B", MC_DISPATCH_FILTER_FRAGMENT_SEALING_INSIDE | MC_DISPATCH_FILTER_FRAGMENT_LOCATION_ABOVE},
        {"B_NOT_A", MC_DISPATCH_FILTER_FRAGMENT_SEALING_OUTSIDE | MC_DISPATCH_FILTER_FRAGMENT_LOCATION_BELOW},
        {"UNION", MC_DISPATCH_FILTER_FRAGMENT_SEALING_OUTSIDE | MC_DISPATCH_FILTER_FRAGMENT_LOCATION_ABOVE},
        {"INTERSECTION", MC_DISPATCH_FILTER_FRAGMENT_SEALING_INSIDE | MC_DISPATCH_FILTER_FRAGMENT_LOCATION_BELOW},
    };

    auto it = booleanOpts.find(boolean_opts);
    if (it == booleanOpts.end()) {
        mcReleaseContext(context);
        return McutBooleanStatus::Failure;
    }

    if (srcMesh.vertexCoordsArray.empty() && (boolean_opts == "UNION" || boolean_opts == "B_NOT_A")) {
        srcMesh = cutMesh;
        src_provenance = cut_provenance;
        mcReleaseContext(context);
        return McutBooleanStatus::Success;
    }

    err = mcDispatch(context,
                     MC_DISPATCH_VERTEX_ARRAY_DOUBLE |
                         MC_DISPATCH_ENFORCE_GENERAL_POSITION |
                         MC_DISPATCH_INCLUDE_FACE_MAP |
                         it->second,
                     reinterpret_cast<const void *>(srcMesh.vertexCoordsArray.data()),
                     reinterpret_cast<const uint32_t *>(srcMesh.faceIndicesArray.data()),
                     srcMesh.faceSizesArray.data(),
                     static_cast<uint32_t>(srcMesh.vertexCoordsArray.size() / 3),
                     static_cast<uint32_t>(srcMesh.faceSizesArray.size()),
                     reinterpret_cast<const void *>(cutMesh.vertexCoordsArray.data()),
                     cutMesh.faceIndicesArray.data(),
                     cutMesh.faceSizesArray.data(),
                     static_cast<uint32_t>(cutMesh.vertexCoordsArray.size() / 3),
                     static_cast<uint32_t>(cutMesh.faceSizesArray.size()));
    if (err != MC_NO_ERROR) {
        BOOST_LOG_TRIVIAL(debug) << "MCUT mcDispatch provenance fails! err=" << err;
        mcReleaseContext(context);
        if (boolean_opts == "UNION") {
            merge_mcut_meshes_with_provenance(srcMesh, src_provenance, cutMesh, cut_provenance);
            return McutBooleanStatus::Success;
        }
        return do_boolean_single_with_igl_provenance(srcMesh, src_provenance, cutMesh, cut_provenance, boolean_opts);
    }

    uint32_t numConnComps = 0;
    err = mcGetConnectedComponents(context, MC_CONNECTED_COMPONENT_TYPE_FRAGMENT, 0, NULL, &numConnComps);
    if (err != MC_NO_ERROR || numConnComps == 0) {
        BOOST_LOG_TRIVIAL(debug) << "MCUT mcGetConnectedComponents provenance fails! err=" << err << ", numConnComps" << numConnComps;
        mcReleaseContext(context);
        if (numConnComps == 0 && boolean_opts == "UNION") {
            merge_mcut_meshes_with_provenance(srcMesh, src_provenance, cutMesh, cut_provenance);
            return McutBooleanStatus::Success;
        }
        return do_boolean_single_with_igl_provenance(srcMesh, src_provenance, cutMesh, cut_provenance, boolean_opts);
    }

    std::vector<McConnectedComponent> connectedComponents(numConnComps, MC_NULL_HANDLE);
    err = mcGetConnectedComponents(context, MC_CONNECTED_COMPONENT_TYPE_FRAGMENT, uint32_t(connectedComponents.size()), connectedComponents.data(), NULL);
    if (err != MC_NO_ERROR) {
        mcReleaseContext(context);
        return do_boolean_single_with_igl_provenance(srcMesh, src_provenance, cutMesh, cut_provenance, boolean_opts);
    }

    McutMesh outMesh;
    std::vector<MeshFaceProvenance> out_provenance;
    int vertex_index_offset = 0;
    bool ok = true;
    for (int n = 0; n < int(numConnComps) && ok; ++n) {
        McConnectedComponent connComp = connectedComponents[n];

        McSize numBytes = 0;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_VERTEX_DOUBLE, 0, NULL, &numBytes);
        if (err != MC_NO_ERROR) {
            ok = false;
            break;
        }
        uint32_t ccVertexCount = uint32_t(numBytes / (sizeof(double) * 3));
        std::vector<double> ccVertices(size_t(ccVertexCount) * 3u, 0);
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_VERTEX_DOUBLE, numBytes, ccVertices.data(), NULL);
        if (err != MC_NO_ERROR) {
            ok = false;
            break;
        }

        numBytes = 0;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_TRIANGULATION, 0, NULL, &numBytes);
        if (err != MC_NO_ERROR) {
            ok = false;
            break;
        }
        std::vector<uint32_t> ccFaceIndices(numBytes / sizeof(uint32_t), 0);
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_TRIANGULATION, numBytes, ccFaceIndices.data(), NULL);
        if (err != MC_NO_ERROR) {
            ok = false;
            break;
        }
        std::vector<uint32_t> faceSizes(ccFaceIndices.size() / 3, 3);
        const uint32_t ccFaceCount = uint32_t(faceSizes.size());

        numBytes = 0;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_TRIANGULATION_MAP, 0, NULL, &numBytes);
        if (err != MC_NO_ERROR) {
            ok = false;
            break;
        }
        std::vector<uint32_t> ccFaceMap(numBytes / sizeof(uint32_t), 0);
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FACE_TRIANGULATION_MAP, numBytes, ccFaceMap.data(), NULL);
        if (err != MC_NO_ERROR || ccFaceMap.size() != ccFaceCount) {
            ok = false;
            break;
        }

        McPatchLocation patchLocation = (McPatchLocation) 0;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_PATCH_LOCATION, sizeof(McPatchLocation), &patchLocation, NULL);
        if (err != MC_NO_ERROR) {
            ok = false;
            break;
        }
        McFragmentLocation fragmentLocation = (McFragmentLocation) 0;
        err = mcGetConnectedComponentData(context, connComp, MC_CONNECTED_COMPONENT_DATA_FRAGMENT_LOCATION, sizeof(McFragmentLocation), &fragmentLocation, NULL);
        if (err != MC_NO_ERROR) {
            ok = false;
            break;
        }

        outMesh.vertexCoordsArray.insert(outMesh.vertexCoordsArray.end(), ccVertices.begin(), ccVertices.end());
        for (size_t i = 0; i < ccFaceIndices.size(); ++i)
            ccFaceIndices[i] += uint32_t(vertex_index_offset);

        int faceVertexOffsetBase = 0;
        for (uint32_t f = 0; f < ccFaceCount; ++f) {
            const bool reverseWindingOrder = (fragmentLocation == MC_FRAGMENT_LOCATION_BELOW) && (patchLocation == MC_PATCH_LOCATION_OUTSIDE);
            const int faceSize = int(faceSizes[f]);
            if (faceSize != 3) {
                ok = false;
                break;
            }
            if (reverseWindingOrder) {
                std::vector<uint32_t> faceIndex(static_cast<size_t>(faceSize));
                for (int v = faceSize - 1; v >= 0; --v)
                    faceIndex[size_t(v)] = ccFaceIndices[size_t(faceVertexOffsetBase) + size_t(v)];
                std::copy(faceIndex.begin(), faceIndex.end(), ccFaceIndices.begin() + faceVertexOffsetBase);
            }
            MeshFaceProvenance face_provenance;
            if (!mapped_output_face_provenance(srcMesh,
                                               src_provenance,
                                               cutMesh,
                                               cut_provenance,
                                               ccFaceMap[f],
                                               ccVertices,
                                               ccFaceIndices,
                                               uint32_t(faceVertexOffsetBase),
                                               vertex_index_offset,
                                               face_provenance)) {
                ok = false;
                break;
            }
            out_provenance.emplace_back(face_provenance);
            faceVertexOffsetBase += faceSize;
        }
        if (!ok)
            break;

        outMesh.faceIndicesArray.insert(outMesh.faceIndicesArray.end(), ccFaceIndices.begin(), ccFaceIndices.end());
        outMesh.faceSizesArray.insert(outMesh.faceSizesArray.end(), faceSizes.begin(), faceSizes.end());
        vertex_index_offset += int(ccVertexCount);
    }

    mcReleaseConnectedComponents(context, 0, NULL);
    mcReleaseContext(context);
    if (!ok || out_provenance.size() != outMesh.faceSizesArray.size())
        return do_boolean_single_with_igl_provenance(srcMesh, src_provenance, cutMesh, cut_provenance, boolean_opts);

    srcMesh = std::move(outMesh);
    src_provenance = std::move(out_provenance);
    return McutBooleanStatus::Success;
}

void do_boolean(McutMesh& srcMesh, const McutMesh& cutMesh, const std::string& boolean_opts)
{
    TriangleMesh tri_src = mcut_to_triangle_mesh(srcMesh);
    std::vector<indexed_triangle_set> src_parts = its_split(tri_src.its);

    TriangleMesh tri_cut = mcut_to_triangle_mesh(cutMesh);
    std::vector<indexed_triangle_set> cut_parts = its_split(tri_cut.its);

    if (src_parts.empty() && boolean_opts == "UNION") {
        srcMesh = cutMesh;
        return;
    }
    if(cut_parts.empty()) return;

    // when src mesh has multiple connected components, mcut refuses to work.
    // But we can force it to work by spliting the src mesh into disconnected components,
    // and do booleans seperately, then merge all the results.
    indexed_triangle_set all_its;
    bool ok = true;
    if (boolean_opts == "UNION" || boolean_opts == "A_NOT_B") {
        for (size_t i = 0; i < src_parts.size() && ok; i++) {
            auto src_part = triangle_mesh_to_mcut(src_parts[i]);
            for (size_t j = 0; j < cut_parts.size(); j++) {
                if (boolean_opts == "A_NOT_B") {
                    TriangleMesh current_src_part = mcut_to_triangle_mesh(*src_part);
                    if (!its_bounds_overlap_mcut(current_src_part.its, cut_parts[j]))
                        continue;
                }
                auto cut_part = triangle_mesh_to_mcut(cut_parts[j]);
                if (!do_boolean_single_with_cgal_fallback(*src_part, *cut_part, boolean_opts)) {
                    ok = false;
                    break;
                }
            }
            if (!ok)
                break;
            TriangleMesh tri_part = mcut_to_triangle_mesh(*src_part);
            its_merge(all_its, tri_part.its);
        }
    }
    else if (boolean_opts == "INTERSECTION") {
        for (size_t i = 0; i < src_parts.size() && ok; i++) {
            for (size_t j = 0; j < cut_parts.size(); j++) {
                if (!its_bounds_overlap_mcut(src_parts[i], cut_parts[j]))
                    continue;
                auto src_part = triangle_mesh_to_mcut(src_parts[i]);
                auto cut_part = triangle_mesh_to_mcut(cut_parts[j]);
                if (!do_boolean_single_with_cgal_fallback(*src_part, *cut_part, boolean_opts)) {
                    ok = false;
                    break;
                }
                TriangleMesh tri_part = mcut_to_triangle_mesh(*src_part);
                its_merge(all_its, tri_part.its);
            }
        }
    }
    if (!ok) {
        srcMesh = *triangle_mesh_to_mcut(indexed_triangle_set{});
        return;
    }
    srcMesh = *triangle_mesh_to_mcut(all_its);
}

bool do_boolean_with_provenance(McutMesh &srcMesh,
                                std::vector<MeshFaceProvenance> &src_provenance,
                                const McutMesh &cutMesh,
                                const std::vector<MeshFaceProvenance> &cut_provenance,
                                const std::string &boolean_opts)
{
    TriangleMesh tri_src = mcut_to_triangle_mesh(srcMesh);
    TriangleMesh tri_cut = mcut_to_triangle_mesh(cutMesh);
    if (src_provenance.size() != tri_src.its.indices.size() || cut_provenance.size() != tri_cut.its.indices.size())
        return false;

    std::vector<ProvenancedIts> src_parts = split_with_provenance(tri_src.its, src_provenance);
    std::vector<ProvenancedIts> cut_parts = split_with_provenance(tri_cut.its, cut_provenance);

    if (src_parts.empty() && boolean_opts == "UNION") {
        srcMesh = cutMesh;
        src_provenance = cut_provenance;
        return true;
    }
    if (cut_parts.empty())
        return true;

    indexed_triangle_set all_its;
    std::vector<MeshFaceProvenance> all_provenance;
    if (boolean_opts == "UNION" || boolean_opts == "A_NOT_B") {
        for (size_t i = 0; i < src_parts.size(); ++i) {
            auto src_part = triangle_mesh_to_mcut(src_parts[i].its);
            std::vector<MeshFaceProvenance> src_part_provenance = src_parts[i].provenance;
            for (size_t j = 0; j < cut_parts.size(); ++j) {
                if (boolean_opts == "A_NOT_B") {
                    TriangleMesh current_src_part = mcut_to_triangle_mesh(*src_part);
                    if (!its_bounds_overlap_mcut(current_src_part.its, cut_parts[j].its))
                        continue;
                }
                auto cut_part = triangle_mesh_to_mcut(cut_parts[j].its);
                const McutBooleanStatus status = do_boolean_single_with_provenance(*src_part, src_part_provenance, *cut_part, cut_parts[j].provenance, boolean_opts);
                if (status == McutBooleanStatus::Failure)
                    return false;
                if (status == McutBooleanStatus::NoOutput && boolean_opts != "A_NOT_B")
                    return false;
            }
            TriangleMesh tri_part = mcut_to_triangle_mesh(*src_part);
            if (tri_part.its.indices.size() != src_part_provenance.size())
                return false;
            append_with_provenance(all_its, all_provenance, tri_part.its, src_part_provenance);
        }
    } else if (boolean_opts == "INTERSECTION") {
        for (size_t i = 0; i < src_parts.size(); ++i) {
            for (size_t j = 0; j < cut_parts.size(); ++j) {
                if (!its_bounds_overlap_mcut(src_parts[i].its, cut_parts[j].its))
                    continue;
                auto src_part = triangle_mesh_to_mcut(src_parts[i].its);
                auto cut_part = triangle_mesh_to_mcut(cut_parts[j].its);
                std::vector<MeshFaceProvenance> src_part_provenance = src_parts[i].provenance;
                const McutBooleanStatus status = do_boolean_single_with_provenance(*src_part, src_part_provenance, *cut_part, cut_parts[j].provenance, boolean_opts);
                if (status == McutBooleanStatus::Failure)
                    return false;
                if (status == McutBooleanStatus::NoOutput)
                    continue;
                TriangleMesh tri_part = mcut_to_triangle_mesh(*src_part);
                if (tri_part.its.indices.size() != src_part_provenance.size())
                    return false;
                append_with_provenance(all_its, all_provenance, tri_part.its, src_part_provenance);
            }
        }
    } else
        return false;

    srcMesh = *triangle_mesh_to_mcut(all_its);
    src_provenance = std::move(all_provenance);
    return true;
}

void make_boolean(const TriangleMesh &src_mesh, const TriangleMesh &cut_mesh, std::vector<TriangleMesh> &dst_mesh, const std::string &boolean_opts)
{
    McutMesh srcMesh, cutMesh;
    triangle_mesh_to_mcut(src_mesh, srcMesh);
    triangle_mesh_to_mcut(cut_mesh, cutMesh);
    //dst_mesh = make_boolean(srcMesh, cutMesh, boolean_opts);
    do_boolean(srcMesh, cutMesh, boolean_opts);
    TriangleMesh tri_src = mcut_to_triangle_mesh(srcMesh);
    if (!tri_src.empty())
        dst_mesh.push_back(std::move(tri_src));
}

bool make_boolean_with_provenance(const TriangleMesh &src_mesh,
                                  size_t src_source_index,
                                  const TriangleMesh &cut_mesh,
                                  size_t cut_source_index,
                                  std::vector<ProvenancedMesh> &dst_mesh,
                                  const std::string &boolean_opts)
{
    McutMesh srcMesh, cutMesh;
    triangle_mesh_to_mcut(src_mesh, srcMesh);
    triangle_mesh_to_mcut(cut_mesh, cutMesh);
    std::vector<MeshFaceProvenance> src_provenance = identity_provenance(src_mesh.its, src_source_index);
    std::vector<MeshFaceProvenance> cut_provenance = identity_provenance(cut_mesh.its, cut_source_index);

    if (!do_boolean_with_provenance(srcMesh, src_provenance, cutMesh, cut_provenance, boolean_opts))
        return false;

    TriangleMesh tri_src = mcut_to_triangle_mesh(srcMesh);
    if (!tri_src.empty()) {
        if (tri_src.its.indices.size() != src_provenance.size())
            return false;
        dst_mesh.push_back({ std::move(tri_src), std::move(src_provenance) });
    }
    return true;
}

} // namespace mcut


} // namespace MeshBoolean
} // namespace Slic3r
