#ifndef libslic3r_MeshBoolean_hpp_
#define libslic3r_MeshBoolean_hpp_

#include <array>
#include <memory>
#include <exception>
#include <vector>

#include <libslic3r/TriangleMesh.hpp>
#include <Eigen/Geometry>

namespace Slic3r {

namespace MeshBoolean {

using EigenMesh = std::pair<Eigen::MatrixXd, Eigen::MatrixXi>;

TriangleMesh eigen_to_triangle_mesh(const EigenMesh &emesh);
EigenMesh triangle_mesh_to_eigen(const TriangleMesh &mesh);

void minus(EigenMesh &A, const EigenMesh &B);
void self_union(EigenMesh &A);
    
void minus(TriangleMesh& A, const TriangleMesh& B);
void self_union(TriangleMesh& mesh);

namespace cgal {

struct CGALMesh;
struct CGALMeshDeleter { void operator()(CGALMesh *ptr); };
using CGALMeshPtr = std::unique_ptr<CGALMesh, CGALMeshDeleter>;

CGALMeshPtr clone(const CGALMesh &m);

void save_CGALMesh(const std::string& fname, const CGALMesh& cgal_mesh);

CGALMeshPtr triangle_mesh_to_cgal(
    const std::vector<stl_vertex> &V,
    const std::vector<stl_triangle_vertex_indices> &F);

inline CGALMeshPtr triangle_mesh_to_cgal(const indexed_triangle_set &M)
{
    return triangle_mesh_to_cgal(M.vertices, M.indices);
}
inline CGALMeshPtr triangle_mesh_to_cgal(const TriangleMesh &M)
{
    return triangle_mesh_to_cgal(M.its);
}

TriangleMesh cgal_to_triangle_mesh(const CGALMesh &cgalmesh);
indexed_triangle_set cgal_to_indexed_triangle_set(const CGALMesh &cgalmesh);

// Do boolean mesh difference with CGAL bypassing igl.
void minus(TriangleMesh &A, const TriangleMesh &B);
void plus(TriangleMesh &A, const TriangleMesh &B);
void intersect(TriangleMesh &A, const TriangleMesh &B);

void minus(indexed_triangle_set &A, const indexed_triangle_set &B);
void plus(indexed_triangle_set &A, const indexed_triangle_set &B);
void intersect(indexed_triangle_set &A, const indexed_triangle_set &B);

void minus(CGALMesh &A, CGALMesh &B);
void plus(CGALMesh &A, CGALMesh &B);
void intersect(CGALMesh &A, CGALMesh &B);

bool does_self_intersect(const TriangleMesh &mesh);
bool does_self_intersect(const CGALMesh &mesh);

//BBS
std::vector<TriangleMesh> segment(const TriangleMesh& src, double smoothing_alpha = 0.5, int segment_number = 5);
TriangleMesh merge(std::vector<TriangleMesh> meshes);

bool does_bound_a_volume(const CGALMesh &mesh);
bool empty(const CGALMesh &mesh);

// Repair a mesh using CGAL. Returns true on success. Optionally returns a summary of repairs and an error string.
bool repair(TriangleMesh &mesh, RepairedMeshErrors *repaired_errors = nullptr, std::string *error = nullptr);
}

namespace mcut {
struct McutMesh;
struct McutMeshDeleter
{
    void operator()(McutMesh *ptr);
};
using McutMeshPtr = std::unique_ptr<McutMesh, McutMeshDeleter>;
bool empty(const McutMesh &mesh);

struct MeshFaceProvenance
{
    bool valid { false };
    size_t source_index { 0 };
    size_t source_triangle { 0 };
    std::array<Vec3f, 3> source_barycentric;
};

struct ProvenancedMesh
{
    TriangleMesh mesh;
    std::vector<MeshFaceProvenance> provenance;
};

McutMeshPtr  triangle_mesh_to_mcut(const indexed_triangle_set &M);
TriangleMesh mcut_to_triangle_mesh(const McutMesh &mcutmesh);
std::vector<MeshFaceProvenance> identity_provenance(const indexed_triangle_set &its, size_t source_index);

// do boolean and save result to srcMesh
// return true if sucessful
bool do_boolean_single(McutMesh& srcMesh, const McutMesh& cutMesh, const std::string& boolean_opts);
// do boolean of mesh with multiple volumes and save result to srcMesh
// Both srcMesh and cutMesh may have multiple volumes.
void do_boolean(McutMesh &srcMesh, const McutMesh &cutMesh, const std::string &boolean_opts);
bool do_boolean_with_provenance(McutMesh                         &srcMesh,
                                std::vector<MeshFaceProvenance>  &src_provenance,
                                const McutMesh                   &cutMesh,
                                const std::vector<MeshFaceProvenance> &cut_provenance,
                                const std::string                &boolean_opts);


// do boolean and convert result to TriangleMesh
void make_boolean(const TriangleMesh &src_mesh, const TriangleMesh &cut_mesh, std::vector<TriangleMesh> &dst_mesh, const std::string &boolean_opts);
bool make_boolean_with_provenance(const TriangleMesh          &src_mesh,
                                  size_t                       src_source_index,
                                  const TriangleMesh          &cut_mesh,
                                  size_t                       cut_source_index,
                                  std::vector<ProvenancedMesh> &dst_mesh,
                                  const std::string           &boolean_opts);
} // namespace mcut

} // namespace MeshBoolean
} // namespace Slic3r
#endif // libslic3r_MeshBoolean_hpp_
