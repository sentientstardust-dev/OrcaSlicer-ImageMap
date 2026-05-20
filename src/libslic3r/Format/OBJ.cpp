#include "../libslic3r.h"
#include "../Model.hpp"
#include "../TriangleMesh.hpp"

#include "OBJ.hpp"
#include "objparser.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>
#include <earcut.hpp>

#ifdef _WIN32
#define DIR_SEPARATOR '\\'
#else
#define DIR_SEPARATOR '/'
#endif

//Translation
#include "I18N.hpp"
#define _L(s) Slic3r::I18N::translate(s)

namespace Slic3r {

bool load_obj(const char *path, TriangleMesh *meshptr, ObjInfo& obj_info, std::string &message, ObjTriangulationFn objTriangulationFn)
{
    if (meshptr == nullptr)
        return false;

    ObjParser::ObjData data;
    ObjParser::MtlData mtl_data;
    if (! ObjParser::objparse(path, data)) {
        BOOST_LOG_TRIVIAL(error) << "load_obj: failed to parse " << path;
        message = _L("load_obj: failed to parse");
        return false;
    }
    bool exist_mtl = false;
    boost::filesystem::path obj_path(path);
    const boost::filesystem::path obj_dir = obj_path.parent_path();
    obj_info.obj_dircetory = obj_dir.string();
    std::vector<std::string> mtl_warnings;
    if (data.mtllibs.size() > 0) { // read mtl
        for (auto mtl_name : data.mtllibs) {
            if (mtl_name.size() == 0){
                continue;
            }
            exist_mtl = true;

            const boost::filesystem::path raw_mtl_path(mtl_name);
            boost::filesystem::path resolved_mtl_path;
            if (raw_mtl_path.is_absolute()) {
                if (boost::filesystem::exists(raw_mtl_path))
                    resolved_mtl_path = raw_mtl_path;
                else
                    resolved_mtl_path = obj_dir / raw_mtl_path.filename();
            } else {
                const boost::filesystem::path relative_path = obj_dir / raw_mtl_path;
                if (boost::filesystem::exists(relative_path))
                    resolved_mtl_path = relative_path;
                else
                    resolved_mtl_path = obj_dir / raw_mtl_path.filename();
            }

            const std::string resolved_mtl_path_str = resolved_mtl_path.string();
            if (boost::filesystem::exists(resolved_mtl_path)) {
                if (!ObjParser::mtlparse(resolved_mtl_path_str.c_str(), mtl_data)) {
                    BOOST_LOG_TRIVIAL(error) << "load_obj:load_mtl: failed to parse " << resolved_mtl_path_str;
                    mtl_warnings.emplace_back(resolved_mtl_path_str);
                }
            }
            else {
                BOOST_LOG_TRIVIAL(error) << "load_obj: failed to load mtl_path:" << resolved_mtl_path_str;
                mtl_warnings.emplace_back(resolved_mtl_path_str);
            }
        }

        if (!mtl_warnings.empty())
            message = _L("load mtl in obj: failed to parse or load; importing model without some material/texture data");
    }

    size_t num_triangles = 0;
    obj_info.triangulation_info = ObjTriangulationInfo();
    for (size_t i = 0; i < data.vertices.size(); ++ i) {
        size_t j = i;
        for (; j < data.vertices.size() && data.vertices[j].coordIdx != -1; ++ j) ;
        if (size_t num_face_vertices = j - i; num_face_vertices > 0) {
            if (num_face_vertices < 3) {
                BOOST_LOG_TRIVIAL(error) << "load_obj: failed to parse " << path
                                         << ". The file contains polygons with less than 3 vertices.";
                message = _L("The file contains polygons with less than 3 vertices.");
                return false;
            }
            ++obj_info.triangulation_info.face_count;
            obj_info.triangulation_info.max_face_vertices = std::max(obj_info.triangulation_info.max_face_vertices, num_face_vertices);
            if (num_face_vertices > 3)
                ++obj_info.triangulation_info.non_triangular_face_count;
            if (num_face_vertices > 4)
                ++obj_info.triangulation_info.complex_polygon_face_count;
            num_triangles += num_face_vertices - 2;
            i = j;
        }
    }
    obj_info.triangulation_info.generated_triangle_count = num_triangles;
    if (obj_info.triangulation_info.complex_polygon_face_count > 0 &&
        objTriangulationFn &&
        !objTriangulationFn(obj_info.triangulation_info)) {
        BOOST_LOG_TRIVIAL(info) << "load_obj: canceled auto-triangulation of " << path;
        message = _L("OBJ import was canceled.");
        return false;
    }

    indexed_triangle_set its;
    size_t               num_vertices = data.coordinates.size() / OBJ_VERTEX_LENGTH;
    its.vertices.reserve(num_vertices);
    its.indices.reserve(num_triangles);
    obj_info.triangle_uvs.reserve(num_triangles);
    obj_info.triangle_uvs_valid.reserve(num_triangles);
    if (exist_mtl) {
        obj_info.is_single_mtl = data.usemtls.size() == 1 && mtl_data.new_mtl_unmap.size() == 1;
        obj_info.face_colors.reserve(num_triangles);
    }
    for (size_t i = 0; i < num_vertices; ++ i) {
        size_t j = i * OBJ_VERTEX_LENGTH;
        its.vertices.emplace_back(data.coordinates[j], data.coordinates[j + 1], data.coordinates[j + 2]);
        if (data.has_vertex_color) {
            RGBA color{std::clamp(data.coordinates[j + 3], 0.f, 1.f),
                       std::clamp(data.coordinates[j + 4], 0.f, 1.f),
                       std::clamp(data.coordinates[j + 5], 0.f, 1.f),
                       std::clamp(data.coordinates[j + 6], 0.f, 1.f)};
            obj_info.vertex_colors.emplace_back(color);
        }
    }
    auto read_uv = [&data](int uv_idx, Vec2f &out_uv) {
        if (uv_idx < 0)
            return false;
        const size_t off = size_t(uv_idx) * 2;
        if (off + 1 >= data.textureCoordinates.size())
            return false;
        out_uv = Vec2f(data.textureCoordinates[off], data.textureCoordinates[off + 1]);
        return true;
    };
    auto append_triangle_uv = [&obj_info, &read_uv](int uv0_idx, int uv1_idx, int uv2_idx) {
        std::array<Vec2f, 3> triangle_uv{Vec2f::Zero(), Vec2f::Zero(), Vec2f::Zero()};
        const bool has_all_uv =
            read_uv(uv0_idx, triangle_uv[0]) &&
            read_uv(uv1_idx, triangle_uv[1]) &&
            read_uv(uv2_idx, triangle_uv[2]);
        obj_info.triangle_uvs.emplace_back(triangle_uv);
        obj_info.triangle_uvs_valid.emplace_back(uint8_t(has_all_uv ? 1 : 0));
    };
    auto set_face_color = [&data, &mtl_data, &obj_info, &read_uv]
                          (int face_index, const std::string &mtl_name, const std::array<int, 3> &triangle_uv_indices) {
        const auto material_it = mtl_data.new_mtl_unmap.find(mtl_name);
        if (material_it != mtl_data.new_mtl_unmap.end() && material_it->second) {
            const auto &material = *material_it->second;
            RGBA face_color;
            bool is_merge_ka_kd = true;
            for (size_t n = 0; n < 3; n++) {
                if (float(material.Ka[n] + material.Kd[n]) > 1.0) {
                    is_merge_ka_kd=false;
                    break;
                }
            }
            for (size_t n = 0; n < 3; n++) {
                if (is_merge_ka_kd) {
                    face_color[n] = std::clamp(float(material.Ka[n] + material.Kd[n]), 0.f, 1.f);
                }
                else {
                    face_color[n] = std::clamp(float(material.Kd[n]), 0.f, 1.f);
                }
            }
            face_color[3] = material.Tr;
            if (!material.map_Kd.empty()) {
                const std::string &png_name = material.map_Kd;
                obj_info.has_uv_png = true;
                obj_info.pngs.emplace(png_name, false);
                obj_info.uv_map_pngs[face_index] = png_name;
            }
            if (data.textureCoordinates.size() > 0) {
                Vec2f uv0 = Vec2f::Zero();
                Vec2f uv1 = Vec2f::Zero();
                Vec2f uv2 = Vec2f::Zero();
                if (read_uv(triangle_uv_indices[0], uv0) &&
                    read_uv(triangle_uv_indices[1], uv1) &&
                    read_uv(triangle_uv_indices[2], uv2)) {
                    std::array<Vec2f, 3> uv_array{uv0, uv1, uv2};
                    obj_info.uvs.emplace_back(uv_array);
                }
            }
            obj_info.face_colors.emplace_back(face_color);
        }
        else {
            if (obj_info.lost_material_name.empty()) {
                obj_info.lost_material_name = mtl_name;
            }
        }
    };
    size_t usemtl_idx = 0;
    auto face_material = [&data, &usemtl_idx](size_t face_vertex_idx) -> const std::string* {
        if (data.usemtls.empty())
            return nullptr;
        while (usemtl_idx + 1 < data.usemtls.size() && data.usemtls[usemtl_idx + 1].vertexIdxFirst <= int(face_vertex_idx))
            ++usemtl_idx;
        const ObjParser::ObjUseMtl &usemtl = data.usemtls[usemtl_idx];
        if (int(face_vertex_idx) < usemtl.vertexIdxFirst)
            return nullptr;
        if (usemtl.vertexIdxEnd >= 0 && int(face_vertex_idx) >= usemtl.vertexIdxEnd)
            return nullptr;
        return &usemtl.name;
    };
    struct FaceCorner {
        int coord_idx;
        int uv_idx;
    };
    auto face_normal = [&its](const std::vector<FaceCorner> &face) {
        Vec3f normal = Vec3f::Zero();
        for (size_t i = 0; i < face.size(); ++i) {
            const Vec3f &current = its.vertices[face[i].coord_idx];
            const Vec3f &next = its.vertices[face[(i + 1) % face.size()].coord_idx];
            normal.x() += (current.y() - next.y()) * (current.z() + next.z());
            normal.y() += (current.z() - next.z()) * (current.x() + next.x());
            normal.z() += (current.x() - next.x()) * (current.y() + next.y());
        }
        if (normal.squaredNorm() <= std::numeric_limits<float>::epsilon()) {
            const Vec3f &first = its.vertices[face[0].coord_idx];
            for (size_t i = 1; i + 1 < face.size(); ++i) {
                normal = (its.vertices[face[i].coord_idx] - first).cross(its.vertices[face[i + 1].coord_idx] - first);
                if (normal.squaredNorm() > std::numeric_limits<float>::epsilon())
                    break;
            }
        }
        return normal;
    };
    auto projected_point = [&its](const FaceCorner &corner, const Vec3f &normal) {
        const Vec3f &point = its.vertices[corner.coord_idx];
        const float ax = std::abs(normal.x());
        const float ay = std::abs(normal.y());
        const float az = std::abs(normal.z());
        if (ax >= ay && ax >= az)
            return std::array<double, 2>{double(point.y()), double(point.z())};
        if (ay >= ax && ay >= az)
            return std::array<double, 2>{double(point.x()), double(point.z())};
        return std::array<double, 2>{double(point.x()), double(point.y())};
    };
    auto append_triangle = [&its, &append_triangle_uv, &set_face_color, exist_mtl](const std::vector<FaceCorner> &face,
                                                                                   uint32_t a,
                                                                                   uint32_t b,
                                                                                   uint32_t c,
                                                                                   const Vec3f &normal,
                                                                                   const std::string *mtl_name) {
        if (a >= face.size() || b >= face.size() || c >= face.size())
            return false;
        const FaceCorner *corners[3] = {&face[a], &face[b], &face[c]};
        if (normal.squaredNorm() > std::numeric_limits<float>::epsilon()) {
            const Vec3f tri_normal = (its.vertices[corners[1]->coord_idx] - its.vertices[corners[0]->coord_idx]).cross(
                its.vertices[corners[2]->coord_idx] - its.vertices[corners[0]->coord_idx]);
            if (tri_normal.squaredNorm() > std::numeric_limits<float>::epsilon() && tri_normal.dot(normal) < 0.f)
                std::swap(corners[1], corners[2]);
        }
        its.indices.emplace_back(corners[0]->coord_idx, corners[1]->coord_idx, corners[2]->coord_idx);
        append_triangle_uv(corners[0]->uv_idx, corners[1]->uv_idx, corners[2]->uv_idx);
        const int face_index = int(its.indices.size()) - 1;
        if (exist_mtl && mtl_name != nullptr)
            set_face_color(face_index, *mtl_name, {corners[0]->uv_idx, corners[1]->uv_idx, corners[2]->uv_idx});
        return true;
    };
    auto append_face = [&projected_point, &append_triangle, &face_normal, &message, path](const std::vector<FaceCorner> &face,
                                                                                         const std::string *mtl_name) {
        const Vec3f normal = face_normal(face);
        if (face.size() == 3)
            return append_triangle(face, 0, 1, 2, normal, mtl_name);
        if (face.size() == 4)
            return append_triangle(face, 0, 1, 2, normal, mtl_name) &&
                   append_triangle(face, 0, 2, 3, normal, mtl_name);

        std::vector<std::vector<std::array<double, 2>>> polygon(1);
        polygon.front().reserve(face.size());
        for (const FaceCorner &corner : face)
            polygon.front().emplace_back(projected_point(corner, normal));

        const std::vector<uint32_t> triangulated = mapbox::earcut<uint32_t>(polygon);
        if (triangulated.empty() || triangulated.size() % 3 != 0) {
            BOOST_LOG_TRIVIAL(error) << "load_obj: failed to triangulate polygon in " << path;
            message = _L("The file contains a polygon that could not be triangulated.");
            return false;
        }
        for (size_t i = 0; i < triangulated.size(); i += 3)
            if (!append_triangle(face, triangulated[i], triangulated[i + 1], triangulated[i + 2], normal, mtl_name)) {
                BOOST_LOG_TRIVIAL(error) << "load_obj: failed to triangulate polygon in " << path;
                message = _L("The file contains a polygon that could not be triangulated.");
                return false;
            }
        return true;
    };

    for (size_t i = 0; i < data.vertices.size();)
        if (data.vertices[i].coordIdx == -1)
            ++ i;
        else {
            const size_t face_vertex_idx = i;
            std::vector<FaceCorner> face;
            while (i < data.vertices.size())
                if (const ObjParser::ObjVertex &vertex = data.vertices[i ++]; vertex.coordIdx == -1) {
                    break;
                } else {
                    if (vertex.coordIdx < 0 || vertex.coordIdx >= int(its.vertices.size())) {
                        BOOST_LOG_TRIVIAL(error) << "load_obj: failed to parse " << path << ". The file contains invalid vertex index.";
                        message = _L("The file contains invalid vertex index.");
                        return false;
                    }
                    face.push_back({vertex.coordIdx, vertex.textureCoordIdx});
                }
            if (!face.empty() && !append_face(face, face_material(face_vertex_idx)))
                return false;
        }

    if (obj_info.has_uv_png && !obj_info.uv_map_pngs.empty()) {
        std::set<std::string> unique_textures;
        for (const auto &face_to_png : obj_info.uv_map_pngs)
            if (!face_to_png.second.empty())
                unique_textures.insert(face_to_png.second);
        if (unique_textures.size() == 1)
            obj_info.single_texture_image = *unique_textures.begin();
    }

    *meshptr = TriangleMesh(std::move(its));
    if (meshptr->empty()) {
        BOOST_LOG_TRIVIAL(error) << "load_obj: This OBJ file couldn't be read because it's empty. " << path;
        message = _L("This OBJ file couldn't be read because it's empty.");
        return false;
    }
    if (meshptr->volume() < 0) {
        meshptr->flip_triangles();
        for (std::array<Vec2f, 3> &triangle_uv : obj_info.triangle_uvs)
            std::swap(triangle_uv[1], triangle_uv[2]);
    }
    return true;
}

bool load_obj(const char *path,
              Model *model,
              ObjInfo& obj_info,
              std::string &message,
              const char *object_name_in,
              ObjTriangulationFn objTriangulationFn)
{
    TriangleMesh mesh;

    bool ret = load_obj(path, &mesh, obj_info, message, objTriangulationFn);

    if (ret) {
        std::string  object_name;
        if (object_name_in == nullptr) {
            const char *last_slash = strrchr(path, DIR_SEPARATOR);
            object_name.assign((last_slash == nullptr) ? path : last_slash + 1);
        } else
           object_name.assign(object_name_in);
        model->add_object(object_name.c_str(), path, std::move(mesh));
    }

    return ret;
}

bool store_obj(const char *path, TriangleMesh *mesh)
{
    //FIXME returning false even if write failed.
    mesh->WriteOBJFile(path);
    return true;
}

bool store_obj(const char *path, ModelObject *model_object)
{
    TriangleMesh mesh = model_object->mesh();
    return store_obj(path, &mesh);
}

bool store_obj(const char *path, Model *model)
{
    TriangleMesh mesh = model->mesh();
    return store_obj(path, &mesh);
}

}; // namespace Slic3r
