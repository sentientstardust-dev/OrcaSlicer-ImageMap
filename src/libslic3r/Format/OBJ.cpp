#include "../libslic3r.h"
#include "../Model.hpp"
#include "../TriangleMesh.hpp"

#include "OBJ.hpp"
#include "objparser.hpp"

#include <string>
#include <set>

#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>

#ifdef _WIN32
#define DIR_SEPARATOR '\\'
#else
#define DIR_SEPARATOR '/'
#endif

//Translation
#include "I18N.hpp"
#define _L(s) Slic3r::I18N::translate(s)

namespace Slic3r {

bool load_obj(const char *path, TriangleMesh *meshptr, ObjInfo& obj_info, std::string &message)
{
    if (meshptr == nullptr)
        return false;
    // Parse the OBJ file.
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
    // Count the faces and verify, that all faces are triangular.
    size_t num_faces = 0;
    size_t num_quads = 0;
    for (size_t i = 0; i < data.vertices.size(); ++ i) {
        // Find the end of face.
        size_t j = i;
        for (; j < data.vertices.size() && data.vertices[j].coordIdx != -1; ++ j) ;
        if (size_t num_face_vertices = j - i; num_face_vertices > 0) {
            if (num_face_vertices > 4) {
                // Non-triangular and non-quad faces are not supported as of now.
                BOOST_LOG_TRIVIAL(error) << "load_obj: failed to parse " << path << ". The file contains polygons with more than 4 vertices.";
                message = _L("The file contains polygons with more than 4 vertices.");
                return false;
            } else if (num_face_vertices < 3) {
                // Non-triangular and non-quad faces are not supported as of now.
                BOOST_LOG_TRIVIAL(error) << "load_obj: failed to parse " << path << ". The file contains polygons with less than 2 vertices.";
                message = _L("The file contains polygons with less than 2 vertices.");
                return false;
            }
            if (num_face_vertices == 4)
                ++ num_quads;
            ++ num_faces;
            i = j;
        }
    }
    // Convert ObjData into indexed triangle set.
    indexed_triangle_set its;
    size_t               num_vertices = data.coordinates.size() / OBJ_VERTEX_LENGTH;
    its.vertices.reserve(num_vertices);
    its.indices.reserve(num_faces + num_quads);
    obj_info.triangle_uvs.reserve(num_faces + num_quads);
    obj_info.triangle_uvs_valid.reserve(num_faces + num_quads);
    if (exist_mtl) {
        obj_info.is_single_mtl = data.usemtls.size() == 1 && mtl_data.new_mtl_unmap.size() == 1;
        obj_info.face_colors.reserve(num_faces + num_quads);
    }
    bool has_color = data.has_vertex_color;
    for (size_t i = 0; i < num_vertices; ++ i) {
        size_t j = i * OBJ_VERTEX_LENGTH;
        its.vertices.emplace_back(data.coordinates[j], data.coordinates[j + 1], data.coordinates[j + 2]);
        if (data.has_vertex_color) {
            RGBA color{std::clamp(data.coordinates[j + 3], 0.f, 1.f), std::clamp(data.coordinates[j + 4], 0.f, 1.f), std::clamp(data.coordinates[j + 5], 0.f, 1.f),
                       std::clamp(data.coordinates[j + 6], 0.f, 1.f)};
            obj_info.vertex_colors.emplace_back(color);
        }
    }
    int indices[ONE_FACE_SIZE];
    int uvs[ONE_FACE_SIZE];
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
    for (size_t i = 0; i < data.vertices.size();)
        if (data.vertices[i].coordIdx == -1)
            ++ i;
        else {
            int cnt = 0;
            while (i < data.vertices.size())
                if (const ObjParser::ObjVertex &vertex = data.vertices[i ++]; vertex.coordIdx == -1) {
                    break;
                } else {
                    assert(cnt < ONE_FACE_SIZE);
                    if (cnt >= ONE_FACE_SIZE) {
                        BOOST_LOG_TRIVIAL(error) << "load_obj: failed to parse " << path << ". The file contains polygons with more than 4 vertices.";
                        message = _L("The file contains polygons with more than 4 vertices.");
                        return false;
                    }
                    if (vertex.coordIdx < 0 || vertex.coordIdx >= int(its.vertices.size())) {
                        BOOST_LOG_TRIVIAL(error) << "load_obj: failed to parse " << path << ". The file contains invalid vertex index.";
                        message = _L("The file contains invalid vertex index.");
                        return false;
                    }
                    indices[cnt] = vertex.coordIdx;
                    uvs[cnt]     = vertex.textureCoordIdx;
                    cnt++;
                }
            if (cnt) {
                assert(cnt == 3 || cnt == 4);
                // Insert one or two faces (triangulate a quad).
                its.indices.emplace_back(indices[0], indices[1], indices[2]);
                append_triangle_uv(uvs[0], uvs[1], uvs[2]);
                int  face_index =its.indices.size() - 1;
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
                        face_color[3] = material.Tr; // alpha
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
                auto set_face_color_by_mtl = [&data, &set_face_color](int face_index, const std::array<int, 3> &triangle_uv_indices) {
                    if (data.usemtls.size() == 1) {
                        set_face_color(face_index, data.usemtls[0].name, triangle_uv_indices);
                    } else {
                        for (size_t k = 0; k < data.usemtls.size(); k++) {
                            auto mtl = data.usemtls[k];
                            if (face_index >= mtl.face_start && face_index <= mtl.face_end) {
                                set_face_color(face_index, data.usemtls[k].name, triangle_uv_indices);
                                break;
                            }
                        }
                    }
                };
                if (exist_mtl) {
                    set_face_color_by_mtl(face_index, {uvs[0], uvs[1], uvs[2]});
                }
                if (cnt == 4) {
                    its.indices.emplace_back(indices[0], indices[2], indices[3]);
                    append_triangle_uv(uvs[0], uvs[2], uvs[3]);
                    int face_index = its.indices.size() - 1;
                    if (exist_mtl) {
                        set_face_color_by_mtl(face_index, {uvs[0], uvs[2], uvs[3]});
                    }
                }
            }
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

bool load_obj(const char *path, Model *model, ObjInfo& obj_info, std::string &message, const char *object_name_in)
{
    TriangleMesh mesh;

    bool ret = load_obj(path, &mesh, obj_info, message);

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
