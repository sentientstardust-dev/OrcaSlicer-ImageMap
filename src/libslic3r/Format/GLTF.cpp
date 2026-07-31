// original author: sentientstardust

#include "GLTF.hpp"

#include "libslic3r/Model.hpp"
#include "libslic3r/TriangleMesh.hpp"
#include "libslic3r/Utils.hpp"
#include "libslic3r/miniz_extension.hpp"

#include <tiny_gltf_v3.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <string>
#include <system_error>
#include <sys/stat.h>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>
#include <boost/nowide/cstdio.hpp>
#include <boost/nowide/fstream.hpp>

#include <Eigen/Geometry>

#ifdef _WIN32
#include <io.h>
#endif

namespace Slic3r {
namespace {

static std::string percent_decode(const std::string &value);

static std::string tg3_string(const tg3_str &str)
{
    return str.data == nullptr ? std::string() : std::string(str.data, str.len);
}

static uint32_t rgba_to_packed(const RGBA &color)
{
    const uint32_t r = uint32_t(std::lround(std::clamp(color[0], 0.f, 1.f) * 255.f)) & 0xFFu;
    const uint32_t g = uint32_t(std::lround(std::clamp(color[1], 0.f, 1.f) * 255.f)) & 0xFFu;
    const uint32_t b = uint32_t(std::lround(std::clamp(color[2], 0.f, 1.f) * 255.f)) & 0xFFu;
    const uint32_t a = uint32_t(std::lround(std::clamp(color[3], 0.f, 1.f) * 255.f)) & 0xFFu;
    return (r << 24) | (g << 16) | (b << 8) | a;
}

static int32_t tg3_read_file(uint8_t **out_data, uint64_t *out_size, const char *path, uint32_t path_len, void *)
{
    if (out_data == nullptr || out_size == nullptr || path == nullptr)
        return 0;

    *out_data = nullptr;
    *out_size = 0;

    const std::string filename = percent_decode(std::string(path, path_len));
    boost::nowide::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    if (!ifs.is_open())
        return 0;

    const std::streamoff stream_size = ifs.tellg();
    if (stream_size <= 0)
        return 0;
    ifs.seekg(0, std::ios::beg);

    uint8_t *data = static_cast<uint8_t *>(std::malloc(size_t(stream_size)));
    if (data == nullptr)
        return 0;

    if (!ifs.read(reinterpret_cast<char *>(data), stream_size)) {
        std::free(data);
        return 0;
    }

    *out_data = data;
    *out_size = uint64_t(stream_size);
    return 1;
}

static void tg3_free_file(uint8_t *data, uint64_t, void *)
{
    std::free(data);
}

static int attribute_index(const tg3_primitive &primitive, const char *name)
{
    const size_t name_len = std::strlen(name);
    for (uint32_t i = 0; i < primitive.attributes_count; ++i) {
        const tg3_str &key = primitive.attributes[i].key;
        if (key.len == name_len && key.data != nullptr && std::memcmp(key.data, name, name_len) == 0)
            return primitive.attributes[i].value;
    }
    return -1;
}

static size_t component_size(int32_t component_type)
{
    switch (component_type) {
    case TG3_COMPONENT_TYPE_BYTE:
    case TG3_COMPONENT_TYPE_UNSIGNED_BYTE:
        return 1;
    case TG3_COMPONENT_TYPE_SHORT:
    case TG3_COMPONENT_TYPE_UNSIGNED_SHORT:
        return 2;
    case TG3_COMPONENT_TYPE_INT:
    case TG3_COMPONENT_TYPE_UNSIGNED_INT:
    case TG3_COMPONENT_TYPE_FLOAT:
        return 4;
    case TG3_COMPONENT_TYPE_DOUBLE:
        return 8;
    default:
        return 0;
    }
}

static size_t accessor_component_count(int32_t type)
{
    switch (type) {
    case TG3_TYPE_SCALAR:
        return 1;
    case TG3_TYPE_VEC2:
        return 2;
    case TG3_TYPE_VEC3:
        return 3;
    case TG3_TYPE_VEC4:
        return 4;
    default:
        return 0;
    }
}

struct AccessorView
{
    const tg3_accessor    *accessor{nullptr};
    const tg3_buffer_view *buffer_view{nullptr};
    const tg3_buffer      *buffer{nullptr};
    size_t                 component_size{0};
    size_t                 component_count{0};
    size_t                 element_size{0};
    size_t                 stride{0};
    uint64_t               base_offset{0};
};

static bool make_accessor_view(const tg3_model &model, int accessor_idx, AccessorView &view)
{
    view = {};
    if (accessor_idx < 0 || uint32_t(accessor_idx) >= model.accessors_count)
        return false;

    const tg3_accessor &accessor = model.accessors[accessor_idx];
    if (accessor.buffer_view < 0 || uint32_t(accessor.buffer_view) >= model.buffer_views_count || accessor.sparse.is_sparse)
        return false;

    const tg3_buffer_view &buffer_view = model.buffer_views[accessor.buffer_view];
    if (buffer_view.buffer < 0 || uint32_t(buffer_view.buffer) >= model.buffers_count)
        return false;

    const tg3_buffer &buffer = model.buffers[buffer_view.buffer];
    const size_t comp_size = component_size(accessor.component_type);
    const size_t comp_count = accessor_component_count(accessor.type);
    if (comp_size == 0 || comp_count == 0)
        return false;

    const size_t element_size = comp_size * comp_count;
    const size_t stride = buffer_view.byte_stride == 0 ? element_size : size_t(buffer_view.byte_stride);
    if (stride < element_size)
        return false;
    if (buffer_view.byte_offset > std::numeric_limits<uint64_t>::max() - accessor.byte_offset)
        return false;

    const uint64_t base_offset = buffer_view.byte_offset + accessor.byte_offset;
    if (base_offset > buffer.data.count)
        return false;
    if (accessor.count > 0) {
        const uint64_t last_offset = base_offset + uint64_t(stride) * (accessor.count - 1);
        if (last_offset < base_offset || last_offset > buffer.data.count || uint64_t(element_size) > buffer.data.count - last_offset)
            return false;
    }

    view.accessor        = &accessor;
    view.buffer_view     = &buffer_view;
    view.buffer          = &buffer;
    view.component_size  = comp_size;
    view.component_count = comp_count;
    view.element_size    = element_size;
    view.stride          = stride;
    view.base_offset     = base_offset;
    return true;
}

template<class T> static T read_unaligned(const uint8_t *data)
{
    T value;
    std::memcpy(&value, data, sizeof(T));
    return value;
}

static double normalized_signed(double value, double max_value)
{
    return std::max(value / max_value, -1.0);
}

static bool read_component(const AccessorView &view, uint64_t element_idx, size_t component_idx, bool force_normalized, double &out)
{
    if (view.accessor == nullptr || element_idx >= view.accessor->count || component_idx >= view.component_count)
        return false;

    const uint8_t *ptr = view.buffer->data.data + view.base_offset + element_idx * view.stride + component_idx * view.component_size;
    const bool normalized = force_normalized || view.accessor->normalized != 0;

    switch (view.accessor->component_type) {
    case TG3_COMPONENT_TYPE_BYTE: {
        const int8_t v = read_unaligned<int8_t>(ptr);
        out = normalized ? normalized_signed(double(v), 127.0) : double(v);
        return true;
    }
    case TG3_COMPONENT_TYPE_UNSIGNED_BYTE: {
        const uint8_t v = read_unaligned<uint8_t>(ptr);
        out = normalized ? double(v) / 255.0 : double(v);
        return true;
    }
    case TG3_COMPONENT_TYPE_SHORT: {
        const int16_t v = read_unaligned<int16_t>(ptr);
        out = normalized ? normalized_signed(double(v), 32767.0) : double(v);
        return true;
    }
    case TG3_COMPONENT_TYPE_UNSIGNED_SHORT: {
        const uint16_t v = read_unaligned<uint16_t>(ptr);
        out = normalized ? double(v) / 65535.0 : double(v);
        return true;
    }
    case TG3_COMPONENT_TYPE_INT: {
        const int32_t v = read_unaligned<int32_t>(ptr);
        out = normalized ? normalized_signed(double(v), 2147483647.0) : double(v);
        return true;
    }
    case TG3_COMPONENT_TYPE_UNSIGNED_INT: {
        const uint32_t v = read_unaligned<uint32_t>(ptr);
        out = normalized ? double(v) / 4294967295.0 : double(v);
        return true;
    }
    case TG3_COMPONENT_TYPE_FLOAT:
        out = double(read_unaligned<float>(ptr));
        return true;
    case TG3_COMPONENT_TYPE_DOUBLE:
        out = read_unaligned<double>(ptr);
        return true;
    default:
        return false;
    }
}

static bool read_index(const AccessorView &view, uint64_t element_idx, uint32_t &out)
{
    if (view.accessor == nullptr || view.accessor->type != TG3_TYPE_SCALAR || element_idx >= view.accessor->count)
        return false;

    const uint8_t *ptr = view.buffer->data.data + view.base_offset + element_idx * view.stride;
    switch (view.accessor->component_type) {
    case TG3_COMPONENT_TYPE_UNSIGNED_BYTE:
        out = read_unaligned<uint8_t>(ptr);
        return true;
    case TG3_COMPONENT_TYPE_UNSIGNED_SHORT:
        out = read_unaligned<uint16_t>(ptr);
        return true;
    case TG3_COMPONENT_TYPE_UNSIGNED_INT:
        out = read_unaligned<uint32_t>(ptr);
        return true;
    default:
        return false;
    }
}

static bool read_vec3(const AccessorView &view, uint64_t element_idx, Vec3f &out)
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (view.component_count < 3 ||
        !read_component(view, element_idx, 0, false, x) ||
        !read_component(view, element_idx, 1, false, y) ||
        !read_component(view, element_idx, 2, false, z))
        return false;
    out = Vec3f(float(x), float(y), float(z));
    return true;
}

static bool read_vec2(const AccessorView &view, uint64_t element_idx, Vec2f &out)
{
    double x = 0.0;
    double y = 0.0;
    if (view.component_count < 2 ||
        !read_component(view, element_idx, 0, false, x) ||
        !read_component(view, element_idx, 1, false, y))
        return false;
    out = Vec2f(float(x), float(y));
    return true;
}

static RGBA read_color_or_white(const AccessorView *view, uint64_t element_idx)
{
    if (view == nullptr)
        return {1.f, 1.f, 1.f, 1.f};

    double r = 1.0;
    double g = 1.0;
    double b = 1.0;
    double a = 1.0;
    const bool force_normalized = view->accessor->component_type != TG3_COMPONENT_TYPE_FLOAT &&
                                  view->accessor->component_type != TG3_COMPONENT_TYPE_DOUBLE;
    if (view->component_count < 3 ||
        !read_component(*view, element_idx, 0, force_normalized, r) ||
        !read_component(*view, element_idx, 1, force_normalized, g) ||
        !read_component(*view, element_idx, 2, force_normalized, b))
        return {1.f, 1.f, 1.f, 1.f};

    if (view->component_count >= 4)
        read_component(*view, element_idx, 3, force_normalized, a);

    return {float(std::clamp(r, 0.0, 1.0)),
            float(std::clamp(g, 0.0, 1.0)),
            float(std::clamp(b, 0.0, 1.0)),
            float(std::clamp(a, 0.0, 1.0))};
}

static bool buffer_view_bytes(const tg3_model &model, int buffer_view_idx, const uint8_t *&data, size_t &size)
{
    data = nullptr;
    size = 0;
    if (buffer_view_idx < 0 || uint32_t(buffer_view_idx) >= model.buffer_views_count)
        return false;
    const tg3_buffer_view &buffer_view = model.buffer_views[buffer_view_idx];
    if (buffer_view.buffer < 0 || uint32_t(buffer_view.buffer) >= model.buffers_count)
        return false;
    const tg3_buffer &buffer = model.buffers[buffer_view.buffer];
    if (buffer_view.byte_offset > buffer.data.count || buffer_view.byte_length > buffer.data.count - buffer_view.byte_offset)
        return false;
    data = buffer.data.data + buffer_view.byte_offset;
    size = size_t(buffer_view.byte_length);
    return true;
}

static int hex_value(char c)
{
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return 10 + c - 'a';
    if (c >= 'A' && c <= 'F')
        return 10 + c - 'A';
    return -1;
}

static std::string percent_decode(const std::string &value)
{
    std::string out;
    out.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '%' && i + 2 < value.size()) {
            const int hi = hex_value(value[i + 1]);
            const int lo = hex_value(value[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(char((hi << 4) | lo));
                i += 2;
                continue;
            }
        }
        out.push_back(value[i]);
    }
    return out;
}

static int base64_value(char c)
{
    if (c >= 'A' && c <= 'Z')
        return c - 'A';
    if (c >= 'a' && c <= 'z')
        return 26 + c - 'a';
    if (c >= '0' && c <= '9')
        return 52 + c - '0';
    if (c == '+')
        return 62;
    if (c == '/')
        return 63;
    return -1;
}

static bool decode_base64(const std::string &input, std::vector<uint8_t> &out)
{
    out.clear();
    uint32_t val = 0;
    int valb = -8;
    for (const char c : input) {
        if (std::isspace(static_cast<unsigned char>(c)))
            continue;
        if (c == '=')
            break;
        const int decoded = base64_value(c);
        if (decoded < 0)
            return false;
        val = (val << 6) + decoded;
        valb += 6;
        if (valb >= 0) {
            out.push_back(uint8_t((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return !out.empty();
}

static bool decode_data_uri(const std::string &uri, std::vector<uint8_t> &bytes, std::string &mime_type)
{
    bytes.clear();
    mime_type.clear();

    if (uri.rfind("data:", 0) != 0)
        return false;

    const size_t comma = uri.find(',');
    if (comma == std::string::npos)
        return false;

    const std::string meta = uri.substr(5, comma - 5);
    const std::string payload = uri.substr(comma + 1);
    const bool is_base64 = meta.find(";base64") != std::string::npos;
    const size_t semicolon = meta.find(';');
    mime_type = semicolon == std::string::npos ? meta : meta.substr(0, semicolon);

    if (!is_base64)
        return false;

    return decode_base64(payload, bytes);
}

struct GltfTextureState
{
    std::vector<int> image_to_imported;
    std::vector<int> texture_to_imported;
};

static bool decode_gltf_image(const tg3_model &model,
                              const boost::filesystem::path &base_dir,
                              const tg3_image &image,
                              ImportedTextureImage &out)
{
    out = {};
    out.name = tg3_string(image.name);

    if (image.buffer_view >= 0) {
        const uint8_t *data = nullptr;
        size_t data_size = 0;
        if (!buffer_view_bytes(model, image.buffer_view, data, data_size))
            return false;
        return decode_image_texture_rgba_from_memory(data,
                                                     data_size,
                                                     tg3_string(image.mime_type),
                                                     out.rgba,
                                                     out.width,
                                                     out.height);
    }

    const std::string uri = tg3_string(image.uri);
    if (uri.empty())
        return false;

    if (uri.rfind("data:", 0) == 0) {
        std::vector<uint8_t> bytes;
        std::string mime_type;
        if (!decode_data_uri(uri, bytes, mime_type))
            return false;
        return decode_image_texture_rgba_from_memory(bytes.data(),
                                                     bytes.size(),
                                                     mime_type,
                                                     out.rgba,
                                                     out.width,
                                                     out.height);
    }

    if (uri.find(':') != std::string::npos && !boost::algorithm::istarts_with(uri, "file:"))
        return false;

    std::string decoded_uri = percent_decode(uri);
    if (boost::algorithm::istarts_with(decoded_uri, "file://"))
        decoded_uri = decoded_uri.substr(7);
    else if (boost::algorithm::istarts_with(decoded_uri, "file:"))
        decoded_uri = decoded_uri.substr(5);

    boost::filesystem::path image_path(decoded_uri);
    if (!image_path.is_absolute())
        image_path = base_dir / image_path;

    out.name = image_path.string();
    return decode_image_texture_rgba_from_file(image_path.lexically_normal().string(), out.rgba, out.width, out.height);
}

static int imported_texture_index_for_gltf_texture(const tg3_model &model,
                                                   const boost::filesystem::path &base_dir,
                                                   int texture_idx,
                                                   GltfTextureState &texture_state,
                                                   GltfImportInfo &info)
{
    if (texture_idx < 0 || uint32_t(texture_idx) >= model.textures_count)
        return -1;

    if (texture_state.texture_to_imported.empty())
        texture_state.texture_to_imported.assign(model.textures_count, -2);
    if (texture_state.image_to_imported.empty())
        texture_state.image_to_imported.assign(model.images_count, -2);

    int &cached_texture = texture_state.texture_to_imported[texture_idx];
    if (cached_texture != -2)
        return cached_texture;

    const tg3_texture &texture = model.textures[texture_idx];
    if (texture.source < 0 || uint32_t(texture.source) >= model.images_count) {
        cached_texture = -1;
        return -1;
    }

    int &cached_image = texture_state.image_to_imported[texture.source];
    if (cached_image != -2) {
        cached_texture = cached_image;
        return cached_texture;
    }

    ImportedTextureImage imported;
    if (!decode_gltf_image(model, base_dir, model.images[texture.source], imported)) {
        BOOST_LOG_TRIVIAL(error) << "glTF material texture image failed to decode image_index=" << texture.source;
        cached_image = -1;
        cached_texture = -1;
        return -1;
    }

    cached_image = int(info.textures.size());
    info.textures.emplace_back(std::move(imported));
    cached_texture = cached_image;
    return cached_texture;
}

static RGBA material_base_color(const tg3_material *material)
{
    if (material == nullptr)
        return {1.f, 1.f, 1.f, 1.f};

    return {
        float(std::clamp(material->pbr_metallic_roughness.base_color_factor[0], 0.0, 1.0)),
        float(std::clamp(material->pbr_metallic_roughness.base_color_factor[1], 0.0, 1.0)),
        float(std::clamp(material->pbr_metallic_roughness.base_color_factor[2], 0.0, 1.0)),
        float(std::clamp(material->pbr_metallic_roughness.base_color_factor[3], 0.0, 1.0))
    };
}

static const tg3_value *find_object_value(const tg3_value &value, const char *key)
{
    if (value.type != TG3_VALUE_OBJECT || key == nullptr)
        return nullptr;

    const size_t key_len = std::strlen(key);
    for (uint32_t i = 0; i < value.object_count; ++i) {
        const tg3_kv_pair &pair = value.object_data[i];
        if (pair.key.data != nullptr && pair.key.len == key_len && std::memcmp(pair.key.data, key, key_len) == 0)
            return &pair.value;
    }
    return nullptr;
}

static const tg3_value *find_extension_value(const tg3_extras_ext &ext, const char *name)
{
    if (name == nullptr)
        return nullptr;

    const size_t name_len = std::strlen(name);
    for (uint32_t i = 0; i < ext.extensions_count; ++i) {
        const tg3_extension &extension = ext.extensions[i];
        if (extension.name.data != nullptr && extension.name.len == name_len && std::memcmp(extension.name.data, name, name_len) == 0)
            return &extension.value;
    }
    return nullptr;
}

static bool value_to_int(const tg3_value &value, int &out)
{
    if (value.type != TG3_VALUE_INT)
        return false;
    if (value.int_val < std::numeric_limits<int>::min() || value.int_val > std::numeric_limits<int>::max())
        return false;
    out = int(value.int_val);
    return true;
}

static bool texture_info_from_extension_value(const tg3_value *value, int &texture_idx, int &tex_coord)
{
    if (value == nullptr || value->type != TG3_VALUE_OBJECT)
        return false;

    const tg3_value *index_value = find_object_value(*value, "index");
    if (index_value == nullptr || !value_to_int(*index_value, texture_idx))
        return false;

    tex_coord = 0;
    if (const tg3_value *tex_coord_value = find_object_value(*value, "texCoord"))
        value_to_int(*tex_coord_value, tex_coord);

    return true;
}

static bool diffuse_factor_from_extension_value(const tg3_value *value, RGBA &out)
{
    if (value == nullptr || value->type != TG3_VALUE_ARRAY || value->array_count < 3)
        return false;

    std::array<float, 4> factor{1.f, 1.f, 1.f, 1.f};
    for (uint32_t i = 0; i < std::min<uint32_t>(value->array_count, 4); ++i) {
        const tg3_value &component = value->array_data[i];
        if (component.type == TG3_VALUE_REAL)
            factor[i] = float(std::clamp(component.real_val, 0.0, 1.0));
        else if (component.type == TG3_VALUE_INT)
            factor[i] = float(std::clamp(double(component.int_val), 0.0, 1.0));
        else
            return false;
    }

    out = factor;
    return true;
}

struct MaterialTextureSource
{
    RGBA color{1.f, 1.f, 1.f, 1.f};
    int  texture_idx{-1};
    int  tex_coord{0};
};

static MaterialTextureSource material_texture_source(const tg3_material *material)
{
    MaterialTextureSource source;
    if (material == nullptr)
        return source;

    source.color = material_base_color(material);
    source.texture_idx = material->pbr_metallic_roughness.base_color_texture.index;
    source.tex_coord = material->pbr_metallic_roughness.base_color_texture.tex_coord;

    const tg3_value *spec_gloss = find_extension_value(material->ext, "KHR_materials_pbrSpecularGlossiness");
    if (spec_gloss != nullptr) {
        RGBA diffuse_factor;
        if (diffuse_factor_from_extension_value(find_object_value(*spec_gloss, "diffuseFactor"), diffuse_factor))
            source.color = diffuse_factor;

        int diffuse_texture_idx = -1;
        int diffuse_tex_coord = 0;
        if (source.texture_idx < 0 &&
            texture_info_from_extension_value(find_object_value(*spec_gloss, "diffuseTexture"), diffuse_texture_idx, diffuse_tex_coord)) {
            source.texture_idx = diffuse_texture_idx;
            source.tex_coord = diffuse_tex_coord;
        }
    }

    return source;
}

static Eigen::Matrix4d node_transform(const tg3_node &node)
{
    if (node.has_matrix) {
        Eigen::Matrix4d matrix;
        for (int col = 0; col < 4; ++col)
            for (int row = 0; row < 4; ++row)
                matrix(row, col) = node.matrix[col * 4 + row];
        return matrix;
    }

    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    transform.translate(Eigen::Vector3d(node.translation[0], node.translation[1], node.translation[2]));
    Eigen::Quaterniond rotation(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
    if (rotation.norm() > 0.0)
        transform.rotate(rotation.normalized());
    transform.scale(Eigen::Vector3d(node.scale[0], node.scale[1], node.scale[2]));
    return transform.matrix();
}

static bool build_primitive_indices(const tg3_model &model, const tg3_primitive &primitive, const AccessorView &positions, std::vector<uint32_t> &indices)
{
    indices.clear();
    if (primitive.indices >= 0) {
        AccessorView index_view;
        if (!make_accessor_view(model, primitive.indices, index_view))
            return false;
        indices.reserve(size_t(index_view.accessor->count));
        for (uint64_t i = 0; i < index_view.accessor->count; ++i) {
            uint32_t index = 0;
            if (!read_index(index_view, i, index) || index >= positions.accessor->count)
                return false;
            indices.emplace_back(index);
        }
    } else {
        if (positions.accessor->count > std::numeric_limits<uint32_t>::max())
            return false;
        indices.reserve(size_t(positions.accessor->count));
        for (uint64_t i = 0; i < positions.accessor->count; ++i)
            indices.emplace_back(uint32_t(i));
    }
    return true;
}

static void triangle_source_indices(int mode, const std::vector<uint32_t> &indices, size_t triangle_idx, std::array<uint32_t, 3> &out)
{
    if (mode == TG3_MODE_TRIANGLES || mode == -1) {
        out = {indices[triangle_idx * 3], indices[triangle_idx * 3 + 1], indices[triangle_idx * 3 + 2]};
    } else if (mode == TG3_MODE_TRIANGLE_STRIP) {
        if (triangle_idx % 2 == 0)
            out = {indices[triangle_idx], indices[triangle_idx + 1], indices[triangle_idx + 2]};
        else
            out = {indices[triangle_idx + 1], indices[triangle_idx], indices[triangle_idx + 2]};
    } else {
        out = {indices[0], indices[triangle_idx + 1], indices[triangle_idx + 2]};
    }
}

static size_t triangle_count_for_mode(int mode, size_t index_count)
{
    if (mode == TG3_MODE_TRIANGLES || mode == -1)
        return index_count / 3;
    if (mode == TG3_MODE_TRIANGLE_STRIP || mode == TG3_MODE_TRIANGLE_FAN)
        return index_count >= 3 ? index_count - 2 : 0;
    return 0;
}

static void append_mesh_primitive(const tg3_model &model,
                                  const boost::filesystem::path &base_dir,
                                  const tg3_primitive &primitive,
                                  const Eigen::Matrix4d &transform,
                                  indexed_triangle_set &its,
                                  GltfTextureState &texture_state,
                                  GltfImportInfo &info)
{
    const int mode = primitive.mode == -1 ? TG3_MODE_TRIANGLES : primitive.mode;
    if (mode != TG3_MODE_TRIANGLES && mode != TG3_MODE_TRIANGLE_STRIP && mode != TG3_MODE_TRIANGLE_FAN)
        return;

    AccessorView positions;
    if (!make_accessor_view(model, attribute_index(primitive, "POSITION"), positions) ||
        positions.accessor->type != TG3_TYPE_VEC3)
        return;

    std::vector<uint32_t> source_indices;
    if (!build_primitive_indices(model, primitive, positions, source_indices))
        return;

    const tg3_material *material = nullptr;
    bool has_material = false;
    int imported_texture_idx = -1;
    int texture_coord = 0;
    if (primitive.material >= 0 && uint32_t(primitive.material) < model.materials_count) {
        material = &model.materials[primitive.material];
        has_material = true;
    }

    const MaterialTextureSource texture_source = material_texture_source(material);
    if (material != nullptr) {
        texture_coord = std::max(texture_source.tex_coord, 0);
        imported_texture_idx = imported_texture_index_for_gltf_texture(model, base_dir, texture_source.texture_idx, texture_state, info);
    }

    const RGBA face_color = texture_source.color;
    const int color_accessor_idx = attribute_index(primitive, "COLOR_0");
    AccessorView color_view_storage;
    AccessorView *color_view = nullptr;
    if (make_accessor_view(model, color_accessor_idx, color_view_storage) &&
        (color_view_storage.accessor->type == TG3_TYPE_VEC3 || color_view_storage.accessor->type == TG3_TYPE_VEC4)) {
        color_view = &color_view_storage;
        info.has_vertex_colors = true;
    }

    std::string uv_attr = "TEXCOORD_" + std::to_string(texture_coord);
    const int uv_accessor_idx = attribute_index(primitive, uv_attr.c_str());
    AccessorView uv_view_storage;
    AccessorView *uv_view = nullptr;
    if (imported_texture_idx >= 0 &&
        make_accessor_view(model, uv_accessor_idx, uv_view_storage) &&
        uv_view_storage.accessor->type == TG3_TYPE_VEC2)
        uv_view = &uv_view_storage;

    const size_t tri_count = triangle_count_for_mode(mode, source_indices.size());
    for (size_t tri_idx = 0; tri_idx < tri_count; ++tri_idx) {
        std::array<uint32_t, 3> src{};
        triangle_source_indices(mode, source_indices, tri_idx, src);
        if (src[0] == src[1] || src[0] == src[2] || src[1] == src[2])
            continue;

        std::array<Vec3f, 3> tri_positions;
        std::array<RGBA, 3> tri_colors;
        std::array<Vec2f, 3> tri_uv{Vec2f(0.f, 0.f), Vec2f(0.f, 0.f), Vec2f(0.f, 0.f)};
        bool valid_triangle = true;
        bool valid_uv = uv_view != nullptr;
        for (int corner = 0; corner < 3; ++corner) {
            Vec3f position;
            if (!read_vec3(positions, src[corner], position)) {
                valid_triangle = false;
                break;
            }

            const Eigen::Vector4d transformed = transform * Eigen::Vector4d(position.x(), position.y(), position.z(), 1.0);
            tri_positions[corner] = Vec3f(float(transformed.x()), float(transformed.y()), float(transformed.z()));
            tri_colors[corner] = read_color_or_white(color_view, src[corner]);

            if (uv_view != nullptr) {
                if (read_vec2(*uv_view, src[corner], tri_uv[corner]))
                    tri_uv[corner].y() = 1.f - tri_uv[corner].y();
                else
                    valid_uv = false;
            }
        }

        if (!valid_triangle)
            continue;

        Vec3i32 face;
        for (int corner = 0; corner < 3; ++corner) {
            face[corner] = int(its.vertices.size());
            its.vertices.emplace_back(tri_positions[corner]);
            info.vertex_colors.emplace_back(tri_colors[corner]);
            info.vertex_colors_rgba.emplace_back(rgba_to_packed(tri_colors[corner]));
        }

        its.indices.emplace_back(face);
        info.triangle_uvs.emplace_back(tri_uv);
        info.triangle_uvs_valid.emplace_back(valid_uv ? uint8_t(1) : uint8_t(0));
        info.triangle_texture_indices.emplace_back(valid_uv ? imported_texture_idx : -1);
        info.material_colors.emplace_back(face_color);
        info.has_material_colors = info.has_material_colors || has_material;
    }
}

static void append_node_meshes(const tg3_model &model,
                               const boost::filesystem::path &base_dir,
                               int node_idx,
                               const Eigen::Matrix4d &parent_transform,
                               std::vector<uint8_t> &visited,
                               indexed_triangle_set &its,
                               GltfTextureState &texture_state,
                               GltfImportInfo &info)
{
    if (node_idx < 0 || uint32_t(node_idx) >= model.nodes_count)
        return;
    if (visited[node_idx] != 0)
        return;
    visited[node_idx] = 1;

    const tg3_node &node = model.nodes[node_idx];
    const Eigen::Matrix4d transform = parent_transform * node_transform(node);
    if (node.mesh >= 0 && uint32_t(node.mesh) < model.meshes_count) {
        const tg3_mesh &mesh = model.meshes[node.mesh];
        for (uint32_t primitive_idx = 0; primitive_idx < mesh.primitives_count; ++primitive_idx)
            append_mesh_primitive(model, base_dir, mesh.primitives[primitive_idx], transform, its, texture_state, info);
    }

    for (uint32_t child_idx = 0; child_idx < node.children_count; ++child_idx)
        append_node_meshes(model, base_dir, node.children[child_idx], transform, visited, its, texture_state, info);

    visited[node_idx] = 0;
}

static std::vector<int> scene_root_nodes(const tg3_model &model)
{
    if (model.scenes_count > 0) {
        const int scene_idx = model.default_scene >= 0 && uint32_t(model.default_scene) < model.scenes_count ? model.default_scene : 0;
        const tg3_scene &scene = model.scenes[scene_idx];
        return std::vector<int>(scene.nodes, scene.nodes + scene.nodes_count);
    }

    std::vector<uint8_t> is_child(model.nodes_count, 0);
    for (uint32_t node_idx = 0; node_idx < model.nodes_count; ++node_idx) {
        const tg3_node &node = model.nodes[node_idx];
        for (uint32_t child_idx = 0; child_idx < node.children_count; ++child_idx)
            if (node.children[child_idx] >= 0 && uint32_t(node.children[child_idx]) < model.nodes_count)
                is_child[node.children[child_idx]] = 1;
    }

    std::vector<int> roots;
    for (uint32_t node_idx = 0; node_idx < model.nodes_count; ++node_idx)
        if (is_child[node_idx] == 0)
            roots.emplace_back(int(node_idx));
    return roots;
}

static std::string first_error_message(const tg3_error_stack &errors)
{
    const uint32_t count = tg3_errors_count(&errors);
    if (count == 0)
        return {};

    const tg3_error_entry *entry = tg3_errors_get(&errors, 0);
    if (entry == nullptr || entry->message == nullptr)
        return {};

    return entry->message;
}

static void finish_import_info(GltfImportInfo &info)
{
    if (!info.has_vertex_colors) {
        info.vertex_colors.clear();
        info.vertex_colors_rgba.clear();
    }

    if (!info.has_material_colors) {
        info.material_colors.clear();
        info.is_single_material_color = false;
        return;
    }

    info.is_single_material_color = !info.material_colors.empty();
    for (size_t i = 1; i < info.material_colors.size(); ++i) {
        if (!color_is_equal(info.material_colors.front(), info.material_colors[i])) {
            info.is_single_material_color = false;
            break;
        }
    }
}

static tg3_str gltf_string(const char *value)
{
    return value == nullptr ? tg3_str{} : tg3_str{value, uint32_t(std::strlen(value))};
}

static constexpr uint64_t GLTF_EXPORT_MAX_BINARY_SIZE = 2ull * 1024ull * 1024ull * 1024ull;

struct GltfExportPrimitive
{
    tg3_primitive                value{};
    std::vector<tg3_str_int_pair> attributes;
};

struct GltfExportMesh
{
    tg3_str                         name{};
    std::vector<GltfExportPrimitive> primitives;
    std::vector<tg3_primitive>       values;
};

static tg3_material gltf_export_material(const tg3_str &name, int texture_index, bool blend)
{
    tg3_material material{};
    material.name = name;
    material.alpha_mode = gltf_string(blend ? "BLEND" : "OPAQUE");
    material.alpha_cutoff = 0.5;
    material.pbr_metallic_roughness.base_color_factor[0] = 1.0;
    material.pbr_metallic_roughness.base_color_factor[1] = 1.0;
    material.pbr_metallic_roughness.base_color_factor[2] = 1.0;
    material.pbr_metallic_roughness.base_color_factor[3] = 1.0;
    material.pbr_metallic_roughness.base_color_texture.index = texture_index;
    material.pbr_metallic_roughness.base_color_texture.tex_coord = 0;
    material.pbr_metallic_roughness.metallic_factor = 0.0;
    material.pbr_metallic_roughness.roughness_factor = 1.0;
    material.pbr_metallic_roughness.metallic_roughness_texture.index = TG3_INDEX_NONE;
    material.normal_texture.index = TG3_INDEX_NONE;
    material.normal_texture.scale = 1.0;
    material.occlusion_texture.index = TG3_INDEX_NONE;
    material.occlusion_texture.strength = 1.0;
    material.emissive_texture.index = TG3_INDEX_NONE;
    return material;
}

static tg3_node gltf_export_node(const tg3_str &name, int mesh)
{
    tg3_node node{};
    node.name = name;
    node.camera = TG3_INDEX_NONE;
    node.skin = TG3_INDEX_NONE;
    node.mesh = mesh;
    node.light = TG3_INDEX_NONE;
    node.emitter = TG3_INDEX_NONE;
    node.rotation[3] = 1.0;
    node.scale[0] = 1.0;
    node.scale[1] = 1.0;
    node.scale[2] = 1.0;
    return node;
}

static bool gltf_export_triangle_valid(const indexed_triangle_set &its, size_t triangle_idx)
{
    if (triangle_idx >= its.indices.size())
        return false;
    const stl_triangle_vertex_indices &triangle = its.indices[triangle_idx];
    return triangle[0] >= 0 &&
           triangle[1] >= 0 &&
           triangle[2] >= 0 &&
           size_t(triangle[0]) < its.vertices.size() &&
           size_t(triangle[1]) < its.vertices.size() &&
           size_t(triangle[2]) < its.vertices.size();
}

static bool gltf_export_triangle_has_uvs(const ModelVolume &volume, size_t triangle_idx)
{
    if (triangle_idx >= volume.imported_texture_uv_valid.size() ||
        volume.imported_texture_uv_valid[triangle_idx] == 0)
        return false;
    const size_t offset = triangle_idx * 6;
    if (offset + 5 >= volume.imported_texture_uvs_per_face.size())
        return false;
    for (size_t idx = 0; idx < 6; ++idx)
        if (!std::isfinite(volume.imported_texture_uvs_per_face[offset + idx]))
            return false;
    return true;
}

static bool gltf_export_texture_valid(const ModelVolume &volume)
{
    const indexed_triangle_set &its = volume.mesh().its;
    const uint64_t pixel_count = uint64_t(volume.imported_texture_width) *
                                 uint64_t(volume.imported_texture_height);
    if (volume.imported_texture_width == 0 ||
        volume.imported_texture_height == 0 ||
        volume.imported_texture_width > uint32_t(std::numeric_limits<int>::max()) ||
        volume.imported_texture_height > uint32_t(std::numeric_limits<int>::max()) ||
        pixel_count > std::numeric_limits<size_t>::max() / 4 ||
        volume.imported_texture_rgba.size() < size_t(pixel_count) * 4 ||
        volume.imported_texture_uv_valid.size() != its.indices.size() ||
        volume.imported_texture_uvs_per_face.size() < its.indices.size() * 6)
        return false;
    for (size_t triangle_idx = 0; triangle_idx < its.indices.size(); ++triangle_idx)
        if (gltf_export_triangle_valid(its, triangle_idx) &&
            gltf_export_triangle_has_uvs(volume, triangle_idx))
            return true;
    return false;
}

class GltfExportBuilder
{
public:
    GltfExportBuilder(const ModelObject &object,
                      GltfExportColorMode color_mode,
                      const std::function<void()> &check_cancel)
        : m_object(object)
        , m_color_mode(color_mode)
        , m_check_cancel(check_cancel)
    {
    }

    bool build(tg3_model &model, std::string &message)
    {
        const size_t model_part_count = std::count_if(m_object.volumes.begin(), m_object.volumes.end(), [](const ModelVolume *volume) {
            return volume != nullptr && volume->is_model_part();
        });
        m_meshes.reserve(model_part_count);
        m_mesh_values.reserve(model_part_count);
        m_nodes.reserve(model_part_count + 1);
        m_root_children.reserve(model_part_count);
        m_buffer_views.reserve(model_part_count * 7);
        m_accessors.reserve(model_part_count * 6);
        m_images.reserve(model_part_count);
        m_textures.reserve(model_part_count);
        m_materials.reserve(model_part_count + 1);

        m_materials.emplace_back(gltf_export_material(add_string("Untextured"), TG3_INDEX_NONE, false));
        m_nodes.emplace_back(gltf_export_node(add_string(m_object.name.empty() ? "Object" : m_object.name), TG3_INDEX_NONE));

        bool vertex_alpha = false;
        for (const ModelVolume *volume : m_object.volumes) {
            if (m_check_cancel)
                m_check_cancel();
            if (volume == nullptr || !volume->is_model_part())
                continue;
            if (!append_volume(*volume, vertex_alpha, message))
                return false;
        }

        if (m_meshes.empty()) {
            message = "The object contains no exportable model-part triangles.";
            return false;
        }

        if (m_color_mode == GltfExportColorMode::VertexColors && vertex_alpha)
            m_materials.front().alpha_mode = gltf_string("BLEND");

        for (GltfExportMesh &mesh : m_meshes) {
            mesh.values.reserve(mesh.primitives.size());
            for (GltfExportPrimitive &primitive : mesh.primitives) {
                primitive.value.attributes = primitive.attributes.data();
                primitive.value.attributes_count = uint32_t(primitive.attributes.size());
                mesh.values.emplace_back(primitive.value);
            }
            tg3_mesh value{};
            value.name = mesh.name;
            value.primitives = mesh.values.data();
            value.primitives_count = uint32_t(mesh.values.size());
            m_mesh_values.emplace_back(value);
        }

        m_nodes.front().children = m_root_children.data();
        m_nodes.front().children_count = uint32_t(m_root_children.size());

        tg3_scene scene{};
        scene.name = add_string("Scene");
        m_scene_nodes.emplace_back(0);
        scene.nodes = m_scene_nodes.data();
        scene.nodes_count = uint32_t(m_scene_nodes.size());
        m_scenes.emplace_back(scene);

        tg3_buffer buffer{};
        buffer.name = add_string("Buffer");
        buffer.data = tg3_span_u8{m_binary.data(), uint64_t(m_binary.size())};
        m_buffers.emplace_back(buffer);

        model = {};
        model.asset.version = gltf_string("2.0");
        model.asset.generator = gltf_string("OrcaSlicer");
        model.accessors = m_accessors.data();
        model.accessors_count = uint32_t(m_accessors.size());
        model.buffers = m_buffers.data();
        model.buffers_count = uint32_t(m_buffers.size());
        model.buffer_views = m_buffer_views.data();
        model.buffer_views_count = uint32_t(m_buffer_views.size());
        model.materials = m_materials.data();
        model.materials_count = uint32_t(m_materials.size());
        model.meshes = m_mesh_values.data();
        model.meshes_count = uint32_t(m_mesh_values.size());
        model.nodes = m_nodes.data();
        model.nodes_count = uint32_t(m_nodes.size());
        model.textures = m_textures.data();
        model.textures_count = uint32_t(m_textures.size());
        model.images = m_images.data();
        model.images_count = uint32_t(m_images.size());
        model.samplers = m_samplers.data();
        model.samplers_count = uint32_t(m_samplers.size());
        model.scenes = m_scenes.data();
        model.scenes_count = uint32_t(m_scenes.size());
        model.default_scene = 0;
        return true;
    }

private:
    tg3_str add_string(const std::string &value)
    {
        m_strings.emplace_back(value);
        const std::string &stored = m_strings.back();
        return tg3_str{stored.data(), uint32_t(stored.size())};
    }

    int append_buffer_view(const void *data, size_t size, int target, std::string &message)
    {
        if (data == nullptr || size == 0) {
            message = "The GLB export encountered an empty data buffer.";
            return TG3_INDEX_NONE;
        }
        while ((m_binary.size() & 3u) != 0u)
            m_binary.emplace_back(0);
        if (m_binary.size() > GLTF_EXPORT_MAX_BINARY_SIZE ||
            size > GLTF_EXPORT_MAX_BINARY_SIZE - m_binary.size()) {
            message = "The GLB export exceeds the 2 GiB binary buffer limit.";
            return TG3_INDEX_NONE;
        }
        const uint64_t offset = m_binary.size();
        const uint8_t *bytes = static_cast<const uint8_t *>(data);
        m_binary.insert(m_binary.end(), bytes, bytes + size);

        tg3_buffer_view view{};
        view.buffer = 0;
        view.byte_offset = offset;
        view.byte_length = size;
        view.target = target;
        m_buffer_views.emplace_back(view);
        return int(m_buffer_views.size() - 1);
    }

    int append_accessor(int buffer_view,
                        int component_type,
                        uint64_t count,
                        int type,
                        bool normalized,
                        const std::array<double, 3> *minimum,
                        const std::array<double, 3> *maximum)
    {
        tg3_accessor accessor{};
        accessor.buffer_view = buffer_view;
        accessor.component_type = component_type;
        accessor.count = count;
        accessor.type = type;
        accessor.normalized = normalized ? 1 : 0;
        accessor.sparse.indices.buffer_view = TG3_INDEX_NONE;
        accessor.sparse.values.buffer_view = TG3_INDEX_NONE;
        if (minimum != nullptr && maximum != nullptr) {
            m_minimums.emplace_back(*minimum);
            m_maximums.emplace_back(*maximum);
            accessor.min_values = m_minimums.back().data();
            accessor.min_values_count = 3;
            accessor.max_values = m_maximums.back().data();
            accessor.max_values_count = 3;
        }
        m_accessors.emplace_back(accessor);
        return int(m_accessors.size() - 1);
    }

    bool append_texture(const ModelVolume &volume, int &material_index, std::string &message)
    {
        if (m_check_cancel)
            m_check_cancel();
        size_t png_size = 0;
        void *png_data = tdefl_write_image_to_png_file_in_memory_ex(
            volume.imported_texture_rgba.data(),
            int(volume.imported_texture_width),
            int(volume.imported_texture_height),
            4,
            &png_size,
            MZ_DEFAULT_LEVEL,
            1);
        if (png_data == nullptr || png_size == 0) {
            if (png_data != nullptr)
                mz_free(png_data);
            message = "Failed to encode an image texture as PNG.";
            return false;
        }
        const int buffer_view = append_buffer_view(png_data, png_size, 0, message);
        mz_free(png_data);
        if (buffer_view < 0)
            return false;
        if (m_check_cancel)
            m_check_cancel();

        if (m_samplers.empty()) {
            tg3_sampler sampler{};
            sampler.name = add_string("Linear repeat");
            sampler.min_filter = TG3_TEXTURE_FILTER_LINEAR;
            sampler.mag_filter = TG3_TEXTURE_FILTER_LINEAR;
            sampler.wrap_s = TG3_TEXTURE_WRAP_REPEAT;
            sampler.wrap_t = TG3_TEXTURE_WRAP_REPEAT;
            m_samplers.emplace_back(sampler);
        }

        const int image_index = int(m_images.size());
        tg3_image image{};
        image.name = add_string(volume.name.empty() ? "Texture" : volume.name);
        image.width = int(volume.imported_texture_width);
        image.height = int(volume.imported_texture_height);
        image.component = 4;
        image.bits = 8;
        image.pixel_type = TG3_COMPONENT_TYPE_UNSIGNED_BYTE;
        image.buffer_view = buffer_view;
        image.mime_type = gltf_string("image/png");
        image.as_is = 1;
        m_images.emplace_back(image);

        const int texture_index = int(m_textures.size());
        tg3_texture texture{};
        texture.name = image.name;
        texture.sampler = 0;
        texture.source = image_index;
        m_textures.emplace_back(texture);

        bool blend = false;
        const uint64_t pixel_count = uint64_t(volume.imported_texture_width) *
                                     uint64_t(volume.imported_texture_height);
        for (uint64_t pixel = 0; pixel < pixel_count; ++pixel) {
            if (volume.imported_texture_rgba[size_t(pixel) * 4 + 3] < 255) {
                blend = true;
                break;
            }
        }
        material_index = int(m_materials.size());
        m_materials.emplace_back(gltf_export_material(image.name, texture_index, blend));
        return true;
    }

    bool append_primitive(const ModelVolume &volume,
                          const std::vector<size_t> &triangles,
                          bool include_uvs,
                          bool include_colors,
                          int material_index,
                          bool &vertex_alpha,
                          GltfExportMesh &mesh,
                          std::string &message)
    {
        if (triangles.empty())
            return true;

        const indexed_triangle_set &its = volume.mesh().its;
        std::vector<float> positions;
        std::vector<float> normals;
        std::vector<float> uvs;
        std::vector<uint8_t> colors;
        positions.reserve(triangles.size() * 9);
        normals.reserve(triangles.size() * 9);
        if (include_uvs)
            uvs.reserve(triangles.size() * 6);
        if (include_colors)
            colors.reserve(triangles.size() * 12);

        std::array<double, 3> minimum = {
            std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max()
        };
        std::array<double, 3> maximum = {
            -std::numeric_limits<double>::max(),
            -std::numeric_limits<double>::max(),
            -std::numeric_limits<double>::max()
        };

        const Transform3d transform = volume.get_matrix();
        const bool mirrored = transform.linear().determinant() < 0.0;
        const std::array<int, 3> corner_order = mirrored ? std::array<int, 3>{0, 2, 1} :
                                                           std::array<int, 3>{0, 1, 2};
        for (size_t triangle_offset = 0; triangle_offset < triangles.size(); ++triangle_offset) {
            if (m_check_cancel && (triangle_offset & 1023u) == 0u)
                m_check_cancel();
            const size_t triangle_idx = triangles[triangle_offset];
            if (!gltf_export_triangle_valid(its, triangle_idx))
                continue;
            const stl_triangle_vertex_indices &triangle = its.indices[triangle_idx];
            std::array<Vec3f, 3> transformed;
            for (size_t output_corner = 0; output_corner < 3; ++output_corner) {
                const int source_corner = corner_order[output_corner];
                const Vec3d local = its.vertices[size_t(triangle[source_corner])].cast<double>();
                const Vec3d millimetres = transform * local - m_object.origin_translation;
                if (!millimetres.allFinite()) {
                    message = "The object contains a non-finite transformed vertex.";
                    return false;
                }
                transformed[output_corner] = Vec3f(float(millimetres.x() * 0.001),
                                                   float(millimetres.z() * 0.001),
                                                   float(-millimetres.y() * 0.001));
                if (!transformed[output_corner].allFinite()) {
                    message = "The object contains a transformed vertex outside the GLB float range.";
                    return false;
                }
            }

            Vec3f normal = (transformed[1] - transformed[0]).cross(transformed[2] - transformed[0]);
            const float length = normal.norm();
            normal = normal.allFinite() &&
                     std::isfinite(length) &&
                     length > std::numeric_limits<float>::epsilon() ?
                         normal / length :
                         Vec3f(0.f, 1.f, 0.f);

            const size_t uv_offset = triangle_idx * 6;
            for (size_t output_corner = 0; output_corner < 3; ++output_corner) {
                const int source_corner = corner_order[output_corner];
                const Vec3f &position = transformed[output_corner];
                positions.insert(positions.end(), {position.x(), position.y(), position.z()});
                normals.insert(normals.end(), {normal.x(), normal.y(), normal.z()});
                for (size_t axis = 0; axis < 3; ++axis) {
                    minimum[axis] = std::min(minimum[axis], double(position[int(axis)]));
                    maximum[axis] = std::max(maximum[axis], double(position[int(axis)]));
                }

                if (include_uvs) {
                    const float u = volume.imported_texture_uvs_per_face[uv_offset + size_t(source_corner) * 2];
                    const float v = volume.imported_texture_uvs_per_face[uv_offset + size_t(source_corner) * 2 + 1];
                    uvs.insert(uvs.end(), {u, 1.f - v});
                }

                if (include_colors) {
                    const uint32_t rgba = volume.imported_vertex_colors_rgba[size_t(triangle[source_corner])];
                    colors.insert(colors.end(), {
                        uint8_t((rgba >> 24) & 0xFFu),
                        uint8_t((rgba >> 16) & 0xFFu),
                        uint8_t((rgba >> 8) & 0xFFu),
                        uint8_t(rgba & 0xFFu)
                    });
                    vertex_alpha |= (rgba & 0xFFu) < 0xFFu;
                }
            }
        }

        if (positions.empty())
            return true;

        GltfExportPrimitive primitive;
        primitive.value.material = material_index;
        primitive.value.indices = TG3_INDEX_NONE;
        primitive.value.mode = TG3_MODE_TRIANGLES;

        const int position_view = append_buffer_view(positions.data(), positions.size() * sizeof(float), TG3_TARGET_ARRAY_BUFFER, message);
        if (position_view < 0)
            return false;
        const int position_accessor = append_accessor(position_view,
                                                      TG3_COMPONENT_TYPE_FLOAT,
                                                      positions.size() / 3,
                                                      TG3_TYPE_VEC3,
                                                      false,
                                                      &minimum,
                                                      &maximum);
        primitive.attributes.push_back({gltf_string("POSITION"), position_accessor});

        const int normal_view = append_buffer_view(normals.data(), normals.size() * sizeof(float), TG3_TARGET_ARRAY_BUFFER, message);
        if (normal_view < 0)
            return false;
        const int normal_accessor = append_accessor(normal_view,
                                                    TG3_COMPONENT_TYPE_FLOAT,
                                                    normals.size() / 3,
                                                    TG3_TYPE_VEC3,
                                                    false,
                                                    nullptr,
                                                    nullptr);
        primitive.attributes.push_back({gltf_string("NORMAL"), normal_accessor});

        if (include_uvs) {
            const int uv_view = append_buffer_view(uvs.data(), uvs.size() * sizeof(float), TG3_TARGET_ARRAY_BUFFER, message);
            if (uv_view < 0)
                return false;
            const int uv_accessor = append_accessor(uv_view,
                                                    TG3_COMPONENT_TYPE_FLOAT,
                                                    uvs.size() / 2,
                                                    TG3_TYPE_VEC2,
                                                    false,
                                                    nullptr,
                                                    nullptr);
            primitive.attributes.push_back({gltf_string("TEXCOORD_0"), uv_accessor});
        }

        if (include_colors) {
            const int color_view = append_buffer_view(colors.data(), colors.size(), TG3_TARGET_ARRAY_BUFFER, message);
            if (color_view < 0)
                return false;
            const int color_accessor = append_accessor(color_view,
                                                       TG3_COMPONENT_TYPE_UNSIGNED_BYTE,
                                                       colors.size() / 4,
                                                       TG3_TYPE_VEC4,
                                                       true,
                                                       nullptr,
                                                       nullptr);
            primitive.attributes.push_back({gltf_string("COLOR_0"), color_accessor});
        }

        mesh.primitives.emplace_back(std::move(primitive));
        return true;
    }

    bool append_volume(const ModelVolume &volume, bool &vertex_alpha, std::string &message)
    {
        const indexed_triangle_set &its = volume.mesh().its;
        std::vector<size_t> all_triangles;
        all_triangles.reserve(its.indices.size());
        for (size_t triangle_idx = 0; triangle_idx < its.indices.size(); ++triangle_idx)
            if (gltf_export_triangle_valid(its, triangle_idx))
                all_triangles.emplace_back(triangle_idx);
        if (all_triangles.empty())
            return true;

        GltfExportMesh mesh;
        mesh.name = add_string(volume.name.empty() ? "Volume" : volume.name);

        if (m_color_mode == GltfExportColorMode::ImageTexture && !volume.imported_texture_rgba.empty()) {
            if (!gltf_export_texture_valid(volume)) {
                message = "A selected image texture contains inconsistent image or UV data.";
                return false;
            }

            int material_index = TG3_INDEX_NONE;
            if (!append_texture(volume, material_index, message))
                return false;

            std::vector<size_t> textured_triangles;
            std::vector<size_t> untextured_triangles;
            textured_triangles.reserve(all_triangles.size());
            untextured_triangles.reserve(all_triangles.size());
            for (const size_t triangle_idx : all_triangles) {
                if (gltf_export_triangle_has_uvs(volume, triangle_idx))
                    textured_triangles.emplace_back(triangle_idx);
                else
                    untextured_triangles.emplace_back(triangle_idx);
            }
            if (!append_primitive(volume, textured_triangles, true, false, material_index, vertex_alpha, mesh, message) ||
                !append_primitive(volume, untextured_triangles, false, false, 0, vertex_alpha, mesh, message))
                return false;
        } else if (m_color_mode == GltfExportColorMode::VertexColors &&
                   !volume.imported_vertex_colors_rgba.empty()) {
            if (volume.imported_vertex_colors_rgba.size() != its.vertices.size()) {
                message = "A selected vertex-color payload does not match its mesh vertices.";
                return false;
            }
            if (!append_primitive(volume, all_triangles, false, true, 0, vertex_alpha, mesh, message))
                return false;
        } else {
            if (!append_primitive(volume, all_triangles, false, false, 0, vertex_alpha, mesh, message))
                return false;
        }

        if (mesh.primitives.empty())
            return true;

        const int mesh_index = int(m_meshes.size());
        m_meshes.emplace_back(std::move(mesh));
        const int node_index = int(m_nodes.size());
        m_nodes.emplace_back(gltf_export_node(m_meshes.back().name, mesh_index));
        m_root_children.emplace_back(node_index);
        return true;
    }

    const ModelObject              &m_object;
    GltfExportColorMode             m_color_mode;
    const std::function<void()>     &m_check_cancel;
    std::deque<std::string>         m_strings;
    std::vector<uint8_t>            m_binary;
    std::vector<tg3_buffer>         m_buffers;
    std::vector<tg3_buffer_view>    m_buffer_views;
    std::vector<tg3_accessor>       m_accessors;
    std::deque<std::array<double, 3>> m_minimums;
    std::deque<std::array<double, 3>> m_maximums;
    std::vector<tg3_image>          m_images;
    std::vector<tg3_sampler>        m_samplers;
    std::vector<tg3_texture>        m_textures;
    std::vector<tg3_material>       m_materials;
    std::vector<GltfExportMesh>     m_meshes;
    std::vector<tg3_mesh>           m_mesh_values;
    std::vector<tg3_node>           m_nodes;
    std::vector<int32_t>            m_root_children;
    std::vector<int32_t>            m_scene_nodes;
    std::vector<tg3_scene>          m_scenes;
};

enum class GltfDestinationState {
    Missing,
    RegularFile,
    Invalid
};

static GltfDestinationState inspect_glb_destination(const std::string &path, std::string &message)
{
    errno = 0;
    std::FILE *file = boost::nowide::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        const int open_error = errno;
        if (open_error == ENOENT)
            return GltfDestinationState::Missing;

        message = "Failed to inspect the selected GLB destination";
        if (open_error != 0)
            message += ": " + std::string(std::strerror(open_error));
        message += ".";
        return GltfDestinationState::Invalid;
    }

#ifdef _WIN32
    struct _stat64 status;
    errno = 0;
    const int status_result = _fstat64(_fileno(file), &status);
    const bool is_regular_file = status_result == 0 && (status.st_mode & _S_IFREG) != 0;
#else
    struct stat status;
    errno = 0;
    const int status_result = ::fstat(::fileno(file), &status);
    const bool is_regular_file = status_result == 0 && S_ISREG(status.st_mode);
#endif
    const int status_error = errno;
    std::fclose(file);

    if (status_result != 0) {
        message = "Failed to inspect the selected GLB destination";
        if (status_error != 0)
            message += ": " + std::string(std::strerror(status_error));
        message += ".";
        return GltfDestinationState::Invalid;
    }
    if (!is_regular_file) {
        message = "The selected GLB destination is not a regular file.";
        return GltfDestinationState::Invalid;
    }

    return GltfDestinationState::RegularFile;
}

static bool write_glb_file(const std::string &path, const uint8_t *data, uint64_t size, std::string &message)
{
    if (data == nullptr || size == 0 || size > uint64_t(std::numeric_limits<std::streamsize>::max())) {
        message = "The generated GLB data is invalid or too large to write.";
        return false;
    }

    const boost::filesystem::path target(path);
    const boost::filesystem::path parent = target.parent_path().empty() ? boost::filesystem::current_path() :
                                                                          target.parent_path();
    const GltfDestinationState destination_state = inspect_glb_destination(path, message);
    if (destination_state == GltfDestinationState::Invalid)
        return false;
    const bool target_exists = destination_state == GltfDestinationState::RegularFile;
    const boost::filesystem::path temporary =
        parent / boost::filesystem::unique_path(".orcaslicer-glb-%%%%-%%%%-%%%%.tmp");
    const boost::filesystem::path backup =
        parent / boost::filesystem::unique_path(".orcaslicer-glb-%%%%-%%%%-%%%%.bak");

    {
        errno = 0;
        boost::nowide::ofstream output(temporary.string(), std::ios::binary | std::ios::trunc);
        if (!output.is_open()) {
            const int open_error = errno;
            message = "Failed to create a temporary GLB file in the selected folder";
            if (open_error != 0)
                message += ": " + std::string(std::strerror(open_error));
            message += ".";
            return false;
        }
        output.write(reinterpret_cast<const char *>(data), std::streamsize(size));
        output.flush();
        if (!output.good()) {
            output.close();
            boost::nowide::remove(temporary.string().c_str());
            message = "Failed while writing the GLB output file.";
            return false;
        }
    }

    if (target_exists) {
        const std::error_code backup_error = rename_file(target.string(), backup.string());
        if (backup_error) {
            boost::nowide::remove(temporary.string().c_str());
            message = "Failed to preserve the existing destination GLB file.";
            return false;
        }
    }

    const std::error_code replace_error = rename_file(temporary.string(), target.string());
    if (replace_error) {
        boost::nowide::remove(temporary.string().c_str());
        if (target_exists)
            rename_file(backup.string(), target.string());
        message = "Failed to move the completed GLB file to its destination.";
        return false;
    }

    if (target_exists)
        boost::nowide::remove(backup.string().c_str());
    return true;
}

} // namespace

bool load_gltf(const char *path, Model *model, GltfImportInfo &import_info, std::string &message)
{
    import_info = {};
    message.clear();
    if (path == nullptr || model == nullptr)
        return false;

    tg3_model parsed_model{};
    tg3_error_stack errors{};
    tg3_error_stack_init(&errors);

    tg3_parse_options options{};
    tg3_parse_options_init(&options);
    options.images_as_is = 1;
    options.validate_indices = 1;
    options.max_external_file_size = 1024ull * 1024ull * 1024ull;
    options.fs.read_file = tg3_read_file;
    options.fs.free_file = tg3_free_file;

    const std::string input_path(path);
    const tg3_error_code parse_result = tg3_parse_file(&parsed_model,
                                                       &errors,
                                                       path,
                                                       uint32_t(input_path.size()),
                                                       &options);
    if (parse_result != TG3_OK) {
        message = first_error_message(errors);
        if (message.empty())
            message = "load_gltf: failed to parse";
        tg3_model_free(&parsed_model);
        tg3_error_stack_free(&errors);
        return false;
    }

    indexed_triangle_set its;
    GltfTextureState texture_state;
    const boost::filesystem::path input_fs_path(input_path);
    const boost::filesystem::path base_dir = input_fs_path.parent_path();
    const std::vector<int> roots = scene_root_nodes(parsed_model);
    std::vector<uint8_t> visited(parsed_model.nodes_count, 0);
    for (int node_idx : roots)
        append_node_meshes(parsed_model,
                           base_dir,
                           node_idx,
                           Eigen::Matrix4d::Identity(),
                           visited,
                           its,
                           texture_state,
                           import_info);

    if (its.indices.empty() || its.vertices.empty()) {
        message = "load_gltf: file contains no supported mesh triangles";
        tg3_model_free(&parsed_model);
        tg3_error_stack_free(&errors);
        return false;
    }

    TriangleMesh mesh(std::move(its));
    if (mesh.volume() < 0.0) {
        mesh.flip_triangles();
        for (std::array<Vec2f, 3> &uvs : import_info.triangle_uvs)
            std::swap(uvs[1], uvs[2]);
    }

    const std::string object_name = input_fs_path.stem().empty() ? input_fs_path.filename().string() : input_fs_path.stem().string();
    model->add_object(object_name.c_str(), path, std::move(mesh));

    finish_import_info(import_info);

    tg3_model_free(&parsed_model);
    tg3_error_stack_free(&errors);
    return true;
}

bool store_glb(const char *path,
               const ModelObject &object,
               GltfExportColorMode color_mode,
               std::string &message,
               const std::function<void()> &check_cancel)
{
    message.clear();
    if (path == nullptr || *path == '\0') {
        message = "No GLB output path was provided.";
        return false;
    }

    tg3_model export_model{};
    GltfExportBuilder builder(object, color_mode, check_cancel);
    if (!builder.build(export_model, message))
        return false;
    if (check_cancel)
        check_cancel();

    tg3_error_stack errors{};
    tg3_error_stack_init(&errors);
    tg3_write_options options{};
    tg3_write_options_init(&options);
    options.pretty_print = 0;
    options.write_binary = 1;

    uint8_t *output_data = nullptr;
    uint64_t output_size = 0;
    const tg3_error_code write_result =
        tg3_write_to_memory(&export_model, &errors, &output_data, &output_size, &options);
    if (write_result != TG3_OK || output_data == nullptr || output_size == 0) {
        message = first_error_message(errors);
        if (message.empty())
            message = "Failed to serialize the object as GLB.";
        if (output_data != nullptr)
            tg3_write_free(output_data, &options);
        tg3_error_stack_free(&errors);
        return false;
    }

    if (check_cancel) {
        try {
            check_cancel();
        } catch (...) {
            tg3_write_free(output_data, &options);
            tg3_error_stack_free(&errors);
            throw;
        }
    }
    bool written = false;
    try {
        written = write_glb_file(path, output_data, output_size, message);
    } catch (...) {
        tg3_write_free(output_data, &options);
        tg3_error_stack_free(&errors);
        throw;
    }
    tg3_write_free(output_data, &options);
    tg3_error_stack_free(&errors);
    return written;
}

} // namespace Slic3r
