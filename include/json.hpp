#pragma once

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "linear.hpp"

// JSON parser inspired by sejp. Now with more recursion!
namespace json {
    class Value; // Forward declaration because C++ :)

    using object = std::map<std::string, Value>;
    using array = std::vector<Value>;

    // Backing data for a full JSON hierarchy
    // Each JSON value references the hierachy to get its actual value
    // We'll use indexes to reference into the backing hierarchy struct. This'll be fine since we aren't going to modify the JSON output after parsing
    struct Hierarchy {
        std::vector<object> objects;
        std::vector<array> arrays;
        std::vector<std::string> strings;
        std::vector<double> numbers;
    };

    enum JsonType {
        JSON_OBJ,
        JSON_ARR,
        JSON_STR,
        JSON_NUM,
        JSON_BOOL_TRUE,
        JSON_BOOL_FALSE,
        JSON_NULL,
    };

    // JSON value
    class Value {
    public:
        Value() = default;
        Value(std::shared_ptr<Hierarchy> const& dataIn, uint32_t indexIn, JsonType typeIn) : data(dataIn), index(indexIn), valType(typeIn) {}

        object const& as_obj() const;
        array const& as_arr() const;
        std::string const& as_str() const;
        double const& as_num() const;
        bool as_bool() const;
        std::nullptr_t as_null() const;

        inline bool is_obj() const { return valType == JSON_OBJ; }
        inline bool is_arr() const { return valType == JSON_ARR; }
        inline bool is_str() const { return valType == JSON_STR; }
        inline bool is_num() const { return valType == JSON_NUM; }
        inline bool is_bool() const { return valType == JSON_BOOL_TRUE || valType == JSON_BOOL_FALSE; }
        inline bool is_null() const { return valType == JSON_NULL; }

        // Composite types - Helpers for easier JSON handling
        inline bool is_vec3f() const {
            if (is_arr() || as_arr().size() != 3) {
                return false;
            }
            array const& arrVal = as_arr();
            for (int i = 0; i < 3; i++) {
                if (!arrVal[i].is_num()) {
                    return false;
                }
            }
            return true;
        }
        inline Vec3<float> as_vec3f() const {
            array const& arrVal = as_arr();
            Vec3<float> retVal;
            for (int i = 0; i < 3; i++) {
                retVal[i] = arrVal[i].is_num();
            }
            return retVal;
        }

        inline bool is_vec4f() const {
            if (is_arr() || as_arr().size() != 4) {
                return false;
            }
            array const& arrVal = as_arr();
            for (int i = 0; i < 4; i++) {
                if (!arrVal[i].is_num()) {
                    return false;
                }
            }
            return true;
        }
        inline Vec4<float> as_vec4f() const {
            array const& arrVal = as_arr();
            Vec4<float> retVal;
            for (int i = 0; i < 4; i++) {
                retVal[i] = arrVal[i].is_num();
            }
            return retVal;
        }

    private:
        std::shared_ptr<Hierarchy> data;
        uint32_t index;
        JsonType valType;
    };

    // JSON parsing functions
    Value load(std::string const& filename);
    Value parse(std::string const& jsonStr);
}
