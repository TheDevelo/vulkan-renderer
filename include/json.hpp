#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// JSON parser inspired by sejp. Now with more recursion!
namespace json {
    class Value; // Forward declaration because C++ :)

    // Backing data for a full JSON hierarchy
    // Each JSON value references the hierachy to get its actual value
    // We'll use indexes to reference into the backing hierarchy struct. This'll be fine since we aren't going to modify the JSON output after parsing
    struct Hierarchy {
        std::vector<std::optional<std::map<std::string, Value>>> objects;
        std::vector<std::optional<std::vector<Value>>> arrays;
        std::vector<std::optional<std::string>> strings;
        std::vector<std::optional<double>> numbers;
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

        std::optional<std::map<std::string, Value>> const& as_obj() const;
        std::optional<std::vector<Value>> const& as_arr() const;
        std::optional<std::string> const& as_str() const;
        std::optional<double> const& as_num() const;
        std::optional<bool> const& as_bool() const;
        std::optional<std::nullptr_t> const& as_null() const;
//    private:
        std::shared_ptr<Hierarchy> data;
        uint32_t index;
        JsonType valType;
    };

    // JSON parsing functions
    Value load(std::string const& filename);
    Value parse(std::string const& jsonStr);
}
