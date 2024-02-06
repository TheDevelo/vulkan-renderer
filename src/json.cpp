#include <charconv>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "json.hpp"

// Helpers for parsing from the stream
void skipWhitespace(std::istream& stream) {
    char c = stream.peek();
    while (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        stream.get();
        c = stream.peek();
    }
}

char peekEOF(std::istream& stream) {
    char c = stream.peek();
    if (c == std::iostream::traits_type::eof()) {
        throw std::runtime_error("JSON parsing error: reached EOF");
    }

    return c;
}

char getEOF(std::istream& stream) {
    char c = stream.get();
    if (c == std::iostream::traits_type::eof()) {
        throw std::runtime_error("JSON parsing error: reached EOF");
    }

    return c;
}

void consumeExact(std::istream& stream, std::string const& expectation) {
    for (char expectedC : expectation) {
        char c = getEOF(stream);
        if (c != expectedC) {
            throw std::runtime_error("JSON parsing error: literal does not match expected value");
        }
    }
}

std::string readString(std::istream& stream) {
    std::string result;
    for (char c = getEOF(stream); c != '"'; c = getEOF(stream)) {
        if (c == '\\') {
            // Handle escapes (except \u because thats a gnarly case I don't want to deal with)
            c = getEOF(stream);
            if (c == '\\' || c == '/' || c == '"') {
                result += c;
            }
            else if (c == 'b') {
                result += '\b';
            }
            else if (c == 'f') {
                result += '\f';
            }
            else if (c == 'n') {
                result += '\n';
            }
            else if (c == 'r') {
                result += '\r';
            }
            else if (c == 't') {
                result += '\t';
            }
            else {
                throw std::runtime_error("JSON parsing error: encountered invalid string escape code");
            }
        }
        else {
            result += c;
        }
    }

    return result;
}

double readNumber(std::istream& stream) {
    // We'll have std::from_chars handle the actual parsing into a double, so the initial parsing is determining the extent of the number + verifying it
    std::string numStr;

    // Advance to and get first digit
    char c = getEOF(stream);
    numStr += c;
    if (c == '-') {
        c = getEOF(stream);
        numStr += c;
    }

    // Check if we need to add integer values
    if (c >= '1' && c <= '9') {
        // Add as many digits as possible
        c = peekEOF(stream);
        while (c >= '0' && c <= '9') {
            // Need to get then peek to not double count the first character
            numStr += c;
            getEOF(stream);
            c = peekEOF(stream);
        }
    }
    else if (c != '0') {
        // Not a digit
        throw std::runtime_error("JSON parsing error: expected digit in number, found non-digit");
    }

    // Fractional part
    c = peekEOF(stream);
    if (c == '.') {
        // Add as many digits as possible
        numStr += getEOF(stream);

        c = peekEOF(stream);
        if (c < '0' || c > '9') {
            throw std::runtime_error("JSON parsing error: no digits in fractional part");
        }
        while (c >= '0' && c <= '9') {
            numStr += c;
            getEOF(stream);
            c = peekEOF(stream);
        }
    }

    // Exponential part
    c = peekEOF(stream);
    if (c == 'e' || c == 'E') {
        // Add as many digits as possible
        numStr += getEOF(stream);
        c = peekEOF(stream);
        if (c == '-' || c == '+') {
            numStr += getEOF(stream);
        }

        c = peekEOF(stream);
        if (c < '0' || c > '9') {
            throw std::runtime_error("JSON parsing error: no digits in exponential part");
        }
        while (c >= '0' && c <= '9') {
            numStr += c;
            getEOF(stream);
            c = peekEOF(stream);
        }
    }

    // Now parse the string into a double
    double result;
    std::from_chars(numStr.data(), numStr.data() + numStr.size(), result);
    return result;
}

namespace json {
    // Recursive helper that parses an individual value
    Value parseValue(std::shared_ptr<Hierarchy> const& data, std::istream& stream) {
        Value v;
        skipWhitespace(stream);

        // Determine the type of what we are parsing based on the first character
        char c = peekEOF(stream);
        if (c == '{') {
            // Object
            stream.get(); // Chomp off the initial {
            std::map<std::string, Value> objVals;

            // Check if our object is fully empty, or if it has values within
            skipWhitespace(stream);
            if (peekEOF(stream) == '}') {
                // Special case - object is fully empty
                stream.get(); // Chomp off the last }
            }
            else {
                // Start populating elements of the object
                while (true) {
                    // Get the key string
                    skipWhitespace(stream);
                    c = getEOF(stream);
                    if (c != '"') {
                        throw std::runtime_error("JSON parsing error: expected string as object key, found non-string");
                    }
                    std::string key = readString(stream);

                    // Get the value
                    skipWhitespace(stream);
                    c = getEOF(stream);
                    if (c != ':') {
                        throw std::runtime_error("JSON parsing error: expected : between key and value");
                    }
                    Value objVal = parseValue(data, stream);
                    objVals.insert_or_assign(key, objVal);

                    // Check for the object end or a comma
                    c = getEOF(stream);
                    if (c == '}') {
                        // object ends
                        break;
                    }
                    else if (c != ',') {
                        // Not a comma or an object end, so invalid
                        throw std::runtime_error("JSON parsing error: found invalid character between object values");
                    }
                }
            }

            data->objects.push_back(objVals);
            v = Value(data, data->objects.size() - 1, JSON_OBJ);
        }
        else if (c == '[') {
            // Array
            stream.get(); // Chomp off the initial [
            std::vector<Value> arrVals;

            // Check if our array is fully empty, or if it has values within
            skipWhitespace(stream);
            if (peekEOF(stream) == ']') {
                // Special case - array is fully empty
                stream.get(); // Chomp off the last ]
            }
            else {
                // Start populating elements of the array
                while (true) {
                    arrVals.emplace_back(parseValue(data, stream));
                    c = getEOF(stream);
                    if (c == ']') {
                        // Array ends
                        break;
                    }
                    else if (c != ',') {
                        // Not a comma or an array end, so invalid
                        throw std::runtime_error("JSON parsing error: found invalid character between array values");
                    }
                }
            }

            data->arrays.push_back(arrVals);
            v = Value(data, data->arrays.size() - 1, JSON_ARR);
        }
        else if (c == '"') {
            // String
            stream.get(); // Chomp off the initial "
            std::string value = readString(stream);
            data->strings.emplace_back(value);
            v = Value(data, data->strings.size() - 1, JSON_STR);
        }
        else if (c == '-' || (c >= '0' && c <= '9')) {
            // Number
            double value = readNumber(stream);
            data->numbers.emplace_back(value);
            v = Value(data, data->numbers.size() - 1, JSON_NUM);
        }
        else if (c == 't') {
            // True bool
            consumeExact(stream, "true");
            v = Value(data, 0, JSON_BOOL_TRUE);
        }
        else if (c == 'f') {
            // False bool
            consumeExact(stream, "false");
            v = Value(data, 0, JSON_BOOL_FALSE);
        }
        else if (c == 'n') {
            // Null
            consumeExact(stream, "null");
            v = Value(data, 0, JSON_NULL);
        }
        else {
            throw std::runtime_error("JSON parsing error: encountered invalid value");
        }

        skipWhitespace(stream);

        return v;
    }

    Value parseStream(std::istream& stream) {
        std::shared_ptr<Hierarchy> data = std::make_shared<Hierarchy>();
        Value root = parseValue(data, stream);

        if (stream.peek() != std::iostream::traits_type::eof()) {
            throw std::runtime_error("JSON parsing error: file has unused trailing data");
        }

        return root;
    }

    Value load(std::string const& filename) {
        std::ifstream in(filename, std::ios::binary);
        return parseStream(in);
    }

    Value parse(std::string const& jsonStr) {
        std::istringstream in(jsonStr, std::ios::binary);
        return parseStream(in);
    }

    std::optional<std::map<std::string, Value>> const& Value::as_obj() const {
        static std::optional<std::map<std::string, Value>> const empty;
        if (valType == JSON_OBJ) {
            return data->objects[index];
        }
        else {
            return empty;
        }
    }

    std::optional<std::vector<Value>> const& Value::as_arr() const {
        static std::optional<std::vector<Value>> const empty;
        if (valType == JSON_ARR) {
            return data->arrays[index];
        }
        else {
            return empty;
        }
    }

    std::optional<std::string> const& Value::as_str() const {
        static std::optional<std::string> const empty;
        if (valType == JSON_STR) {
            return data->strings[index];
        }
        else {
            return empty;
        }
    }

    std::optional<double> const& Value::as_num() const {
        static std::optional<double> const empty;
        if (valType == JSON_NUM) {
            return data->numbers[index];
        }
        else {
            return empty;
        }
    }

    std::optional<bool> const& Value::as_bool() const {
        static std::optional<bool> const true_value(true);
        static std::optional<bool> const false_value(true);
        static std::optional<bool> const empty;
        if (valType == JSON_BOOL_TRUE) {
            return true_value;
        }
        else if (valType == JSON_BOOL_FALSE) {
            return false_value;
        }
        else {
            return empty;
        }
    }

    std::optional<std::nullptr_t> const& Value::as_null() const {
        static std::optional<std::nullptr_t> const null_value(nullptr);
        static std::optional<std::nullptr_t> const empty;
        if (valType == JSON_NULL) {
            return null_value;
        }
        else {
            return empty;
        }
    }
};
