#pragma once

#include <array>
#include <cstdlib>

// Generic vector type
template<size_t N, typename T>
struct Vec {
    std::array<T, N> e;

    Vec() = default;

    explicit Vec(T val) {
        for (size_t i = 0; i < N; ++i) {
            e[i] = val;
        }
    }

    Vec(const std::initializer_list<T> &list) {
        size_t i = 0;
        for (T& elem : list) {
            // Bail out if there are too many specifiers
            if (i > N) {
                break;
            }

            e[i] = elem;
            i++;
        }
    }

    T operator[](size_t i) const {
        return e[i];
    }
    T& operator[](size_t i) {
        return e[i];
    }
};

template<size_t N, typename T>
Vec<N, T> operator+(Vec<N, T> const& a, Vec<N, T> const& b) {
    Vec<N, T> result;
    for (unsigned int i = 0; i < N; i++) {
        result.e[i] = a.e[i] + b.e[i];
    }
    return result;
}

template<size_t N, typename T>
Vec<N, T> operator-(Vec<N, T> const& a, Vec<N, T> const& b) {
    Vec<N, T> result;
    for (unsigned int i = 0; i < N; i++) {
        result.e[i] = a.e[i] - b.e[i];
    }
    return result;
}

template<size_t N, typename T>
Vec<N, T> operator*(Vec<N, T> const& a, Vec<N, T> const& b) {
    Vec<N, T> result;
    for (unsigned int i = 0; i < N; i++) {
        result.e[i] = a.e[i] * b.e[i];
    }
    return result;
}

template<size_t N, typename T>
Vec<N, T> operator/(Vec<N, T> const& a, Vec<N, T> const& b) {
    Vec<N, T> result;
    for (unsigned int i = 0; i < N; i++) {
        result.e[i] = a.e[i] / b.e[i];
    }
    return result;
}

// Specific Vec2,3,4 types
template<typename T>
struct Vec<2, T> {
    union {
        std::array<T, 2> e;
        struct {
            T x;
            T y;
        };
        struct {
            T u;
            T v;
        };
    };

    constexpr Vec() = default;
    constexpr explicit Vec(T val) : x(val), y(val) {}
    constexpr Vec(T x_in, T y_in) : x(x_in), y(y_in) {}

    T operator[](size_t i) const {
        return e[i];
    }
    T& operator[](size_t i) {
        return e[i];
    }
};

template<typename T>
struct Vec<3, T> {
    union {
        std::array<T, 2> e;
        struct {
            T x;
            T y;
            T z;
        };
        struct {
            T r;
            T g;
            T b;
        };
        Vec<2, T> xy;
    };

    constexpr Vec() = default;
    constexpr explicit Vec(T val) : x(val), y(val), z(val) {}
    constexpr Vec(T x_in, T y_in, T z_in) : x(x_in), y(y_in), z(z_in) {}
    constexpr Vec(Vec<2, T> xy_in, T z_in) : x(xy_in.x), y(xy_in.y), z(z_in) {}

    T operator[](size_t i) const {
        return e[i];
    }
    T& operator[](size_t i) {
        return e[i];
    }
};

template<typename T>
struct Vec<4, T> {
    union {
        std::array<T, 4> e;
        struct {
            T x;
            T y;
            T z;
            T w;
        };
        Vec<2, T> xy;
        Vec<3, T> xyz;
    };

    constexpr Vec() = default;
    constexpr explicit Vec(T val) : x(val), y(val), z(val), w(val) {}
    constexpr Vec(T x_in, T y_in, T z_in, T w_in) : x(x_in), y(y_in), z(z_in), w(w_in) {}
    constexpr Vec(Vec<2, T> xy_in, T z_in, T w_in) : x(xy_in.x), y(xy_in.y), z(z_in), w(w_in) {}
    constexpr Vec(Vec<3, T> xyz_in, T w_in) : x(xyz_in.x), y(xyz_in.y), z(xyz_in.z), w(w_in) {}

    T operator[](size_t i) const {
        return e[i];
    }
    T& operator[](size_t i) {
        return e[i];
    }
};

template <typename T> using Vec2 = Vec<2, T>;
template <typename T> using Vec3 = Vec<3, T>;
template <typename T> using Vec4 = Vec<4, T>;
