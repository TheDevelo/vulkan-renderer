#pragma once
#include <array>
#include <cstdlib>
#include <cmath>

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
            if (i >= N) {
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

// Element-wise operations
template<size_t N, typename T>
Vec<N, T> operator+(Vec<N, T> const& a, Vec<N, T> const& b) {
    Vec<N, T> result;
    for (size_t i = 0; i < N; i++) {
        result.e[i] = a.e[i] + b.e[i];
    }
    return result;
}

template<size_t N, typename T>
Vec<N, T> operator-(Vec<N, T> const& a, Vec<N, T> const& b) {
    Vec<N, T> result;
    for (size_t i = 0; i < N; i++) {
        result.e[i] = a.e[i] - b.e[i];
    }
    return result;
}

template<size_t N, typename T>
Vec<N, T> operator*(Vec<N, T> const& a, Vec<N, T> const& b) {
    Vec<N, T> result;
    for (size_t i = 0; i < N; i++) {
        result.e[i] = a.e[i] * b.e[i];
    }
    return result;
}

template<size_t N, typename T>
Vec<N, T> operator/(Vec<N, T> const& a, Vec<N, T> const& b) {
    Vec<N, T> result;
    for (size_t i = 0; i < N; i++) {
        result.e[i] = a.e[i] / b.e[i];
    }
    return result;
}

// Scalar multiplication / division
template<size_t N, typename T>
Vec<N, T> operator*(T const& s, Vec<N, T> const& v) {
    Vec<N, T> result;
    for (size_t i = 0; i < N; i++) {
        result.e[i] = s * v.e[i];
    }
    return result;
}

template<size_t N, typename T>
Vec<N, T> operator*(Vec<N, T> const& v, T const& s) {
    return s * v;
}

template<size_t N, typename T>
Vec<N, T> operator/(Vec<N, T> const& v, T const& s) {
    return (static_cast<T>(1) / s) * v;
}

// Length/Normalization
namespace linear {
    template<size_t N, typename T>
    T dot(const Vec<N, T>& a, const Vec<N, T>& b) {
        T result = static_cast<T>(0);
        for (size_t i = 0; i < N; i++) {
            result += a.e[i] * b.e[i];
        }

        return result;
    }

    template<size_t N, typename T>
    T length2(const Vec<N, T>& v) {
        return dot(v, v);
    }

    template<size_t N, typename T>
    T length(const Vec<N, T>& v) {
        return std::sqrt(length2(v));
    }

    template<size_t N, typename T>
    Vec<N, T> normalize(const Vec<N, T>& v) {
        return v / length(v);
    }
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
        std::array<T, 3> e;
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

// Cross product for Vec3
namespace linear {
    template<typename T>
    Vec3<T> cross(Vec3<T> const& a, Vec3<T> const& b) {
        Vec3<T> result;
        result.x = a.y * b.z - a.z * b.y;
        result.y = a.z * b.x - a.x * b.z;
        result.z = a.x * b.y - a.y * b.x;

        return result;
    }
}

// Matrices are stored in column-major format
// N = # rows, M = # columns
template<size_t N, size_t M, typename T>
struct Mat {
    std::array<T, N * M> e;

    Mat() = default;

    explicit Mat(T val) {
        for (size_t i = 0; i < N * M; ++i) {
            e[i] = val;
        }
    }

    Mat(const std::initializer_list<T> &list) {
        size_t i = 0;
        for (T const& elem : list) {
            // Bail out if there are too many specifiers
            if (i >= N * M) {
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

    inline T at(size_t col, size_t row) const {
        return e[row + col * N];
    }
    inline T& at(size_t col, size_t row) {
        return e[row + col * N];
    }
};

// Matrix calculations
namespace linear {
    template<size_t N, size_t M, size_t P, typename T>
    Mat<N, P, T> mmul(Mat<N, M, T> A, Mat<M, P, T> B) {
        Mat<N, P, T> result;
        for (size_t col = 0; col < P; col++) {
            for (size_t row = 0; row < N; row++) {
                T dot = static_cast<T>(0);
                for (size_t i = 0; i < M; i++) {
                    dot += A.at(i, row) * B.at(col, i);
                }
                result.at(col, row) = dot;
            }
        }

        return result;
    }

    template<size_t N, size_t M, typename T>
    Vec<N, T> mmul(Mat<N, M, T> A, Vec<M, T> x) {
        Vec<N, T> result;
        for (size_t i = 0; i < N; i++) {
            T dot = static_cast<T>(0);
            for (size_t j = 0; j < M; j++) {
                dot += A.at(j, i) * x.e[j];
            }
            result[i] = dot;
        }

        return result;
    }
}

template <typename T> using Mat4 = Mat<4, 4, T>;

// Useful constants
namespace linear {
    const Mat4<float> M4F_IDENTITY = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
}

// Useful matrix/vector calc helper functions
namespace linear {
    // Perspective matrix calculation
    template<typename T>
    Mat4<T> perspective(T fovy, T aspect, T zNear, T zFar) {
        T const invTanHalfFovy = static_cast<T>(1) / std::tan(fovy / static_cast<T>(2));

        Mat4<T> result(static_cast<T>(0));
        result.at(0, 0) = invTanHalfFovy / aspect;
        result.at(1, 1) = -invTanHalfFovy;
        result.at(2, 2) = -zFar / (zFar - zNear);
        result.at(2, 3) = static_cast<T>(-1);
        result.at(3, 2) = -(zFar * zNear) / (zFar - zNear);
        return result;
    }

    // Infinite perspective matrix, calculated as the limit of perspective as zFar -> infinity
    template<typename T>
    Mat4<T> infinitePerspective(T fovy, T aspect, T zNear) {
        T const invTanHalfFovy = static_cast<T>(1) / std::tan(fovy / static_cast<T>(2));

        Mat4<T> result(static_cast<T>(0));
        result.at(0, 0) = invTanHalfFovy / aspect;
        result.at(1, 1) = -invTanHalfFovy;
        result.at(2, 2) = static_cast<T>(-1);
        result.at(2, 3) = static_cast<T>(-1);
        result.at(3, 2) = -zNear;
        return result;
    }

    // Rotation matrix calculation adapted from glm/formula from wikipedia's "Rotation matrix" page
    template<typename T>
    Mat4<T> rotate(T angle, Vec3<T> axisIn) {
        T const cos = std::cos(angle);
        T const sin = std::sin(angle);

        Vec3<T> axis = normalize(axisIn);
        Vec3<T> outer = (static_cast<T>(1) - cos) * axis;

        // Matrix is (cos T)I + (sin T)[axis]_x + (1 - cos T)[u (x) u]. [u (x) u] is the outer product, so outer[i] * axis[j] gives the ijth entry of (1 - cos T)[u (x) u]
        Mat4<T> result(static_cast<T>(0));
        result.at(0, 0) = outer[0] * axis[0] + cos;
        result.at(0, 1) = outer[0] * axis[1] + sin * axis[2];
        result.at(0, 2) = outer[0] * axis[2] - sin * axis[1];

        result.at(1, 0) = outer[1] * axis[0] - sin * axis[2];
        result.at(1, 1) = outer[1] * axis[1] + cos;
        result.at(1, 2) = outer[1] * axis[2] + sin * axis[0];

        result.at(2, 0) = outer[2] * axis[0] + sin * axis[1];
        result.at(2, 1) = outer[2] * axis[1] - sin * axis[0];
        result.at(2, 2) = outer[2] * axis[2] + cos;

        result.at(3, 3) = static_cast<T>(1);

        return result;
    }

    // View matrix helper that points a camera at eye to center, with up being the direction of "up"
    template<typename T>
    Mat4<T> lookAt(Vec3<T> eye, Vec3<T> center, Vec3<T> up) {
        // We calculate the forward, side, and up directions that the XYZ axes get translated to
        Vec3<T> const f = normalize(center - eye); // Forward
        Vec3<T> const s = normalize(cross(f, up)); // Side
        Vec3<T> const u = cross(f, s); // Orthonormal Up (as up isn't guaranteed to be orthonormal to f and s)

        // NOTE: initializer_list just copies to the array, so this is actually the transpose of the real matrix
        Mat4<T> result {
            s.x, -u.x, -f.x, static_cast<T>(0),
            s.y, -u.y, -f.y, static_cast<T>(0),
            s.z, -u.z, -f.z, static_cast<T>(0),
            -dot(s, eye), dot(u, eye), dot(f, eye), static_cast<T>(1)
        };
        return result;
    }

    // Helpers to make parentToLocal and localToParent matrices from scale, quaternion rotation, and translation
    // Quaternion rotation was adapted from Wikipedia
    template<typename T>
    Mat4<T> localToParent(Vec3<T> t, Vec4<T> q, Vec3<T> s) {
        Mat4<T> result {
            s.x * (1 - 2*(q.y*q.y + q.z*q.z)), s.x *      2*(q.x*q.y + q.z*q.w) , s.x *      2*(q.x*q.z - q.y*q.w) , static_cast<T>(0),
            s.y *      2*(q.x*q.y - q.z*q.w) , s.y * (1 - 2*(q.x*q.x + q.z*q.z)), s.y *      2*(q.y*q.z + q.x*q.w) , static_cast<T>(0),
            s.z *      2*(q.x*q.z + q.y*q.w) , s.z *      2*(q.y*q.z - q.x*q.w) , s.z * (1 - 2*(q.x*q.x + q.y*q.y)), static_cast<T>(0),
            t.x                              , t.y                              , t.z                              , static_cast<T>(1),
        };
        return result;
    }

    template<typename T>
    Mat4<T> parentToLocal(Vec3<T> tn, Vec4<T> qn, Vec3<T> sn) {
        // Invert our translation, rotation, and scale. Then adjust our inverted translation so that it ends up corect after rotation/scaling
        Vec4<T> t = Vec4<T>(-tn.x, -tn.y, -tn.z, static_cast<T>(1));
        Vec4<T> q = Vec4<T>(-qn.x, -qn.y, -qn.z, qn.w);
        Vec3<T> s = Vec3<T>(1 / sn.x, 1 / sn.y, 1 / sn.z);

        Mat4<T> result {
            s.x * (1 - 2*(q.y*q.y + q.z*q.z)), s.y *      2*(q.x*q.y + q.z*q.w) , s.z *      2*(q.x*q.z - q.y*q.w) , static_cast<T>(0),
            s.x *      2*(q.x*q.y - q.z*q.w) , s.y * (1 - 2*(q.x*q.x + q.z*q.z)), s.z *      2*(q.y*q.z + q.x*q.w) , static_cast<T>(0),
            s.x *      2*(q.x*q.z + q.y*q.w) , s.y *      2*(q.y*q.z - q.x*q.w) , s.z * (1 - 2*(q.x*q.x + q.y*q.y)), static_cast<T>(0),
            static_cast<T>(0)                , static_cast<T>(0)                , static_cast<T>(0)                , static_cast<T>(1),
        };
        t = mmul(result, t);
        result.at(3, 0) = t.x;
        result.at(3, 1) = t.y;
        result.at(3, 2) = t.z;
        return result;
    }
}

// Radian & Degree helpers
#define DEG2RAD(x) (x * M_PI / 180.0)
#define RAD2DEG(x) (x * 180.0 / M_PI)
#define DEG2RADF(x) static_cast<float>(DEG2RAD(x))
#define RAD2DEGF(x) static_cast<float>(RAD2DEG(x))
