#pragma once

template<unsigned int N, typename T>
class vec {
private:
    T value[N];

public:
    vec<N, T> operator+(vec<N, T> const& other) {
        vec<N, T> result;
        for (unsigned int i = 0; i < N; i++) {
            result.value[i] = value[i] + other.value[i];
        }
        return result;
    }

    vec<N, T> operator-(vec<N, T> const& other) {
        vec<N, T> result;
        for (unsigned int i = 0; i < N; i++) {
            result.value[i] = value[i] - other.value[i];
        }
        return result;
    }

    vec<N, T> operator*(vec<N, T> const& other) {
        vec<N, T> result;
        for (unsigned int i = 0; i < N; i++) {
            result.value[i] = value[i] * other.value[i];
        }
        return result;
    }

    vec<N, T> operator/(vec<N, T> const& other) {
        vec<N, T> result;
        for (unsigned int i = 0; i < N; i++) {
            result.value[i] = value[i] / other.value[i];
        }
        return result;
    }

    T& x() {
        return value[0];
    }

    T& y() {
        return value[1];
    }

    T& z() {
        return value[2];
    }

    T& w() {
        return value[3];
    }
};

template<typename T>
class vec2 {
public:
    T x;
    T y;

    vec2<T> operator+(vec2<T> const& other) {
        return vec2<T>(x + other.x, y + other.y);
    }

    vec2<T> operator-(vec2<T> const& other) {
        return vec2<T>(x - other.x, y - other.y);
    }

    vec2<T> operator*(vec2<T> const& other) {
        return vec2<T>(x * other.x, y * other.y);
    }

    vec2<T> operator/(vec2<T> const& other) {
        return vec2<T>(x / other.x, y / other.y);
    }

    vec2(T x, T y) : x(x), y(y) {
    }
};

template<typename T>
class vec3 {
public:
    T x;
    T y;
    T z;

    vec3<T> operator+(vec3<T> const& other) {
        return vec3<T>(x + other.x, y + other.y, z + other.z);
    }

    vec3<T> operator-(vec3<T> const& other) {
        return vec3<T>(x - other.x, y - other.y, z - other.z);
    }

    vec3<T> operator*(vec3<T> const& other) {
        return vec3<T>(x * other.x, y * other.y, z * other.z);
    }

    vec3<T> operator/(vec3<T> const& other) {
        return vec3<T>(x / other.x, y / other.y, z / other.z);
    }

    vec3(T x, T y, T z) : x(x), y(y), z(z) {
    }
};
