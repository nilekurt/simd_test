#include <array>
#include <iostream>
#include <random>
#include <smmintrin.h>
#include <sstream>
#include <string>

using two_float __attribute__((vector_size(sizeof(float) * 2))) = float;
using four_float __attribute__((vector_size(sizeof(float) * 4))) = float;

auto
to_m128(two_float x) -> __m128
{
    union {
        __m128    whole;
        two_float parts[2];
    } result{};

    result.parts[0] = x;
    return result.whole;
}

auto
to_m128(__m128 v) -> __m128
{
    return v;
}

template<typename T>
auto
from_m128(__m128 x) -> T;

template<>
auto
from_m128(__m128 x) -> float
{
    union {
        __m128 whole;
        float  parts[4];
    } result{};

    result.whole = x;

    return result.parts[0];
}

template<>
auto
from_m128(__m128 x) -> two_float
{
    union {
        __m128    whole;
        two_float parts[2];
    } result{};

    result.whole = x;

    return result.parts[0];
}

template<typename T,
         int DimX,
         int DimY,
         typename = std::enable_if_t<std::is_arithmetic_v<T>>>
struct Mat {
    using vector_type __attribute__((vector_size(sizeof(T) * DimX))) = T;

    explicit Mat(const std::array<T, DimX * DimY> & values)
    {
        data_.values = values;
    }

    auto
    operator+(const Mat & other) -> Mat &
    {
        for (int i = 0; i < DimY; ++i) {
            data_.rows[i] += other.data_.rows[i];
        }

        return *this;
    }

    template<int DimZ>
    auto
    operator*(const Mat<T, DimX, DimZ> & other) -> Mat<T, DimZ, DimY>
    {
        union {
            vector_type                rows[DimY];
            std::array<T, DimZ * DimY> values;
        } result{};

        for (int i = 0; i < DimZ; ++i) {
            for (int j = 0; j < DimY; ++j) {
                if constexpr (std::is_floating_point_v<T> &&
                              (DimX == 2 || (DimX == 4))) {
                    constexpr int dot_product_mask = 0xF1;
                    result.values[i * DimZ + j] =
                        from_m128<T>(_mm_dp_ps(to_m128(other.getRows()[i]),
                                               to_m128(data_.rows[j]),
                                               dot_product_mask));
                }
            }
        }

        return Mat<T, DimZ, DimY>{result.values};
    }

    [[nodiscard]] auto
    getRows() const -> const vector_type (&)[DimY]
    {
        return data_.rows;
    }

    [[nodiscard]] auto
    getValues() const -> const std::array<T, DimX * DimY> &
    {
        return data_.values;
    }

    friend auto
    to_string(const Mat & x) -> std::string
    {
        std::stringstream ss{};

        ss << '{';

        for (int i = 0; i < DimY; ++i) {
            ss << '{';
            for (int j = 0; j < DimX; ++j) {
                ss << x.data_.values[i * DimX + j];
                if (j < DimX - 1) {
                    ss << ", ";
                }
            }
            ss << '}';
            if (i < DimY - 1) {
                ss << ",\n";
            }
        }
        ss << '}';

        return ss.str();
    }

private:
    union {
        vector_type                rows[DimY];
        std::array<T, DimX * DimY> values;
    } data_;
};

template<int M, int N>
auto
random_float_matrix(float lower_bound, float upper_bound) -> Mat<float, M, N>
{
    static std::mt19937            generator{};
    std::uniform_real_distribution dist{lower_bound, upper_bound};

    std::array<float, M * N> result{};

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            result[i * M + j] = dist(generator);
        }
    }
    return Mat<float, M, N>{result};
}

auto
main(int /*argc*/, char * * /*argv*/) -> int
{
    auto a = random_float_matrix<4, 1>(0.0f, 30.0f);
    std::cout << "a = " << to_string(a) << '\n';

    auto b = random_float_matrix<4, 1>(0.0f, 30.0f);
    std::cout << "b = " << to_string(b) << '\n';

    auto m = random_float_matrix<4, 4>(-10.0f, 10.0f);
    std::cout << "m =\n" << to_string(m) << '\n';

    auto c = a + b;
    std::cout << "c = a + b = " << to_string(c) << '\n';

    auto d = m * c;
    std::cout << "d = m * c =\n" << to_string(d) << '\n';

    return 0;
}