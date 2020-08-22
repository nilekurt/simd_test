#include <array>
#include <iostream>
#include <smmintrin.h>
#include <sstream>
#include <string>

template<typename T,
         std::size_t DimX,
         std::size_t DimY,
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
        for (int i = 0; i < static_cast<int>(DimY); ++i) {
            data_.rows[i] += other.data_.rows[i];
        }

        return *this;
    }

    template<std::size_t DimZ>
    auto
    operator*(const Mat<T, DimX, DimZ> & other) -> Mat<T, DimZ, DimY>
    {
        constexpr int Y = DimY;
        constexpr int Z = DimZ;

        union {
            vector_type                rows[DimY];
            std::array<T, DimZ * DimY> values;
        } result;
        for (int i = 0; i < Z; ++i) {
            for (int j = 0; j < Y; ++j) {
                if constexpr (std::is_floating_point_v<T>) {
                    result.rows[i * Z + j] =
                        _mm_dp_ps((__m128)other.getRows()[i],
                                  (__m128)data_.rows[j],
                                  0xFF);
                }
            }
        }

        return Mat<T, DimZ, DimY>{result.values};
    }

    [[nodiscard]] auto
    getRows() const -> const vector_type *
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

        for (int i = 0; i < static_cast<int>(DimX * DimY); ++i) {
            ss << x.data_.values[i];
            if (i < static_cast<int>(DimX * DimY - 1)) {
                ss << ", ";
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

auto
main(int argc, char ** argv) -> int
{
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <x1> <x2> <y1> <y2>\n";
        return -1;
    }

    float x1, x2, y1, y2;

    x1 = strtof(argv[1], nullptr);
    x2 = strtof(argv[2], nullptr);
    y1 = strtof(argv[3], nullptr);
    y2 = strtof(argv[4], nullptr);

    using Vec2f = Mat<float, 4, 1>;
    Vec2f a{{x1, x1, x2, x2}};
    Vec2f b{{y1, y1, y2, y2}};

    auto c = a + b;

    std::cout << to_string(c) << '\n';

    using Mat4f = Mat<float, 4, 4>;
    Mat4f m{{x1, x2, y1, y2, x1, x2, y1, y2}};

    auto d = m * c;

    std::cout << to_string(d) << '\n';

    return 0;
}