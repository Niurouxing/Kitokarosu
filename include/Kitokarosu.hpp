#pragma once

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <mdspan>
#include <random>
#include <type_traits>

inline namespace Kito {

// 初始化随机数生成器
inline static std::mt19937 gen(std::random_device{}());

// 均匀分布整数
template <int min, int max>
inline static int uniform_int_distribution()
{
    static std::uniform_int_distribution<int> dist(min, max);
    return dist(gen);
}

// 正态分布
template <auto mean, auto stddev>
inline static double normal_distribution()
{
    static std::normal_distribution<double> dist(mean, stddev);
    return dist(gen);
}

inline static constexpr double divSqrt2 = 0.7071067811865475;

// ------------------- tagExtractor -------------------

template <typename Tag, typename... Args>
struct tagExtractor;

// 值未能匹配，最终返回默认值
template <typename T, template <T> class Tag, T Value>
struct tagExtractor<Tag<Value>>
{
    static constexpr T value = Value;
};

// 值匹配成功
template <typename T, template <T> class Tag, T Value, T Value2,
          typename... Args>
struct tagExtractor<Tag<Value>, Tag<Value2>, Args...>
{
    static constexpr T value = Value2;
};

// 类型未能匹配，最终返回默认值
template <template <typename> class Tag, typename T>
struct tagExtractor<Tag<T>>
{
    using type = T;
};

// 类型未能匹配，最终返回默认值，多个参数版本
// 注意多个参数版本非常特殊地会将Tag<>保留
template <template <typename...> class Tag, typename... Args>
struct tagExtractor<Tag<Args...>>
{
    using type = Tag<Args...>;
};

// 类型匹配成功，单个参数版本
template <template <typename> class Tag, typename T, typename T2,
          typename... Args>
struct tagExtractor<Tag<T>, Tag<T2>, Args...>
{
    using type = T2;
};

// 类型匹配成功，多个参数版本
template <template <typename...> class Tag, typename... Args, typename... Args2,
          typename... Args3>
struct tagExtractor<Tag<Args...>, Tag<Args2...>, Args3...>
{
    using type = Tag<Args2...>;
};

// 匹配失败，类型不符，继续递归
template <typename Tag, typename Tag2, typename... Args>
struct tagExtractor<Tag, Tag2, Args...> : tagExtractor<Tag, Args...>
{};

// ------------------- Tensor -------------------

template <typename PrecInput, size_t... dimsInput>
class Tensor
{
public:
    std::array<PrecInput, (dimsInput * ...)> data;

    static inline constexpr size_t dims = sizeof...(dimsInput);
    static inline constexpr size_t size = (dimsInput * ...);
    static inline constexpr std::array<size_t, dims> shape = {dimsInput...};

    // mdspan of data
    std::mdspan<PrecInput, std::extents<std::size_t, dimsInput...>,
                std::layout_left>
        data_span{data.data()};

    // constructor
    constexpr Tensor() = default;
    constexpr Tensor(auto... vals) : data{static_cast<PrecInput>(vals)...} {}

    // if from a l-value std::array, copy the data
    template <size_t N, typename T>
    Tensor(const std::array<T, N> &arr)
    {
        static_assert(N == size, "Size of array must match the size of tensor");

        if constexpr (std::is_same_v<PrecInput, T>)
        {
            std::copy(arr.begin(), arr.end(), data.begin());
        }
        else
        {
            std::transform(arr.begin(), arr.end(), data.begin(),
                           [](auto x) { return static_cast<PrecInput>(x); });
        }
    }

    // if from a r-value std::array, move the data
    template <size_t N, typename T>
    Tensor(std::array<T, N> &&arr)
    {
        static_assert(N == size, "Size of array must match the size of tensor");

        if constexpr (std::is_same_v<PrecInput, T>)
        {
            std::move(arr.begin(), arr.end(), data.begin());
        }
        else
        {
            std::transform(arr.begin(), arr.end(), data.begin(),
                           [](auto x) { return static_cast<PrecInput>(x); });
        }
    }

    // [] operator
    auto &operator[](auto... indices) const
        requires(sizeof...(indices) == dims && sizeof...(indices) > 1)
    {
        return data_span[indices...];
    }

    auto &operator[](auto... indices)
        requires(sizeof...(indices) == dims && sizeof...(indices) > 1)
    {
        return data_span[indices...];
    }

    auto operator[](size_t index) const { return data[index]; }

    auto &operator[](size_t index) { return data[index]; }

    // auto convert to PrecInput*
    operator PrecInput *() { return data.data(); }

    // auto convert to const PrecInput*
    operator const PrecInput *() const { return data.data(); }

    // auto convert to std::array<PrecInput, size>
    operator std::array<PrecInput, size>() const { return data; }

    operator std::array<PrecInput, size> &() { return data; }

    // range-based for loop
    auto begin() { return data.begin(); }

    auto end() { return data.end(); }

    auto begin() const { return data.begin(); }

    auto end() const { return data.end(); }

    auto clear()
    {
        memset(data.data(), 0, size * sizeof(PrecInput));
        return *this;
    }

    friend std::ostream &operator<<(std::ostream &os, const Tensor &val)
    {
        // 保存原始格式状态
        std::ios_base::fmtflags original_flags = os.flags();
        std::streamsize original_precision = os.precision();

        // 初始化最大宽度和最长小数位数
        int max_width = 0, max_decimal = 0;

        // 临时字符串和流用于格式化
        std::string temp_str;
        std::vector<std::string> formatted_elements;
        formatted_elements.reserve(val.size);

        for (size_t i = 0; i < val.size; i++)
        {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(6) << val.data[i]; // 选择适当的精度
            temp_str = ss.str();

            // 调整为非科学计数法的最大精度显示
            size_t dot_pos = temp_str.find('.');
            if (dot_pos != std::string::npos)
            {
                int decimal_places = temp_str.length() - dot_pos - 1;
                max_decimal = std::max(max_decimal, decimal_places);
            }

            formatted_elements.push_back(temp_str);
            max_width = std::max(max_width, static_cast<int>(temp_str.length()));
        }

        // 设置输出格式
        os << std::fixed << std::setprecision(max_decimal);

        // 输出
        os << "[";
        if constexpr (Tensor ::dims == 2)
        {
            auto [row, col] = Tensor ::shape;
            for (size_t i = 0; i < row; i++)
            {
                if (i != 0)
                {
                    os << std::endl
                       << " ";
                }
                for (size_t j = 0; j < col; j++)
                {
                    os << std::setw(max_width) << std::right
                       << formatted_elements[i * col + j];
                    if (j != col - 1)
                    {
                        os << ", ";
                    }
                }
            }
        }
        else
        {
            for (size_t i = 0; i < val.size; i++)
            {
                os << std::setw(max_width) << std::right << formatted_elements[i];
                if (i != val.size - 1)
                {
                    os << ", ";
                }
            }
        }
        os << "]";

        // 恢复原始格式状态
        os.flags(original_flags);
        os.precision(original_precision);

        return os;
    }

    void print(std::string name = "")
    {
        if (name != "")
        {
            std::cout << name << " : " << std::endl;
        }

        std::cout << *this << std::endl;
    }
};

// ------------------- Detection -------------------

// specify using float or double
template <typename PrecType>
struct Prec;

// specify using real domain or complex domain
struct RD;
struct CD;

template <typename DomainType>
struct Dom
{
    using type = DomainType;
};

template <size_t N>
struct Rx
{
    static constexpr size_t value = N;
};

template <size_t N>
struct Tx
{
    static constexpr size_t value = N;
};

struct QAM16
{
    inline static constexpr size_t bitLength = 4;
    inline static constexpr std::array<double, 4> symbolsRD = {
        -0.31622776601683794, -0.9486832980505138, 0.31622776601683794,
        0.9486832980505138};
};

struct QAM64
{
    inline static constexpr size_t bitLength = 6;
    inline static constexpr std::array<double, 8> symbolsRD = {
        -0.4629100498862757, -0.1543033499620919, -0.7715167498104595,
        -1.0801234497346432, 0.1543033499620919, 0.4629100498862757,
        0.7715167498104595, 1.0801234497346432};
};

struct QAM256
{
    inline static constexpr size_t bitLength = 8;
    inline static constexpr std::array<double, 16> symbolsRD = {
        -0.3834824944236852, -0.5368754921931592, -0.2300894966542111,
        -0.07669649888473704, -0.8436614877321074, -0.6902684899626333,
        -0.9970544855015815, -1.1504474832710556, 0.3834824944236852,
        0.5368754921931592, 0.2300894966542111, 0.07669649888473704,
        0.8436614877321074, 0.6902684899626333, 0.9970544855015815,
        1.1504474832710556};
};

template <typename ModType>
struct Mod
{
    using type = ModType;
};

template <typename... Args>
class Detection_s;

template <typename PrecInput, size_t RxAntNumInput, size_t TxAntNumInput,
          typename ModTypeInput>
class Detection_s<Prec<PrecInput>, Dom<RD>, Rx<RxAntNumInput>,
                  Tx<TxAntNumInput>, Mod<ModTypeInput>>
{
public:
    inline static constexpr size_t RxAntNum = RxAntNumInput;
    inline static constexpr size_t TxAntNum = TxAntNumInput;
    using ModType = ModTypeInput;
    using PrecType = PrecInput;

    inline static constexpr std::array symbolsRD = ModType::symbolsRD;

    // std::array<size_t, 2 * TxAntNum> TxIndices;
    // std::array<PrecType, 2 * TxAntNum> TxSymbols;
    // std::array<PrecType, 2 * RxAntNum> RxSymbols;
    // std::array<PrecType, 2 * RxAntNum * 2 * TxAntNum> H;

    Tensor<size_t, 2 * TxAntNum> TxIndices;
    Tensor<PrecType, 2 * TxAntNum> TxSymbols;
    Tensor<PrecType, 2 * RxAntNum> RxSymbols;
    Tensor<PrecType, 2 * RxAntNum, 2 * TxAntNum> H;

    double Nv = 10;
    double sqrtNvDiv2 = std::sqrt(Nv / 2);

    inline static auto create()
    {
        return Detection_s<Prec<PrecInput>, Dom<RD>, Rx<RxAntNumInput>,
                           Tx<TxAntNumInput>, Mod<ModTypeInput>>();
    }

    void setSNR(double SNRdB)
    {
        Nv = TxAntNum * RxAntNum /
             (std::pow(10, SNRdB / 10) * ModType::bitLength * TxAntNum);
        sqrtNvDiv2 = std::sqrt(Nv / 2);
    }

    inline void generateTxIndices()
    {
        std::generate(TxIndices.begin(), TxIndices.end(), []() {
            return uniform_int_distribution<0, ModType::symbolsRD.size() - 1>();
        });
    }

    inline void generateTxSymbols()
    {
        std::transform(TxIndices.begin(), TxIndices.end(), TxSymbols.begin(),
                       [](size_t index) { return symbolsRD[index]; });
    }

    inline void generateH()
    {
        for (size_t j = 0; j < TxAntNum; j++)
        {
            for (size_t i = 0; i < RxAntNum; i++)
            {
                auto temp = normal_distribution<0, divSqrt2>();

                H[i, j] = temp;
                H[i + RxAntNum, j + TxAntNum] = temp;

                temp = normal_distribution<0, divSqrt2>();
                H[i, j + TxAntNum] = temp;
                H[i + RxAntNum, j] = -temp;
            }
        }
    }

    inline void generateRxSymbols()
    {
        RxSymbols.clear();
        for (size_t j = 0; j < 2 * TxAntNum; j++)
        {
            for (size_t i = 0; i < 2 * RxAntNum; i++)
            {
                RxSymbols[i] += H[i, j] * TxSymbols[j];
            }
        }

        std::transform(RxSymbols.begin(), RxSymbols.end(), RxSymbols.begin(),
                       [this](auto x) {
                           return x + normal_distribution<0, 1>() * sqrtNvDiv2;
                       });
    }

    inline void generate()
    {
        generateTxIndices();
        generateTxSymbols();
        generateH();
        generateRxSymbols();
    }

    template <typename T>
        requires std::is_same_v<T, float> || std::is_same_v<T, double>
    inline auto judge(std::array<T, 2 * TxAntNum> &symbolsEst)
    {

        static std::array<size_t, 2 * TxAntNum> wrongBits;

        std::transform(
            symbolsEst.begin(), symbolsEst.end(), TxIndices.begin(),
            wrongBits.begin(), [](T symbol, size_t index) {
                auto closest = std::min_element(symbolsRD.begin(), symbolsRD.end(),
                                                [symbol](auto x, auto y) {
                                                    return std::abs(x - symbol) <
                                                           std::abs(y - symbol);
                                                }) -
                               symbolsRD.begin();
                return std::bitset<ModType::bitLength>(closest ^ index).count();
            });

        return std::accumulate(wrongBits.begin(), wrongBits.end(), 0);
    }

    inline auto judge(std::array<int, 2 * TxAntNum> &bitsEst)
    {
        static std::array<size_t, 2 * TxAntNum> wrongBits;

        std::transform(bitsEst.begin(), bitsEst.end(), ModType::bitsRD.begin(),
                       wrongBits.begin(),
                       [](int bit, int bitRD) { return bit ^ bitRD; });

        return std::accumulate(wrongBits.begin(), wrongBits.end(), 0);
    }
};

template <typename... Args>
struct DetectionInputHelper
{
    inline static constexpr auto RxAntNum = tagExtractor<Rx<4>, Args...>::value;
    inline static constexpr auto TxAntNum = tagExtractor<Tx<4>, Args...>::value;
    using ModType = tagExtractor<Mod<QAM16>, Args...>::type;
    using DomainType = tagExtractor<Dom<RD>, Args...>::type;
    using PrecType = tagExtractor<Prec<float>, Args...>::type;

    static_assert(RxAntNum > 0, "RxAntNum must be greater than 0");
    static_assert(TxAntNum > 0, "TxAntNum must be greater than 0");
    static_assert(RxAntNum >= TxAntNum,
                  "RxAntNum must be greater than or equal to TxAntNum");

    static_assert(std::is_same<ModType, QAM16>::value ||
                      std::is_same<ModType, QAM64>::value ||
                      std::is_same<ModType, QAM256>::value,
                  "ModType must be QAM16, QAM64 or QAM256");
    static_assert(std::is_same<DomainType, RD>::value ||
                      std::is_same<DomainType, CD>::value,
                  "DomainType must be RD or CD");
    static_assert(std::is_same<PrecType, float>::value ||
                      std::is_same<PrecType, double>::value,
                  "PrecType must be float or double");

    using type = Detection_s<Prec<PrecType>, Dom<DomainType>, Rx<RxAntNum>,
                             Tx<TxAntNum>, Mod<ModType>>;
};

template <typename... Args>
using Detection = typename DetectionInputHelper<Args...>::type;

} // namespace Kito
