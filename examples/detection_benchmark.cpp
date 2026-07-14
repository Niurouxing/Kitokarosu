#include "Kitokarosu.hpp"

#include <cstdlib>
#include <print>
#include <string_view>

using namespace Kito;
using namespace Kito::literals;

namespace {

inline constexpr auto tx_shape = 8_Tx;
inline constexpr auto rx_shape = 8_Rx;
using modulation = q16;
using system_type = mimo<tx_shape, rx_shape, modulation>;
using code_type = uncoded<system_type::model_type::bits_per_frame>;

template <detector_policy Detector>
void benchmark(std::string_view name, Detector detector, int first_snr,
               int last_snr, std::size_t errors, std::size_t frames,
               std::uint64_t seed) {
  seed_random(seed);

  auto curve = (system_type{} | code_type{} | detector)(
      literals::snr_db{static_cast<double>(first_snr)},
      literals::snr_db{static_cast<double>(last_snr)}, 1_dB,
      literals::error_limit{errors}, literals::frame_limit{frames});

  std::println("\n{}", name);
  for (const auto &point : curve) {
    std::println("  {:5.1f} dB  N={:6}  BER={:10.3e}  FER={:10.3e}",
                 point.snr_db, point.frames, point.ber, point.fer);
  }
}

} // namespace

int main(int argc, char **argv) {
  const auto max_frames =
      argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10))
               : 10'000;
  const auto error_frames =
      argc > 2 ? static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10))
               : 100;
  const int first_snr = argc > 3 ? std::atoi(argv[3]) : 9;
  const int last_snr = argc > 4 ? std::atoi(argv[4]) : 18;
  const auto seed = argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 114514ULL;

  std::println("C++26 detector benchmark: {}x{} {}-QAM", tx_shape.value,
               rx_shape.value, 1U << modulation::bits_per_complex_symbol);

  benchmark("MMSE", mmse, first_snr, last_snr, error_frames, max_frames,
            seed);
  benchmark("K-Best<16>", kbest<16>, first_snr, last_snr, error_frames,
            max_frames, seed);
  benchmark("EP<100, 0.7>", ep<100, 0.7F>, first_snr, last_snr,
            error_frames, max_frames, seed);
}
