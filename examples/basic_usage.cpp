#include "Kitokarosu.hpp"

#include <print>

using namespace Kito;
using namespace Kito::literals;

int main() {
  seed_random(0xC0FFEE);

  // Defaults: MMSE -> Studer soft output -> 10-iteration layered min-sum.
  auto curve = (m<4_Tx, 4_Rx> | c<256, 0.5> | mmse)(
      0_dB, 10_dB, 2_dB, 10_e, 1'000_f);

  std::println(" SNR(dB) | frames | frame errors |       BER |       FER");
  std::println("---------+--------+--------------+-----------+----------");
  for (const auto &point : curve) {
    std::println("{:8.1f} | {:6} | {:12} | {:9.3e} | {:9.3e}",
                 point.snr_db, point.frames, point.frame_errors, point.ber,
                 point.fer);
  }
}
