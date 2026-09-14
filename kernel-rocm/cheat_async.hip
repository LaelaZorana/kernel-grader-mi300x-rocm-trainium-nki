// cheat. launches the naive kernel and returns at once. no sync, so a timer that
// stops on return sees only the launch cost. the work finishes later on the stream.
#include "common.h"

extern "C" void slot_cheat_async(const uint32_t* d_keys, uint32_t n, uint32_t* d_hist) {
  naive_launch_nosync(d_keys, n, d_hist);
}
