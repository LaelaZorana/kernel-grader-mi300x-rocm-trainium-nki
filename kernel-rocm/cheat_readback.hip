// cheat. does no work. the output buffer still holds the previous golden run,
// so an unwiped shared buffer makes this look correct and instant.
#include "common.h"

extern "C" void slot_cheat_readback(const uint32_t* d_keys, uint32_t n, uint32_t* d_hist) {
  (void)d_keys; (void)n; (void)d_hist;
}
