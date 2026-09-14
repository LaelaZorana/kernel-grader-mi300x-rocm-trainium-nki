# hipcc on ROCm 6.x. MI300X is gfx942. for MI250X use ARCH=gfx90a.
HIPCC ?= hipcc
ARCH  ?= gfx942
CXXFLAGS ?= -O3 -std=c++17 --offload-arch=$(ARCH)

SRCS = harness.hip impl_naive.hip impl_beginner_fix.hip impl_agent.hip impl_golden.hip cheat_readback.hip cheat_async.hip cheat_badlaunch.hip
OBJS = $(SRCS:.hip=.o)

all: harness

harness: $(OBJS)
	$(HIPCC) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.hip common.h keygen.h
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) harness harness_unhardened.json harness_hardened.json ref_dump.bin

.PHONY: all clean
