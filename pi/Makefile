OBJDIR = ../lib
BUILDDIR = ../bin

OUTPUT = pi.elf

rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

CUSRC = $(call rwildcard,./,*.cu)
OBJS = $(patsubst %.cu, $(OBJDIR)/%_$(OUTPUT).o, $(CUSRC))


CFLAGS = -g
LDFLAGS = 

CC = nvcc
LD = nvcc

$(OUTPUT): $(OBJS)
	@mkdir -p $(BUILDDIR)
	@echo LD $^
	@$(LD) $(LDFLAGS) -o $(BUILDDIR)/$@ $^

run: $(OUTPUT)
	@echo NVPROF ./$(BUILDDIR)/$(OUTPUT)
	@nvprof ./$(BUILDDIR)/$(OUTPUT)

debug: $(OUTPUT)
	@echo GDB ./$(BUILDDIR)/$(OUTPUT)
	@cuda-gdb ./$(BUILDDIR)/$(OUTPUT) -ex "break main" -ex "run" -ex "tui enable"

clean:
	@rm -f $(BUILDDIR)/$(OUTPUT) $(OBJS)

$(OBJDIR)/%_$(OUTPUT).o: %.cu
	@echo "NVCC $^ -> $@"
	@mkdir -p $(@D)
	@$(CC) $(CFLAGS) -c -o $@ $^

.PHONY: build