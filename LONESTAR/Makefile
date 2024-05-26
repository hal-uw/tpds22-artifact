TOPLEVEL := .
IRAPPS := bfs mst sssp sgd dmr mis cc pr triangle
APPS := bh pta
INPUT_URL := https://pages.cs.wisc.edu/~sinclair/tpds22/USA-road-d.W.gr
# Currently Texas website does not accept wget requests, so use a local copy
# http://iss.ices.utexas.edu/projects/galois/downloads/lonestargpu2-inputs.tar.bz2
BIP_INPUT_URL := http://iss.ices.utexas.edu/projects/galois/downloads/lonestargpu21-bipartite-inputs.tar.xz
INPUT := USA-road-d.W.gr
#lonestargpu2-inputs.tar.bz2
BIP_INPUT := lonestargpu21-bipartite-inputs.tar.xz

.PHONY: rt all clean inputs

all: rt $(APPS)
	for IRAPPS in $(IRAPPS); do make -C apps/$$IRAPPS; done

rt: 
	make -C rt/src

$(APPS):
	make -C apps/$@

$(IRAPPS):
	make -C apps/$@
include arch.mk
include common.mk

inputs:
	@echo "Downloading inputs ..."
	@mkdir -p inputs
	@wget $(INPUT_URL) -O inputs/$(INPUT)
#	@wget $(BIP_INPUT_URL) -O $(BIP_INPUT)
#	@echo "Uncompressing inputs ..."
#	@tar xvf $(INPUT)
#	@tar xvf $(BIP_INPUT)
#	@rm $(INPUT) $(BIP_INPUT)
	@echo "Inputs available at $(TOPLEVEL)/inputs/"

clean:
	make -C rt/src clean
	for APP in $(APPS); do make -C apps/$$APP clean; done 
	for IRAPPS in $(IRAPPS); do make -C apps/$$IRAPPS clean; done

