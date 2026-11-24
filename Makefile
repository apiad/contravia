PYTHON ?= python3
CORE ?= python
SIZE ?= 1920x1080
PRESET ?= alto
SCENE ?= controlled-parked-free
SEED ?= 7
REPEATS ?= 3
FRAMES ?= 120
TIMESTEP ?= fixed-physics

.PHONY: perf-reference
perf-reference:
	@test "$(CORE)" != rust || (echo 'CORE=rust is unavailable: no integrated native core; refusing a Python fallback.' >&2; exit 2)
	@test -n "$(DATA)" -a -n "$(OUT)" -a -n "$(ENGINE_PATH)" -a -n "$(ENGINE_REVISION)" || \
	  (echo 'Set DATA, OUT, ENGINE_PATH (manifoldx package directory), ENGINE_REVISION (full SHA), and PYTHON (existing environment).' >&2; exit 2)
	$(PYTHON) bench/perf_reference.py --data "$(DATA)" --output "$(OUT)" --core "$(CORE)" \
	  --engine-path "$(ENGINE_PATH)" --engine-revision "$(ENGINE_REVISION)" --size "$(SIZE)" \
	  --preset "$(PRESET)" --scene "$(SCENE)" --seed "$(SEED)" --repeats "$(REPEATS)" \
	  --frames "$(FRAMES)" --timestep "$(TIMESTEP)" $(PERF_ARGS)
