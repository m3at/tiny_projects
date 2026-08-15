BUILD ?= build
PRESET ?= Debug

.PHONY: help configure build run spectate soak test check quick bench golden reference netcheck audio-check playthrough melee ablate rendercheck capture capture-one format-check sanitizer clean

help:  ## Show commands
	@grep -E '^[a-zA-Z0-9_/%.-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'

configure:  ## Configure the selected CMake build
	cmake -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=$(PRESET) -DBROADSIDE_DESKTOP=$${DESKTOP:-ON} -DBROADSIDE_ENET=$${ENET:-OFF}

build: configure  ## Build the desktop client and diagnostics
	cmake --build $(BUILD) --parallel

run: build  ## Run the desktop client; pass arguments with ARGS=
	$(BUILD)/bin/broadside $(ARGS)

spectate: build  ## Watch one complete bots-only match; override PLAYERS/SPEED/ARGS
	$(BUILD)/bin/broadside --players $${PLAYERS:-4} --bots-only --speed $${SPEED:-2} $(ARGS)

soak: build  ## Continuously run bots-only visual matches
	$(BUILD)/bin/broadside --players $${PLAYERS:-4} --bots-only --speed $${SPEED:-2} --loop $(ARGS)

test: build  ## Run the CTest suite
	ctest --test-dir $(BUILD) --output-on-failure

check: test  ## Run tests plus the invariant diagnostics
	$(BUILD)/bin/broadside_tool check

quick:  ## Configure and run the dependency-free focused suite
	cmake --preset headless
	cmake --build --preset headless --parallel
	ctest --test-dir build/headless -L quick --output-on-failure

bench: build  ## Benchmark deterministic battle simulation
	$(BUILD)/bin/broadside_tool bench $(ARGS)

golden: build  ## Print the native repeatability fixture
	$(BUILD)/bin/broadside_tool golden

reference: build  ## Parse and validate the 900-row semantic fixture
	$(BUILD)/bin/broadside_tool reference

netcheck: build  ## Exercise authority and replay messaging
	$(BUILD)/bin/broadside_tool netcheck

audio-check: build  ## Check procedural audio quality bounds
	$(BUILD)/bin/broadside_tool audio-check

playthrough: build  ## Drive a complete two-human local match
	$(BUILD)/bin/broadside_tool playthrough

melee: build  ## Measure three/four-ship decisiveness and seat fairness
	$(BUILD)/bin/broadside_tool melee

ablate: build  ## Compare canonical melee targeting with rule variants
	$(BUILD)/bin/broadside_tool ablate

rendercheck: build  ## Validate responsive presentation bounds
	$(BUILD)/bin/broadside_tool rendercheck

capture-one: build  ## Capture one hidden-window frame; override CAPTURE=/SCENARIO=
	$(BUILD)/bin/broadside --scenario $${SCENARIO:-duel} --capture $${CAPTURE:-broadside-capture.png} $(ARGS)

capture: build  ## Capture all six acceptance scenes at both reference resolutions
	@mkdir -p $${CAPTURE_DIR:-build/captures}
	@if [ "$${VERIFY_BASELINES:-1}" = "1" ]; then baseline="--baseline-file native/reference/render_baselines.txt"; else baseline=""; fi; \
	for size in "1280 720" "960 540"; do set -- $$size; \
	  for scene in menu shipyard duel four-way result match-end; do \
	    $(BUILD)/bin/broadside --seed 0xb0ad51de --scenario $$scene --width $$1 --height $$2 $$baseline \
	      --capture $${CAPTURE_DIR:-build/captures}/$$scene-$$1x$$2.png || exit $$?; \
	  done; \
	done

format-check:  ## Verify native C++ formatting without modifying files
	@$${CLANG_FORMAT:-$$(xcrun --find clang-format 2>/dev/null || command -v clang-format)} \
	  --dry-run --Werror $$(find native -type f \( -name '*.cpp' -o -name '*.h' \))

sanitizer:  ## Build and test the ASan/UBSan headless preset
	cmake --preset asan
	cmake --build --preset asan --parallel
	ctest --test-dir build/asan --output-on-failure

clean:  ## Remove compiled objects from the selected build
	cmake --build $(BUILD) --target clean 2>/dev/null || true
