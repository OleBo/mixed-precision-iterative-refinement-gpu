# Paths
PY_DIR = ./python
SRC_DIR = ./src
BUILD_DIR = ./build
TEST_DIR = ./tests/python
LIB_NAME = libmixed_precision_lib.so
LOG_FILE = workflow.log

# Tools
PYTHON = python
PYTEST = pytest

.PHONY: all workflow_baseline workflow_standard workflow_full clean

all: workflow_baseline workflow_standard workflow_full

# --- Workflow 3: Full (Triggers CMake build) ---
# This target maps source changes to the specific build artifact
$(BUILD_DIR)/$(LIB_NAME): $(SRC_DIR)/solver.cu $(SRC_DIR)/refinement.cu
	@echo "$$(date): Rebuilding CUDA library..." | tee -a $(LOG_FILE)
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. >> ../$(LOG_FILE) 2>&1
	cmake --build $(BUILD_DIR) >> $(LOG_FILE) 2>&1
	
	@echo "$$(date): Running C++ tests..." | tee -a $(LOG_FILE)
	cd $(BUILD_DIR) && ctest --output-on-failure -j $$(nproc) >> ../$(LOG_FILE) 2>&1

	@echo "--- Build Verification ---"
	@test -f $@ || (echo "Error: $(LIB_NAME) not found in $(BUILD_DIR)" && exit 1)
	nm -D $@ | grep -E "gpuSolve|refineSolution"
	ldd -r $@ >> $(LOG_FILE) 2>&1

	@echo "$$(date): Running Python binding tests..." | tee -a $(LOG_FILE)
	$(PYTEST) $(TEST_DIR)/test_binding.py -v >> $(LOG_FILE) 2>&1

workflow_full: $(BUILD_DIR)/$(LIB_NAME)
	@echo "$$(date): Starting Full Workflow..." | tee -a $(LOG_FILE)
	-$(PYTHON) $(PY_DIR)/run_full_experiments.py >> $(LOG_FILE) 2>&1
	-$(PYTHON) $(PY_DIR)/plot_full.py >> $(LOG_FILE) 2>&1

# --- Workflow 1: Baselines ---
workflow_baseline:
	@echo "$$(date): Starting Baselines..." | tee -a $(LOG_FILE)
	$(PYTHON) $(PY_DIR)/baseline.py --size 128 --matrix random >> $(LOG_FILE) 2>&1
	$(PYTHON) $(PY_DIR)/baseline.py --size 128 --matrix hilbert >> $(LOG_FILE) 2>&1

# --- Workflow 2: Standard ---
workflow_standard: 
	@echo "$$(date): Starting Standard Workflow..." | tee -a $(LOG_FILE)
	$(PYTHON) $(PY_DIR)/run_experiments.py >> $(LOG_FILE) 2>&1
	$(PYTHON) $(PY_DIR)/aggregate_results.py >> $(LOG_FILE) 2>&1
	$(PYTHON) $(PY_DIR)/plot_results.py >> $(LOG_FILE) 2>&1

clean:
	@echo "Cleaning build artifacts and logs..."
	rm -f $(LOG_FILE)
	rm -rf $(BUILD_DIR)
