# Paths
PY_DIR = ./python
SRC_DIR = ./src
INC_DIR = ./include
LOG_FILE = workflow.log

# Tools & Flags
PYTHON = python
NVCC = nvcc
NVCC_FLAGS = -O3 -Xcompiler -fPIC -shared -I$(INC_DIR)
LIBS = -lcublas -lcusolver

.PHONY: all build_cuda workflow_baseline workflow_standard workflow_full clean

# Main Entry Point
all: workflow_baseline workflow_standard build_cuda workflow_full

# --- Compilation (Requirement for Workflow 3) ---
build_cuda:
	@echo "$$(date): Compiling CUDA solver..." | tee -a $(LOG_FILE)
	$(NVCC) $(NVCC_FLAGS) $(SRC_DIR)/solver.cu $(SRC_DIR)/refinement.cu -o libmpsolver.so $(LIBS) >> $(LOG_FILE) 2>&1

# --- Workflow 1: Baselines ---
workflow_baseline:
	@echo "$$(date): Starting Baselines..." | tee -a $(LOG_FILE)
	$(PYTHON) $(PY_DIR)/baseline.py --size 128 --matrix random >> $(LOG_FILE) 2>&1
	$(PYTHON) $(PY_DIR)/baseline.py --size 128 --matrix hilbert >> $(LOG_FILE) 2>&1

# --- Workflow 2: Standard (Sequential) ---
workflow_standard: 
	@echo "$$(date): Starting Standard Workflow..." | tee -a $(LOG_FILE)
	$(PYTHON) $(PY_DIR)/run_experiments.py >> $(LOG_FILE) 2>&1
	$(PYTHON) $(PY_DIR)/aggregate_results.py >> $(LOG_FILE) 2>&1
	$(PYTHON) $(PY_DIR)/plot_results.py >> $(LOG_FILE) 2>&1

# --- Workflow 3: Full (Runs regardless of other workflow results) ---
# Note: build_cuda must succeed for these to work correctly
workflow_full: build_cuda
	@echo "$$(date): Starting Full Workflow..." | tee -a $(LOG_FILE)
	-$(PYTHON) $(PY_DIR)/run_full_experiments.py >> $(LOG_FILE) 2>&1
	-$(PYTHON) $(PY_DIR)/plot_full.py >> $(LOG_FILE) 2>&1

clean:
	rm -f $(LOG_FILE) libmpsolver.so
