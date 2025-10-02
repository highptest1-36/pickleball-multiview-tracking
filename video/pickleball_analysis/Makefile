# Makefile for Pickleball Video Analysis Pipeline
# Sử dụng: make [command]

# Variables
PYTHON := python
PIP := pip
VENV := venv
VENV_ACTIVATE := $(VENV)/Scripts/activate  # Windows
# VENV_ACTIVATE := $(VENV)/bin/activate    # Linux/Mac

# Default target
.PHONY: help
help:
	@echo "🏓 Pickleball Video Analysis Pipeline"
	@echo "======================================"
	@echo ""
	@echo "Available commands:"
	@echo "  setup          - Complete setup (install dependencies, create dirs)"
	@echo "  install        - Install Python dependencies only"
	@echo "  clean          - Clean temporary files and outputs"
	@echo "  test           - Run test suite"
	@echo "  demo           - Run demo pipeline"
	@echo "  calibrate      - Launch court calibration"
	@echo "  run            - Run full pipeline with default settings"
	@echo "  lint           - Run code linting"
	@echo "  format         - Format code"
	@echo "  docs           - Generate documentation"
	@echo "  venv           - Create virtual environment"
	@echo "  requirements   - Update requirements.txt"
	@echo ""
	@echo "Examples:"
	@echo "  make setup                    # First-time setup"
	@echo "  make run INPUT=data_video/    # Run with specific input"
	@echo "  make demo                     # Quick demo test"

# Setup and installation
.PHONY: setup
setup: venv install dirs config weights
	@echo "✅ Setup completed!"
	@echo "💡 Next: make calibrate (if needed) then make run"

.PHONY: venv
venv:
	@echo "🐍 Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "✅ Virtual environment created"
	@echo "💡 Activate with: $(VENV_ACTIVATE)"

.PHONY: install
install:
	@echo "📦 Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

.PHONY: dirs
dirs:
	@echo "📁 Creating directories..."
	@mkdir -p output/tracking_data
	@mkdir -p output/charts
	@mkdir -p output/reports
	@mkdir -p output/processed_videos
	@mkdir -p logs
	@mkdir -p data_video
	@echo "✅ Directories created"

.PHONY: config
config:
	@echo "⚙️ Setting up configuration..."
	@if [ ! -f config/config.yaml ]; then \
		mkdir -p config; \
		cp config/config.yaml.example config/config.yaml 2>/dev/null || \
		echo "Copy config/config.yaml.example to config/config.yaml manually"; \
	fi
	@echo "✅ Configuration ready"

.PHONY: weights
weights:
	@echo "🤖 Downloading YOLO weights..."
	@$(PYTHON) -c "from ultralytics import YOLO; YOLO('yolov8x.pt')" 2>/dev/null || \
		echo "⚠️ YOLO weights will download on first run"

# Running pipeline
.PHONY: run
run:
	@echo "🚀 Running full pipeline..."
	$(PYTHON) main.py $(if $(INPUT),--input $(INPUT),) $(if $(OUTPUT),--output $(OUTPUT),) $(ARGS)

.PHONY: detection
detection:
	@echo "🎯 Running detection only..."
	$(PYTHON) main.py --mode detection $(if $(INPUT),--input $(INPUT),) $(ARGS)

.PHONY: tracking  
tracking:
	@echo "📍 Running tracking only..."
	$(PYTHON) main.py --mode tracking $(if $(INPUT),--input $(INPUT),) $(ARGS)

.PHONY: analysis
analysis:
	@echo "📊 Running analysis only..."
	$(PYTHON) main.py --mode analysis $(if $(INPUT),--input $(INPUT),) $(ARGS)

.PHONY: calibrate
calibrate:
	@echo "🎯 Launching court calibration..."
	$(PYTHON) src/court_detection.py --calibrate

.PHONY: validate-calibration
validate-calibration:
	@echo "✅ Validating court calibration..."
	$(PYTHON) src/court_detection.py --validate

# Testing and demo
.PHONY: test
test:
	@echo "🧪 Running tests..."
	$(PYTHON) demo.py --test-only config
	$(PYTHON) demo.py --test-only analysis
	@echo "✅ Tests completed"

.PHONY: demo
demo:
	@echo "🎬 Running demo..."
	$(PYTHON) demo.py

.PHONY: quick-demo
quick-demo:
	@echo "⚡ Running quick demo..."
	$(PYTHON) demo.py --quick

# Code quality
.PHONY: lint
lint:
	@echo "🔍 Running linter..."
	@$(PIP) install flake8 pylint 2>/dev/null || echo "Install flake8/pylint for linting"
	@flake8 src/ main.py demo.py --max-line-length=100 2>/dev/null || echo "Flake8 not available"
	@pylint src/ main.py demo.py 2>/dev/null || echo "Pylint not available"

.PHONY: format
format:
	@echo "🎨 Formatting code..."
	@$(PIP) install black isort 2>/dev/null || echo "Install black/isort for formatting"
	@black src/ main.py demo.py --line-length=100 2>/dev/null || echo "Black not available"
	@isort src/ main.py demo.py 2>/dev/null || echo "Isort not available"

# Documentation
.PHONY: docs
docs:
	@echo "📚 Generating documentation..."
	@mkdir -p docs/generated
	@$(PYTHON) -c "import pydoc; pydoc.writedoc('src')" 2>/dev/null || echo "Pydoc generation failed"
	@echo "📖 Check docs/ directory for documentation"

# Utilities
.PHONY: clean
clean:
	@echo "🧹 Cleaning up..."
	@rm -rf output/* 2>/dev/null || rmdir /s /q output 2>/dev/null || echo "Output cleaned"
	@rm -rf logs/* 2>/dev/null || rmdir /s /q logs 2>/dev/null || echo "Logs cleaned"
	@rm -rf demo_output/ 2>/dev/null || rmdir /s /q demo_output 2>/dev/null || echo "Demo output cleaned"
	@find . -name "*.pyc" -delete 2>/dev/null || echo "Python cache cleaned"
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || echo "Pycache cleaned"
	@echo "✅ Cleanup completed"

.PHONY: clean-all
clean-all: clean
	@echo "🧹 Deep cleaning..."
	@rm -rf $(VENV) 2>/dev/null || rmdir /s /q $(VENV) 2>/dev/null || echo "Venv removed"
	@rm -f *.pt 2>/dev/null || del *.pt 2>/dev/null || echo "Model weights cleaned"

.PHONY: requirements
requirements:
	@echo "📋 Updating requirements.txt..."
	$(PIP) freeze > requirements_new.txt
	@echo "✅ New requirements saved to requirements_new.txt"
	@echo "💡 Review and replace requirements.txt if needed"

.PHONY: info
info:
	@echo "ℹ️ System Information"
	@echo "===================="
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Working directory: $(shell pwd)"
	@echo "Virtual env: $(VENV)"
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch: Not installed"

# Batch operations
.PHONY: process-all
process-all:
	@echo "🔄 Processing all videos in data_video/..."
	@for video in data_video/*.mp4; do \
		if [ -f "$$video" ]; then \
			echo "Processing $$video..."; \
			$(PYTHON) main.py --input "$$video" --output "output/$$(basename $$video .mp4)"; \
		fi; \
	done

.PHONY: backup
backup:
	@echo "💾 Creating backup..."
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	tar -czf "backup_$$timestamp.tar.gz" \
		--exclude=venv --exclude=output --exclude=logs --exclude="*.pt" \
		. 2>/dev/null || \
	7z a "backup_$$timestamp.7z" . -xr!venv -xr!output -xr!logs -xr!"*.pt" 2>/dev/null || \
	echo "Create backup manually: tar/7z not available"

# Development helpers
.PHONY: dev-setup
dev-setup: setup
	@echo "🛠️ Setting up development environment..."
	$(PIP) install black flake8 pylint isort pytest notebook jupyter
	@echo "✅ Development tools installed"

.PHONY: notebook
notebook:
	@echo "📓 Starting Jupyter notebook..."
	jupyter notebook --notebook-dir=. --port=8888

.PHONY: profile
profile:
	@echo "⏱️ Profiling pipeline performance..."
	$(PYTHON) -m cProfile -o profile.stats main.py --max-frames 100
	@echo "📊 Profile saved to profile.stats"

# Quick commands for common workflows
.PHONY: first-run
first-run: setup calibrate demo
	@echo "🎉 First-run completed!"
	@echo "💡 Now you can: make run INPUT=your_video.mp4"

.PHONY: daily-workflow
daily-workflow:
	@echo "📅 Daily workflow: clean, update, run..."
	make clean
	$(PIP) install --upgrade -r requirements.txt
	make run

# GPU specific commands
.PHONY: gpu-test
gpu-test:
	@echo "🎮 Testing GPU availability..."
	@$(PYTHON) -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null || echo "PyTorch not available"

.PHONY: cpu-mode
cpu-mode:
	@echo "💻 Switching to CPU mode..."
	@sed -i 's/device: "cuda"/device: "cpu"/' config/config.yaml 2>/dev/null || \
	echo "Manually change device: 'cpu' in config/config.yaml"

.PHONY: gpu-mode  
gpu-mode:
	@echo "🎮 Switching to GPU mode..."
	@sed -i 's/device: "cpu"/device: "cuda"/' config/config.yaml 2>/dev/null || \
	echo "Manually change device: 'cuda' in config/config.yaml"

# Performance monitoring
.PHONY: monitor
monitor:
	@echo "📊 Monitoring system resources during pipeline..."
	@echo "Press Ctrl+C to stop monitoring"
	@while true; do \
		echo "$$(date): CPU: $$(top -bn1 | grep 'Cpu(s)' | awk '{print $$2}' | cut -d'%' -f1)% Memory: $$(free | grep Mem | awk '{printf "%.1f%%", $$3/$$2 * 100.0}')"; \
		sleep 5; \
	done 2>/dev/null || echo "Monitoring not available on this system"

# Help for specific topics
.PHONY: help-setup
help-setup:
	@echo "🔧 Setup Help"
	@echo "============"
	@echo "1. make venv          - Create virtual environment"
	@echo "2. source venv/bin/activate  - Activate venv (Linux/Mac)"
	@echo "3. venv\Scripts\activate     - Activate venv (Windows)"
	@echo "4. make install       - Install dependencies"
	@echo "5. make dirs          - Create directories"
	@echo "6. make calibrate     - Calibrate court"

.PHONY: help-run
help-run:
	@echo "🚀 Running Help"
	@echo "=============="
	@echo "Basic:"
	@echo "  make run                           - Full pipeline"
	@echo "  make run INPUT=video.mp4           - Process single video"
	@echo "  make run INPUT=dir/ OUTPUT=out/    - Custom input/output"
	@echo ""
	@echo "Modes:"
	@echo "  make detection INPUT=video.mp4     - Detection only"
	@echo "  make tracking INPUT=video.mp4      - Tracking only"
	@echo "  make analysis INPUT=data.csv       - Analysis only"
	@echo ""
	@echo "Options:"
	@echo "  make run ARGS='--max-frames 100'   - Limit frames"
	@echo "  make run ARGS='--verbose'          - Verbose output"

# Version and status
.PHONY: version
version:
	@echo "📋 Pickleball Analysis Pipeline v1.0.0"
	@echo "Status: $(shell [ -f config/config.yaml ] && echo 'Configured' || echo 'Not configured')"
	@echo "Calibration: $(shell $(PYTHON) -c 'from src.utils import load_court_points; cp = load_court_points(); print("OK" if any(cp["cameras"][c]["calibration_status"] == "calibrated" for c in cp["cameras"]) else "Needed")' 2>/dev/null || echo 'Unknown')"

# Force targets (don't check file timestamps)
.PHONY: *