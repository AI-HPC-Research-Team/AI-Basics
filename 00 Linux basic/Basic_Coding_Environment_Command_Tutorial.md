# Basic Coding Environment — Command Tutorial (EN)

This tutorial distills the in-class command demos for **Linux Basics**, **Git Basics**.
---

## 0) Quick Setup & Safety

- Run on Linux/macOS, a dev container, Codespaces, or a Colab/Jupyter `%%bash` cell.
- Commands that change users/groups or install software may require `sudo`.
- Use a sandbox directory so you don’t touch real data:
```bash
WORKDIR="$PWD/env_demo_sandbox"; rm -rf "$WORKDIR"; mkdir -p "$WORKDIR"; cd "$WORKDIR"
```

---

## 1) Linux Basics

### 1.1 Inspect the system & environment
```bash
echo "Shell: $SHELL"
whoami
echo "Home: $HOME"
echo "PWD : $PWD"
date
# A few key env vars:
for v in EDITOR HOME LOGNAME MAIL OLD PWD PATH SHELL PS1 PWD USER; do
  printf "%-8s -> %s
" "$v" "${!v-<unset>}"
done
```

### 1.2 Navigate files & directories
```bash
# Tree creation and navigation
mkdir -p test/subdir1/subdir2
echo "Hello everyone" > test/myfile.txt
echo "Goodbye all"   >> test/myfile.txt
ls -lah test
pwd                            # absolute path
cd test/subdir1/subdir2 && cd .. && pwd
```

### 1.3 Permissions (read/write/execute) basics
```bash
# Create a tiny script and show/modify permissions
echo -e '#!/usr/bin/env bash
echo run_ok' > exec_demo.sh
ls -l exec_demo.sh
chmod u+x exec_demo.sh         # add execute for the owner
ls -l exec_demo.sh
./exec_demo.sh                 # expect: run_ok
```

### 1.4 `ls` essentials
```bash
ls -a            # include hidden files
ls -ld */        # list only directories (details about dirs themselves)
ls -lh           # human-readable sizes
ls -lS           # sort by size
ls -lt           # sort by modification time
```

### 1.5 Text processing with `awk` and `sed`
```bash
# Save a long listing and print name + size columns
ls -l > ls_long.txt
awk '{print $9 "	" $5}' ls_long.txt | sed -e '1,1d'   # drop header if present

# Multi-rule replacement with backup
echo "The rain in Spain stays mainly in the plain." > easy_sed.txt
sed -i.bak 's/rain/snow/;s/Spain/Sweden/;s/plain/mountains/' easy_sed.txt
cat easy_sed.txt
```
> Tip: `awk` prints structured columns; `sed` applies regex edits line-by-line.

### 1.6 Compression & archiving
```bash
echo "Some data..." > data.txt
gzip -c data.txt > data.txt.gz     # keep original, write compressed copy
gunzip -c data.txt.gz | wc -c      # stream-decompress and count bytes

tar -cvf archive.tar subdir2 >/dev/null 2>&1  # create an archive (quiet)
tar -tf archive.tar | head -n 5            # list contents
```

### 1.7 User & group management (admin only)
> Demo by reading files; **running useradd/groupadd needs root**.
```bash
# Read user/group database (view-only)
head -n 5 /etc/passwd
head -n 5 /etc/group

# Admin examples (likely require sudo):
# useradd alice
# groupadd students
# usermod -aG students alice
# userdel alice
# groupdel students
# groups           # show groups of current user
```

---

## 2) Git Basics

> Goal: working tree → staging area (index) → repository; branching, merging, and resolving conflicts.

### 2.1 New repository & first commit
```bash
rm -rf git_demo_repo && mkdir git_demo_repo && cd git_demo_repo
git init -q
git config user.name  "Demo User"
git config user.email "demo@example.com"

echo "Hello" > file.txt
git add file.txt
git commit -qm "init: add file.txt"
```

### 2.2 Branch, edit, merge
```bash
git checkout -qb develop
echo "Develop line" >> file.txt
git add file.txt && git commit -qm "develop: add line"

git checkout -q master
echo "Master line" >> file.txt
git add file.txt && git commit -qm "master: add line"

git merge -q develop --no-edit || true          # merge (ignore failure for demo flow)
git log --oneline --graph --decorate -n 6
```

### 2.3 Resolve a simple conflict
```bash
# Merge and resolve
set +e; git merge develop; status=$?; set -e
if [ $status -ne 0 ]; then
  # Teaching shortcut: keep current working copy as "resolved"
  awk '{print}' file.txt > conflict.resolved && mv conflict.resolved file.txt
  git add file.txt
  git commit -qm "resolve: keep both lines"
fi

git status -s
git log --oneline --graph --decorate -n 10
```

### 2.4 Handy commands
```bash
git diff                 # unstaged changes vs index
git diff --staged        # staged changes vs last commit
git reset                # unstage (keep working copy)
git reset --hard         # discard changes in index and working copy
git checkout -- file.txt # restore file from HEAD
git tag v0.1 -m "tag example"
```

---
# Docker Basics — Beginner Tutorial (EN)

## 0) Quick Setup & Safety
- Run these in **Linux/macOS** terminals with Docker Engine installed.
- Use a sandbox directory so you don’t touch real data:
```bash
WORKDIR="$PWD/docker_demo_sandbox"; rm -rf "$WORKDIR"; mkdir -p "$WORKDIR"; cd "$WORKDIR"
```

## 1) Verify Docker (or use a cloud fallback)
```bash
docker version || echo "Docker missing. Use Play with Docker."
```

## 2) Your first container: hello-world
```bash
docker login 
docker pull hello-world
docker run --rm hello-world
```
> Purpose: verifies Docker client/daemon/network are OK. `--rm` auto-cleans the container.

## 3) Run a tiny Linux shell (Alpine)
```bash
docker pull alpine
docker run --rm -it alpine sh
# Inside the container:
#   echo "Hello from Alpine!"
#   uname -a
#   exit
```
> `-it` gives you an interactive shell; container stops when you `exit`.

## 4) Build & run a minimal Python image
Create two files side-by-side:

**Dockerfile.demo**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY app.py /app/app.py
CMD ["python", "app.py"]
```

**app.py**
```python
print("Hello from Dockerized Python!")
```

Build and run:
```bash
docker build -t demo:py -f Dockerfile.demo .
docker run --rm demo:py
```

## 5) Data persistence: volumes & bind mounts
```bash
# Named volume (Docker-managed)
docker volume create demo_vol
docker run --rm -v demo_vol:/data demo:py

# Bind mount your current folder (host) into /app (container)
docker run --rm -v "$PWD":/app demo:py
```
> Volumes persist across runs; bind mounts mirror live local files into the container.

## 6) Simple networking: publish a port
Use the same Python base to serve HTTP:

**server.Dockerfile**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN python -m pip install --no-cache-dir flask
COPY server.py /app/server.py
CMD ["python", "server.py"]
```

**server.py**
```python
from flask import Flask
app = Flask(__name__)

@app.get("/")
def hello():
    return "Hello from containerized Flask!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

Build & run, publishing port **8000** to host:
```bash
docker build -t demo:web -f server.Dockerfile .
docker run --rm -p 8000:8000 demo:web
# open http://localhost:8000 in your browser
```

## 7) Housekeeping (safe cleanup)
```bash
docker ps -a
docker images
docker volume ls
docker rmi demo:py demo:web       # remove images if you wish
docker volume rm demo_vol         # remove named volume
cd "$WORKDIR/.." && rm -rf "$WORKDIR"
```

---

# Conda Basics — Beginner Tutorial (EN)

## 0) Quick Setup & Safety
- Works with **Anaconda** or **Miniconda** (recommended). For faster solves, consider **Mamba**.
- Use a sandbox:
```bash
WORKDIR="$PWD/conda_demo_sandbox"; rm -rf "$WORKDIR"; mkdir -p "$WORKDIR"; cd "$WORKDIR"
```

> On Windows (PowerShell/CMD), activation uses `conda activate`. On Unix shells, it’s the same if `conda init` has been run.

## 1) Check conda & channels
```bash
conda --version
conda config --show channels | sed -n '1,10p'
# Optional: prefer conda-forge (popular community channel)
# conda config --add channels conda-forge
# conda config --set channel_priority strict
```

## 2) Create & activate your first environment
```bash
conda create -y -n demo_env python=3.11
conda activate demo_env
python -V
```

## 3) Install packages the conda way
```bash
# Example: scientific stack
conda install -y numpy pandas matplotlib
python - <<'PY'
import sys, numpy as np, pandas as pd, matplotlib
print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__, "| Pandas:", pd.__version__, "| Matplotlib:", matplotlib.__version__)
PY
```

## 4) Use pip **inside** a conda env (when needed)
Stay in `demo_env`:
```bash
python -m pip install "scikit-learn==1.5.*"
python - <<'PY'
import sklearn, sys
print("Python:", sys.version.split()[0], "| scikit-learn:", sklearn.__version__)
PY
```
> Rule of thumb: install as much as you can with conda; use pip inside the **activated** env for gaps.

## 5) Export & reproduce an environment
Export exact specs:
```bash
conda env export > env_full.yml
sed -n '1,25p' env_full.yml
```
Create a **minimal** human-edited file `env_min.yml`:
```yaml
name: demo_env
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - pandas
  - matplotlib
  - pip
  - pip:
      - scikit-learn==1.5.*
```
Recreate the env on another machine:
```bash
conda env create -f env_min.yml
```

## 6) Add the conda env as a Jupyter kernel (optional)
```bash
conda install -y ipykernel
python -m ipykernel install --user --name demo_env --display-name "Python (demo_env)"
# In Jupyter/VS Code, choose "Python (demo_env)" as the kernel
```

## 7) Update, list, and remove
```bash
conda list | sed -n '1,20p'      # see installed packages
conda update -y numpy            # one package
conda update -y --all            # entire env (be mindful)
conda deactivate                 # leave env
conda env remove -n demo_env     # delete env when no longer needed
cd "$WORKDIR/.." && rm -rf "$WORKDIR"
```

## 8) (Optional) Mamba for speed
```bash
# If mamba is available in base:
conda install -n base -y -c conda-forge mamba
mamba create -y -n fast_env python=3.11 numpy pandas
conda activate fast_env
python -c "import sys; print('Hello from', sys.executable)"
```

---

## In-Class Tips
- **Predict → Run → Reflect**: ask students what they expect each command to do before running it.
- Show mental models:  
  - Docker = **image** (immutable layers) + **container** (runtime) + **volume/bind mount** (data).  
  - Conda = **named env** + **channels** + **spec file (YAML)** for reproducibility.
- Keep a “cleanup cell” ready so students don’t leave junk on shared machines.
