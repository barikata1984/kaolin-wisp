#!/bin/bash
set -e

# =============================================================================
# Container entrypoint
# - Initialises shell config for the runtime user
# - Installs project in editable mode (if pyproject.toml exists)
# - Drops to non-root user via gosu
# =============================================================================

TARGET_USER="${HOST_USER:-developer}"
TARGET_HOME=$(eval echo "~${TARGET_USER}" 2>/dev/null || echo "/home/${TARGET_USER}")
TARGET_GROUP=$(id -gn "${TARGET_USER}" 2>/dev/null || echo "${TARGET_USER}")

# ---- zsh bootstrap (only on first run) --------------------------------------
if [ ! -f "${TARGET_HOME}/.zshrc" ]; then
    mkdir -p "${TARGET_HOME}"
    cat > "${TARGET_HOME}/.zshrc" << 'ZSHRC'
# Minimal zsh config
autoload -Uz compinit && compinit
autoload -Uz vcs_info
precmd() { vcs_info }
zstyle ':vcs_info:git:*' formats ' (%b)'

setopt PROMPT_SUBST
PROMPT='%F{cyan}%~%f${vcs_info_msg_0_} %F{green}>%f '

# History
HISTFILE=~/.zsh_history
HISTSIZE=10000
SAVEHIST=10000
setopt SHARE_HISTORY HIST_IGNORE_DUPS

# Aliases
alias ll='ls -lah --color=auto'
alias la='ls -A --color=auto'
alias python='python3'

# Python venv is already in PATH via container ENV
ZSHRC
    chown "${TARGET_USER}:${TARGET_GROUP}" "${TARGET_HOME}/.zshrc"
fi

# ---- Ensure user directories exist with correct ownership --------------------
for d in "${TARGET_HOME}/.cache" "${TARGET_HOME}/.local" "${TARGET_HOME}/.config" "${TARGET_HOME}/.claude"; do
    mkdir -p "$d"
    chown "${TARGET_USER}:${TARGET_GROUP}" "$d" 2>/dev/null || true
done

# ---- Install project dependencies and editable mode (first run only) --------
SETUP_MARKER="/opt/venv/.project-installed"
if [ ! -f "${SETUP_MARKER}" ]; then
    if [ -f /workspace/requirements.txt ]; then
        echo "Installing project requirements..."
        pip install --no-cache-dir -r /workspace/requirements.txt 2>&1 | tail -3 || \
            echo "WARNING: requirements install failed (non-fatal, continuing...)"
    fi
    if [ -f /workspace/requirements_app.txt ]; then
        echo "Installing app requirements (with Cython pre-installed for glumpy)..."
        pip install --no-cache-dir Cython 2>&1 | tail -1
        pip install --no-cache-dir --no-build-isolation -r /workspace/requirements_app.txt 2>&1 | tail -3 || \
            echo "WARNING: app requirements install failed (non-fatal, continuing...)"
    fi
    if [ -f /workspace/setup.py ]; then
        echo "Installing project in editable mode..."
        FORCE_CUDA=1 python setup.py develop 2>&1 | tail -3 || \
            echo "WARNING: editable install failed (non-fatal, continuing...)"
    elif [ -f /workspace/pyproject.toml ]; then
        echo "Installing project in editable mode..."
        pip install --no-deps -e /workspace 2>&1 | tail -1 || \
            echo "WARNING: editable install failed (non-fatal, continuing...)"
    fi
    touch "${SETUP_MARKER}"
    echo "Project setup complete."
fi

# ---- Drop to non-root user and exec command ---------------------------------
if [ "$(id -u)" = "0" ] && [ "${TARGET_USER}" != "root" ]; then
    exec gosu "${TARGET_USER}" "$@"
else
    exec "$@"
fi
