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

# ---- Install project in editable mode (if pyproject.toml exists) ------------
if [ -f /workspace/pyproject.toml ]; then
    echo "Installing project in editable mode..."
    pip install --no-deps -e /workspace 2>&1 | tail -1 || \
        echo "WARNING: editable install failed (non-fatal, continuing...)"
fi

# ---- Drop to non-root user and exec command ---------------------------------
if [ "$(id -u)" = "0" ] && [ "${TARGET_USER}" != "root" ]; then
    exec gosu "${TARGET_USER}" "$@"
else
    exec "$@"
fi
