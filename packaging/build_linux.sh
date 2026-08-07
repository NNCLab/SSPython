#!/usr/bin/env bash
set -Eeuo pipefail

skip_tests=0
skip_archive=0
debug_console=0

usage() {
    cat <<'EOF'
Usage: packaging/build_linux.sh [options]

Build the SSPython Linux PyInstaller bundle on a Linux host.

Options:
  --skip-tests       Do not run the unit test suite before building.
  --skip-archive     Do not create a compressed tar archive after smoke testing.
  --debug-console    Build with console output enabled.
  -h, --help         Show this help message.
EOF
}

for argument in "$@"; do
    case "$argument" in
        --skip-tests|-SkipTests)
            skip_tests=1
            ;;
        --skip-archive|-SkipArchive)
            skip_archive=1
            ;;
        --debug-console|-DebugConsole)
            debug_console=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $argument" >&2
            usage >&2
            exit 2
            ;;
    esac
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
build_environment="$project_root/.venv-build-linux"
spec_path="$script_dir/SSPython.spec"
dist_path="$project_root/dist"
work_path="$project_root/build/pyinstaller"
application_path="$dist_path/SSPython"
executable_path="$application_path/SSPython"
version_file="$project_root/core/version.py"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to create the isolated build environment." >&2
    exit 1
fi

application_version="$(
    sed -nE 's/^__version__[[:space:]]*=[[:space:]]*"([^"]+)".*$/\1/p' "$version_file" | head -n 1
)"
if [[ -z "$application_version" ]]; then
    echo "Could not read __version__ from $version_file." >&2
    exit 1
fi

previous_project_environment="${UV_PROJECT_ENVIRONMENT-}"
previous_console_setting="${SSPYTHON_BUILD_CONSOLE-}"
previous_qt_platform="${QT_QPA_PLATFORM-}"
previous_matplotlib_backend="${MPLBACKEND-}"
previous_smoke_log="${SSPYTHON_SMOKE_LOG-}"
had_project_environment="${UV_PROJECT_ENVIRONMENT+x}"
had_console_setting="${SSPYTHON_BUILD_CONSOLE+x}"
had_qt_platform="${QT_QPA_PLATFORM+x}"
had_matplotlib_backend="${MPLBACKEND+x}"
had_smoke_log="${SSPYTHON_SMOKE_LOG+x}"

restore_environment() {
    if [[ -n "$had_project_environment" ]]; then
        export UV_PROJECT_ENVIRONMENT="$previous_project_environment"
    else
        unset UV_PROJECT_ENVIRONMENT
    fi

    if [[ -n "$had_console_setting" ]]; then
        export SSPYTHON_BUILD_CONSOLE="$previous_console_setting"
    else
        unset SSPYTHON_BUILD_CONSOLE
    fi

    if [[ -n "$had_qt_platform" ]]; then
        export QT_QPA_PLATFORM="$previous_qt_platform"
    else
        unset QT_QPA_PLATFORM
    fi

    if [[ -n "$had_matplotlib_backend" ]]; then
        export MPLBACKEND="$previous_matplotlib_backend"
    else
        unset MPLBACKEND
    fi

    if [[ -n "$had_smoke_log" ]]; then
        export SSPYTHON_SMOKE_LOG="$previous_smoke_log"
    else
        unset SSPYTHON_SMOKE_LOG
    fi
}
trap restore_environment EXIT

export UV_PROJECT_ENVIRONMENT="$build_environment"
if (( debug_console )); then
    export SSPYTHON_BUILD_CONSOLE=1
else
    export SSPYTHON_BUILD_CONSOLE=0
fi

(
    cd "$project_root"

    uv sync \
        --frozen \
        --no-dev \
        --group build

    if (( ! skip_tests )); then
        export QT_QPA_PLATFORM=offscreen
        export MPLBACKEND=Agg
        uv run \
            --frozen \
            --no-dev \
            --group build \
            python -m unittest discover tests -v
    fi

    uv run \
        --frozen \
        --no-dev \
        --group build \
        python -m PyInstaller \
        --noconfirm \
        --clean \
        --distpath "$dist_path" \
        --workpath "$work_path" \
        "$spec_path"
)

required_bundle_files=(
    "SSPython"
    "_internal/assets/icon.png"
    "_internal/style/dark_theme.qss"
    "_internal/style/light_theme.qss"
    "_internal/mne_icalabel/iclabel/network/assets/ICLabelNet.pt"
)

for relative_path in "${required_bundle_files[@]}"; do
    absolute_path="$application_path/$relative_path"
    if [[ ! -f "$absolute_path" ]]; then
        echo "The frozen bundle is missing required file: $relative_path" >&2
        exit 1
    fi
done

if [[ ! -x "$executable_path" ]]; then
    echo "The frozen executable is missing or is not executable: $executable_path" >&2
    exit 1
fi

lsl_libraries=( "$application_path"/_internal/mne_lsl/lsl/lib/liblsl*.so* )
if (( ${#lsl_libraries[@]} == 0 )) || [[ ! -e "${lsl_libraries[0]}" ]]; then
    echo "The frozen bundle is missing the Linux liblsl shared object." >&2
    exit 1
fi

export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
smoke_log_path="$work_path/frozen-smoke-test.log"
export SSPYTHON_SMOKE_LOG="$smoke_log_path"
rm -f "$smoke_log_path"

if ! (
    cd "$application_path"
    "$executable_path" --smoke-test
); then
    if [[ -f "$smoke_log_path" ]]; then
        cat "$smoke_log_path"
    fi
    echo "Frozen application smoke test failed." >&2
    exit 1
fi

if (( ! skip_archive )); then
    archive_path="$dist_path/SSPython-${application_version}-linux-$(uname -m).tar.gz"
    rm -f "$archive_path"
    tar -C "$dist_path" -czf "$archive_path" SSPython
fi

echo "SSPython $application_version Linux release artifacts are in $dist_path"
