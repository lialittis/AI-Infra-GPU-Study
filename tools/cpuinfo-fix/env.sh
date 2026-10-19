# Source this file (do not execute it) to enable the /proc/cpuinfo fix in the current shell:
#   source tools/cpuinfo-fix/env.sh
# Builds the shim and regenerates the corrected cpuinfo on every source (both are cheap).
_cpuinfo_fix_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$_cpuinfo_fix_dir/build"
gcc -O2 -shared -fPIC -o "$_cpuinfo_fix_dir/build/cpuinfo_redirect.so" "$_cpuinfo_fix_dir/cpuinfo_redirect.c" -ldl
python3 -I "$_cpuinfo_fix_dir/make_cpuinfo.py" > "$_cpuinfo_fix_dir/build/cpuinfo"
export CPUINFO_OVERRIDE="$_cpuinfo_fix_dir/build/cpuinfo"
export LD_PRELOAD="$_cpuinfo_fix_dir/build/cpuinfo_redirect.so${LD_PRELOAD:+:$LD_PRELOAD}"
unset _cpuinfo_fix_dir
