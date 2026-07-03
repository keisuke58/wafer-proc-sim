#!/usr/bin/env bash
# Compile and run the streaming-Sobel testbench with Icarus Verilog.
# Requires: iverilog, vvp.  Exits non-zero if the self-checking TB fails.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="$(mktemp -d)/sobel_sim"
iverilog -g2012 -Wall -o "$out" "$here/sobel3x3.v" "$here/tb_sobel3x3.v"
vvp "$out"
