#!/usr/bin/env bash
# Build + test + run the DISCO APC C# module.
#
#   * Preferred:  dotnet build && dotnet run   (if the .NET SDK is installed)
#   * Fallback :  Mono mcs/mono                (used in this repo's environment)
#
# Usage:  ./build.sh [output_dir]     (default output dir = current directory)
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-.}"

if command -v dotnet >/dev/null 2>&1; then
  echo "== dotnet build =="
  dotnet build -c Release AlgoLib.csproj
  echo "(demo/tests: run via mono fallback below or add Exe csproj wrappers)"
fi

if command -v mcs >/dev/null 2>&1; then
  echo "== mono build =="
  mcs -target:library -out:AlgoLib.dll src/*.cs
  mcs -r:AlgoLib.dll -out:Tests.exe tests/Tests.cs
  mcs -r:AlgoLib.dll -out:Demo.exe  demo/Program.cs
  echo "== tests =="
  mono Tests.exe
  echo "== demo =="
  mono Demo.exe "$OUT"
else
  echo "mcs (Mono) not found; install with: apt-get install -y mono-mcs" >&2
  exit 1
fi
