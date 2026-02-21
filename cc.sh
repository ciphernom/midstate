#!/usr/bin/env bash

# cc.sh
# Concatenates all .rs files in src/ with file path headers

set -euo pipefail

OUTPUT_FILE="all_rust_code.txt"
SRC_DIR="src"

# Remove old file if it exists
[ -f "$OUTPUT_FILE" ] && rm "$OUTPUT_FILE"

echo "Concatenating all .rs files from ${SRC_DIR}/ into ${OUTPUT_FILE}"
echo "──────────────────────────────────────────────────────────────"
echo ""

# Find all .rs files, sort them in a reasonable order
find "$SRC_DIR" -type f -name "*.rs" -print0 \
  | sort -z \
  | while IFS= read -r -d '' file; do

    # Get relative path (without leading src/)
    rel_path="${file#${SRC_DIR}/}"

    echo "Adding:  $rel_path"

    {
      # Header
      printf "\n\n"
      printf "══════════════════════════════════════════════════════════════\n"
      printf " FILE: %s\n" "$rel_path"
      printf "══════════════════════════════════════════════════════════════\n"
      printf "\n"

      # Content
      cat "$file"

      # Small separator after file
      printf "\n\n"

    } >> "$OUTPUT_FILE"

done

echo ""
echo "Done. Output written to: ${OUTPUT_FILE}"
echo "Total lines: $(wc -l < "$OUTPUT_FILE")"
echo ""
echo "Hint: You can view it nicely with:"
echo "    bat --language rust ${OUTPUT_FILE}"
echo "or"
echo "    less +G ${OUTPUT_FILE}"
