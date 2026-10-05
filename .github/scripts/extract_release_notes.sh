#!/usr/bin/env bash
# Print the changelog section of a version as Markdown.
# Usage: extract_release_notes.sh VERSION [CHANGELOG]
set -euo pipefail

VERSION="${1:?Usage: $0 VERSION [CHANGELOG]}"
CHANGELOG="${2:-docs/changelog.rst}"

notes=$(
    awk -v header="[$VERSION]" '
        index($0, header) == 1 { found = 1; getline; next }
        found && /^\[/ { exit }
        found {
            if ($0 ~ /^~+$/) { lines[n - 1] = "### " lines[n - 1]; next }
            gsub(/``/, "`")
            lines[n++] = $0
        }
        END { for (i = 0; i < n; i++) print lines[i] }
    ' "$CHANGELOG" | sed -e '/./,$!d'
)

if [ -z "$notes" ]; then
    echo "No changelog entry found for version $VERSION in $CHANGELOG" >&2
    exit 1
fi

printf '%s\n' "$notes"
