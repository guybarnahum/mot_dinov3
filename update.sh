#!/bin/bash
set -e

# Make sure you're on the main branch
git checkout main

# Rewrite every tracked file by appending + removing a harmless comment
for f in $(git ls-files); do
  # Skip binary files just in case
  if file "$f" | grep -q "text"; then
    echo "" >> "$f"       # append a blank line
    sed -i '' -e '$d' "$f"  # remove the blank line (macOS/BSD sed)
    # For Linux use: sed -i -e '$d' "$f"
  fi
done

git add .
git commit -m "chore: refresh all file timestamps"
git push origin main

