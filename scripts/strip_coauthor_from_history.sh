#!/usr/bin/env bash
# Remove Co-authored-by trailers from all commits (Cursor/Claude auto-injected).
# Run from repo root. Requires force-push after: git push --force-with-lease origin main
set -euo pipefail

FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f \
  --msg-filter 'grep -v -i "^Co-authored-by:"' \
  -- --all

echo "Done. Verify with: git log -3 --format=%B"
echo "Then: git push --force-with-lease origin main"
