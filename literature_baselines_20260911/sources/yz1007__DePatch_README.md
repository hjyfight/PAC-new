# depatch

Automated dependency patching for npm projects. Scans your dependency tree for security vulnerabilities and outdated packages, then generates minimal, testable PRs with verified fixes.

## Features

- **Vulnerability scanning** — uses npm audit + OSV database for comprehensive coverage
- **Smart version resolution** — finds the minimum patch version that fixes each CVE
- **Atomic PR generation** — one PR per vulnerability, with detailed change descriptions
- **Compatibility verification** — checks peer dependency constraints before suggesting updates
- **Changelog integration** — auto-extracts relevant changelog entries for each update
- **Batch mode** — group related updates for efficiency
- **Dry-run mode** — preview changes without modifying files
- **CI/CD scheduling** — run on cron to catch new CVEs within hours

## Installation

```bash
npm install -g depatch
# or use directly
npx depatch scan
```

## Usage

### Scan for Updates

```bash
# Show all available dependency updates
npx depatch scan

# Security-only mode
npx depatch scan --security-only

# Include minor and patch updates
npx depatch scan --level minor

# JSON output for CI
npx depatch scan --json
```

### Generate Patches

```bash
# Create PRs for security fixes
npx depatch patch --security

# Create PRs for all available updates
npx depatch patch --all

# Dry run — show what would change
npx depatch patch --dry-run

# Batch related updates into single PRs
npx depatch patch --batch
```

### GitHub Integration

```bash
# Authenticate and create PRs
npx depatch patch --security --github

# Create PRs with auto-generated labels
npx depatch patch --security --github --labels "dependencies,security"
```

### CI/CD Scheduled Scanning

```yaml
# .github/workflows/depatch.yml
name: Dependency Patches
on:
  schedule:
    - cron: "0 6 * * 1-5" # Weekdays at 6am
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npx depatch patch --security --github
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### Programmatic API

```typescript
import { scan, createPatch } from "depatch";

const updates = await scan("./package.json", { level: "patch" });
console.log(updates); // { dep, current, latest, isSecurity, cves }

const pr = await createPatch(updates[0], {
  branch: "depatch/fix-cve-2024-1234",
  commitMessage: "fix: update express to patch CVE-2024-1234",
});
```

## Output Example

```
  depatch — scanning dependencies...

  Security updates (3):
    ✗ express 4.18.2 → 4.19.2  (CVE-2024-29041)
    ✗ axios 1.6.0 → 1.6.8      (CVE-2024-39338)
    ✗ follow-redirects 1.15.4 → 1.15.6  (CVE-2024-28849)

  Minor updates (5):
    ~ typescript 5.3.3 → 5.4.2
    ~ vitest 1.2.0 → 1.3.1
    ~ eslint 8.56.0 → 8.57.0

  Run `npx depatch patch --security` to create PRs.
```

## Configuration

```json
// .depatchrc.json
{
  "ignore": ["webpack", "babel-loader"],
  "batchBy": "ecosystem",
  "autoMerge": {
    "patch": true,
    "minor": false
  },
  "schedule": "weekday-mornings",
  "labels": ["dependencies"]
}
```

## License

MIT
