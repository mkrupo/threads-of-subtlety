# Parallel development workflow — updated 2026-11-18

The `feature/sentiment` branch tracks `origin/feature/sentiment` on Max's
configured fork. Keep the colleague's repository as `upstream`.

## Why this fits the work

We need selected parser adapters, relation conventions, and provenance from
the colleague's work. We do not need to run the complete HC3/Longformer/motif
training pipeline. Sentiment preparation, baselines, and aggregation can stay
isolated under `research/sentiment/`, with their own locked environment.

A fork preserves common history, supports pull requests in either direction,
and provides an independent place to publish Max's experiment branch. It
does not require the colleague to change his workflow or add Max as a writer.
Forking does not provide the missing external trained model artifacts.

A standalone repository would also work with pinned dependencies, but would
require more adapter packaging and provenance setup now. It remains an option
if the sentiment code later becomes independent.

## Branch and update policy

- Initial local base is `dev` at `e658d857b1b3a803f91873c3a18258850a9c4e65`.
- Preserve that base in the experiment provenance, along with every reused
  research/feature commit. It is not the same as the current sentiment source
  state; uncommitted/new code also needs file hashes or its later commit.
- Do not base new work on `main` assuming it contains the colleague's latest
  changes: it remains the original implementation in the inspected snapshot.
- Fetch all relevant upstream branches, including `dev`,
  `research/unirst2-canonical`, and `feature/ettin-dmrst`.
- Review incoming changes and integrate selected compatible fixes. The
  research branch reorganizes the repository, so a blanket merge is not a
  routine synchronization step for this experiment layout.
- Avoid rebasing or altering an already published experiment snapshot. Make
  a new documented revision when an upstream change affects its inputs.
- Tag or otherwise identify the source commit used for reported runs, and
  keep dataset/model/split/environment hashes alongside the results.

## Current remote layout

```text
origin    -> git@github.com:mkrupo/threads-of-subtlety.git
upstream  -> git@github.com:ruthenian8/threads-of-subtlety.git
```

The local sentiment commit includes the source, tests, lockfile, protocol,
handover and selected results. Raw review text and full generated artifacts
are excluded via `.gitignore`. Imported Overleaf files and mails under
`RST_relation_granularity/` are preserved on disk and ignored. Their substantive
research context is summarized in `agent.md`; the original reference files
are not included in the commit.

GitHub's documented upstream-remote setup:
[Configuring a remote repository for a fork](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/configuring-a-remote-repository-for-a-fork).
