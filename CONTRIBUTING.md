# Contributing to LLM Consciousness Self-Attribution

Thanks for helping. This project is part of [Station House](https://huggingface.co/Station-house),
and contributions here follow the same conventions as the rest of Station House: small,
self-contained pull requests, sources for every claim, review in your own words, and
agent output that is always labelled. This guide covers how to send a change and how to
review one.

## Sending a change

1. **Look before you build.** Check the open issues and pull requests, and the
   [checklist of further work](./README.md#checklist-of-further-work-to-do), so you don't
   duplicate something in flight. For anything larger than a small fix, open an issue
   first and say what you plan to do.

2. **Fork and branch.** You don't have write access to this repo, so work from a fork and
   open a pull request from it.

   ```bash
   gh repo fork JoyeeChen/LLMConsciousnessSelfAttributionRepo --clone
   cd LLMConsciousnessSelfAttributionRepo
   git switch -c <short-descriptive-branch>
   ```

3. **Make one self-contained change.** One pull request does one thing: a new eval question,
   a fix, a config change, a plotting update. Unrelated changes go in separate pull
   requests, so each can be reviewed and merged on its own. Adding an eval question is
   described in [Where the questions live, and how to add one](./README.md#where-the-questions-live-and-how-to-add-one).

4. **Check it.** Run the test suite before you push, and fix anything it reports.

   ```bash
   uv run pytest tests/
   ```

   If your change affects numbers in the dashboard or the charts, regenerate them as
   described in the README, and say in the pull request which logs they came from.

5. **Open the pull request.**

   ```bash
   git push -u origin <short-descriptive-branch>
   gh pr create --fill
   ```

   The description should say:
   - what the change does and why it's worth making,
   - the source or evidence behind any factual claim, number or result it adds or changes
     (a paper, a log file, a run id).

A maintainer will review it for correctness before merging. For results and numbers that
means checking them against the logs or source you cite, since everything downstream
(the dashboard, the write-ups) trusts them.

## Reviewing someone else's pull request

Review is where quality actually gets decided, so anyone can review — you don't need to be
a maintainer. Pick an open pull request and judge it on three things:

- **Need.** Does the project need this? A new eval item or method should cover something
  the repo doesn't already, and a code change should solve a problem contributors actually
  hit.
- **Correctness.** Check claims and numbers against the cited source or logs, that the
  tests pass (`uv run pytest tests/`), and that configs and labels match what the change
  really does.
- **Style.** Does it follow this guide? One self-contained change per pull request, code
  that matches the surrounding style, and a description that explains the change and its
  source.

Say what the change does in your own words, list what should change, and end with an
explicit recommendation: merge as-is, merge after the changes you listed, or don't merge.

### Agent-written comments

Write your review yourself. Using an agent to understand a pull request is fine and
encouraged — have it explain a diff, check a source, or walk you through the code — but
the review itself should be your own words and your own judgement.

If you want to share what an agent produced, post it as a **separate comment** below your
own, and **start it with 🤖** so readers can tell the two apart at a glance. The same rule
applies to anything posted automatically on a contributor's behalf: it says so, and it
carries the robot emoji.

If you use a coding assistant on your own changes, point it at [`AGENT.md`](./AGENT.md), which
holds the instructions for assistants working in this repo.
