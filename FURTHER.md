# Further — the sources behind the talk, and things worth poking at

Everything cited in the session, plus a few tools. One line each on why it's here.

## The system you're shipping (part 1)

- **τ-bench** (Sierra / Yao et al.) — https://arxiv.org/abs/2406.12045 — where `pass^k` comes from, and the measured 60%-once vs under-25%-across-eight numbers on the reliability slide.
- **Thoughts on a month with Devin** (Answer.AI, Jan 2025) — https://www.answer.ai/posts/2025-01-08-devin.html — three engineers, twenty real tasks, three successes. The planning-failure story, told by the people it happened to.
- **Replit incident coverage** (The Register, Jul 2025) — https://www.theregister.com/2025/07/21/replit_saastr_vibe_coding_incident/ — the code-freeze database wipe. What "no verification layer" costs.
- **Moffatt v. Air Canada** (ABA summary of the 2024 tribunal ruling) — https://www.americanbar.org/groups/business_law/resources/business-law-today/2024-february/bc-tribunal-confirms-companies-remain-liable-information-provided-ai-chatbot/ — a policy that lives in a document instead of the loop is legally still your policy.

## Benchmarks vs. reality

- **METR: many SWE-bench-passing PRs would not be merged** (Mar 2026) — https://metr.org/notes/2026-03-10-many-swe-bench-passing-prs-would-not-be-merged-into-main/ — real maintainers merged "passing" AI PRs ~24 points below the benchmark rate. The asterisk on every vendor score.
- **Artificial Analysis — coding agents** — https://artificialanalysis.ai/agents/coding-agents — live cost-per-task and token-per-task across model+harness combos. Good for scouting; their own methodology page says mergeability isn't measured.
- **SkateBench** (Theo Browne) — https://skatebench.t3.gg/ — a personal benchmark on skateboarding-trick knowledge where leaderboard giants score anywhere from 97% to 15%. The case for evaluating on *your* task, in one page.

## Evals (parts 2 and 3 preview)

- **OpenAI cookbook: testing prompts for regressions** — https://developers.openai.com/cookbook/examples/evaluation/use-cases/regression — a vendor's own take on golden-set-style regression testing.
- **Beyond vibe checks: a PM's guide to evals** (Aman Khan, Lenny's Newsletter, Apr 2025) — https://www.lennysnewsletter.com/p/beyond-vibe-checks-a-pms-complete — the non-engineering register: human vs code vs judge evals. Partially paywalled.

## Tools worth poking at

- **This repo's `--live` mode** — swap the scripted brain for a real model with one flag (`cp .env.example .env`, add an OpenRouter key). Watching a cheap model fight the harness teaches more than any article here.
- **OpenRouter** — https://openrouter.ai — one API key, most models. We already have access; it's how the live mode here runs.
- **Respan** — https://www.respan.ai — gateway + tracing + evals in one product; free tier. We're currently evaluating it — treat as "interesting," not "endorsed."
