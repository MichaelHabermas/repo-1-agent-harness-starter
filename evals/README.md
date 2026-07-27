# evals/

Nothing here yet. That is the point of this repo.

You have an agent and a harness. You can see what it did, bound what it costs,
and reproduce any run. What you cannot do is say whether it was *right*, or
stop a bad one from shipping.

This directory is where that goes, in section two:

    evals/goldens/      the golden set -- versioned, owned, four buckets
    evals/graders/      deterministic, trajectory, and judge
    evals/labels/       what a domain expert said, by hand
    evals/runner.py     pass^1 and pass^k
    evals/calibrate.py  is the judge trustworthy?
    scripts/gate.py     turns a failing metric into a failing build

In the meantime, the thing that makes all of it possible is already running:
every call to the app appends a full trace to `traces/runs.jsonl`. Those traces
are the raw material. Generate some before section two -- read a few by hand and
you will already have opinions about what belongs in the golden set.
