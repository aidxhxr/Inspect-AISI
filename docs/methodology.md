# Methodology

This document explains what each measurement in the suite means and why it is
computed the way it is. Nothing here reports a result; results come from the
logs produced by `aisi-evals run`.

## Design choices that apply everywhere

**Paired comparisons.** Every model answers the same samples in the same order
of presentation, and `aisi-evals compare` tests the per-sample difference
between two models rather than comparing two independent means. On a fixed
question set the difference between models is usually far less variable than
the scores themselves, so paired analysis detects smaller effects with the
same budget.

**Standard errors that respect the design.** When a question appears more than
once (epochs, or the permutations in `hellaswag_permuted`), observations are
not independent. Standard errors are clustered by question using the
finite-cluster-corrected estimator in Appendix A of Miller (2024), "Adding
Error Bars to Evals". The implementation is checked against Inspect's own
`stderr(cluster=...)` in the test suite.

**Multiple comparisons.** When a leaderboard compares several models to the
best one, the paired p-values are Holm-corrected.

**Sample size before running.** `aisi-evals power --n N` prints the minimum
difference detectable at 80 percent power for N samples per model, and
`--diff D` prints the N needed to detect D. The point is to decide whether a
run can answer the question before paying for it.

**Grader dependence is reported, not hidden.** Model-graded verdicts are a
source of error in their own right. The math scorer records which stage of
its cascade decided each sample and reports the share that reached the model
grader; the QA scorer keeps the grader's raw letter and the full completion
in the score so verdicts can be audited in the log viewer.

## Tasks

### `simpleqa_calibrated`

Short-form factuality on SimpleQA-Verified. The prompt asks for a final answer
or an explicit "I don't know", plus a stated confidence from 0 to 100.

Grading follows the SimpleQA paper's three-way scheme: correct, incorrect, or
not attempted. The grader is given the question, the gold target and the
predicted answer only, and is told not to use outside knowledge.

Reported metrics:

- `accuracy`: correct over all samples (abstentions count as wrong).
- `correct_given_attempted`: correct over attempted; the precision of the
  answers the model chose to give.
- `attempted_rate`, `abstention_rate`.
- `simpleqa_f1`: harmonic mean of accuracy and correct-given-attempted, the
  single number the SimpleQA paper recommends.
- `brier`: mean squared gap between stated confidence and correctness over
  attempted answers. Lower is better; 0.25 is what constant 50 percent
  confidence scores on a balanced set.
- `ece`: expected calibration error with ten equal-width bins. Zero means
  that within each confidence band the model is right exactly as often as it
  claims.
- `aurc`: area under the risk-coverage curve, the error rate the model would
  have at each coverage if it answered in order of confidence. Lower means
  confidence is a useful signal for deciding which answers to trust.
- `selective_accuracy`: accuracy on the most confident half of attempted
  answers.

A sample with a missing or malformed confidence still counts toward accuracy
but is excluded from the calibration metrics; `mean_confidence` is reported
so the exclusion rate can be noticed.

### `math500`

MATH-500 with a scoring cascade: the final `ANSWER:` line is normalized
(boxed, text, spacing, trivial prefixes); if it matches the target exactly
that settles it; otherwise both sides are parsed with sympy and compared
symbolically; only if sympy cannot decide does a grader model see the two
answers. `decided_by` reports the share of verdicts from each stage, which
bounds how much of the score depends on the grader.

### `hellaswag_permuted`

Each HellaSwag question is presented `n_permutations` times with the options
shuffled (the first copy keeps the original order). Beyond accuracy with
question-clustered standard errors, two quantities are reported:

- `permutation_consistency`: the fraction of questions where the model picked
  the same option text under every ordering. A model that is right for the
  right reasons should be close to 1 regardless of its accuracy.
- `position_bias`: the gap between accuracy when the correct option is at
  its best position and at its worst. Large values mean the letter, not the
  content, is driving answers.

### `ctf`

Five capture-the-flag challenges of increasing difficulty, each in its own
Docker container with no network access. A react agent with bash and python
tools has a bounded number of messages, tokens and submit attempts. The flag
is injected at image build time and never appears in the prompt. Accuracy is
reported overall and per difficulty level.

## What the tests prove and what they do not

The test suite runs every task end to end against Inspect's mock model
provider with scripted outputs, so parsing, grading logic, metric arithmetic
and log analysis are verified without network access or API keys. The tests
do not exercise real models, real datasets or Docker builds; those happen
when the suite is run.
