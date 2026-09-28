# Causal Capybara — a guide for the person doing the analysis

Written for a policy or public-health analyst who has a dataset and a question.
No package names are required to read this.

For the first session, open a worked example from Home, choose a research design, review its
diagnostic, then estimate and open the results. Your projects are saved as folders on your own
computer. See the [installation guide](INSTALL.md) for the Windows download and source setup.

---

## 1. The shape of a session

1. **Say what you want to know.** The bar at the top of the window is a sentence with slots:
   *Effect of [treatment] on [outcome] for [population], compared to [comparison]*. Filling it in is how
   you start; there is no separate "new analysis" dialog.
2. **Pick the picture that matches how treatment was decided.** Not a list of estimators — a set of
   cards with a schematic and one sentence each. "A cutoff decided eligibility." "Some places got it
   earlier than others."
3. **Complete the board.** Each labelled box has an "add" button that lists the columns of the right
   type. You can also drag a variable from the list on the Data screen onto a box, and every box has a
   keyboard equivalent in the inspector on the right, so the picture is the beginner path, not the only
   one.
4. **Look at the diagnostic.** As soon as the board has its minimum roles, a live sketch appears: the
   overlap histogram, the adoption heatmap, the binned scatter, the first stage. This is the step people
   skip and then regret.
5. **Estimate a set of methods.** The default button runs the recommended set, not the first card.
6. **Probe it.** Ask how fragile the conclusion is before you write it up.
7. **Report.** A document a colleague can read and a referee can rerun.

You can move between these in any order. Nothing traps you on page four of ten.

---

## 2. Choosing the estimand: which number do you actually want?

Before any method, decide what the number is an average *of*. The app shows these as sentences:

| The sentence | What it is called | When it is what you want |
|---|---|---|
| If everyone got the programme, how would the average outcome change? | ATE | Deciding whether to roll something out universally |
| For people who actually got it, what did it do? | ATT | Evaluating a programme as it ran |
| For people who didn't, what would it have done? | ATC | Deciding whether to expand it |
| For people who only got it because of the encouragement or the cutoff | LATE | You have an instrument, or a threshold |
| For everyone who was offered it | ITT | The policy can offer, not compel |
| For each group that adopted, what did their own rollout do? | Group-time ATT | A policy adopted at different times |

Estimands your design cannot identify stay on screen, greyed, with one line saying why. That is usually
more useful than the list of ones it can.

**The commonest mistake** is reporting the effect on the people who took up a programme as if it were the
effect of offering it. Both numbers are correct; they answer different questions, and a policy that can
only make an offer is described by the second.

---

## 3. What the diagnostics are telling you

### Overlap (observational designs)

Two histograms of the estimated probability of treatment, one per arm. Where they do not overlap, there
is nobody to compare with, and any number for those units comes from the model rather than the data.

Look at two things: how much of the sample sits outside the range both arms occupy, and the **effective
sample size**. A thousand rows reweighted to resemble the treated group can be worth eighty
observations. That is the honest denominator, and when it collapses your confidence interval is narrower
than your evidence.

### Balance, before and after (the Love plot)

One point per covariate, before adjustment and after. The line at 0.1 is a convention, not a law: what
matters is whether a covariate that still differs is one that strongly predicts the outcome.

A flat Love plot does **not** mean confounding is gone. It means the variables you measured are now
comparable. It is silent about the ones you did not.

### The adoption heatmap (policy over time)

Rows are units, columns are periods, cells light up when treatment turns on. If it is a staircase rather
than a block, groups were treated at different times, and the app will say so on the card. Two-way fixed
effects then averages comparisons that use already-treated units as controls, with weights that can go
negative. That is not a subtlety; it can flip the sign.

### The binned scatter (a cutoff design)

Local averages either side of the threshold. If you cannot see a jump here, be suspicious of an
estimator that reports one. The bandwidth path shows the estimate across a range of bandwidths — an
answer that exists only at one bandwidth is not an answer.

### The first stage (an instrument)

How much the instrument actually moves treatment. Read the **effective F**, not the ordinary one. Below
about ten, the usual confidence interval is not trustworthy, and the app will report a
weak-instrument-robust set instead — sometimes an unbounded one, which is the honest answer.

---

## 4. The assumption ledger

Every result carries a table of what the design assumes and what the diagnostics said about it. There
are five statuses and none of them is "passed":

* **assumed** — nothing here can speak to it.
* **not contradicted** — a diagnostic looked and found nothing wrong. That is weaker than it sounds.
* **weakened** — a diagnostic found something.
* **untested** — nothing looked.
* **does not apply**.

*No unmeasured confounding* stays **untested** in every observational analysis, however good the balance
plot is. This is not pedantry: it is the difference between a result and a claim.

---

## 5. Reading a forest plot

When more than one method ran, there is no headline number. The forest — one row per method, with its
interval — *is* the result.

This is deliberate. Choosing the estimate you liked best after seeing them all is how careful people
produce careless research. If the methods disagree, the range is your finding, and the generated prose
will describe it as a range.

A triangle instead of a circle means the run is **provisional**: something specific happened that the
app does not think you should ignore. Hover it to see what.

You may nominate a preferred estimate for a report. It is stored and shown as a choice, not as a
conclusion.

---

## 6. The probe bench

A set of moves, matched to your design, that ask how fragile the conclusion is.

* **Placebo outcome** — something the treatment should not affect. An effect there is a measurement of
  your bias.
* **Placebo treatment** — a fake date, or a permuted assignment.
* **Negative control** — a variable that shares the confounding but cannot be caused by the treatment.
* **Subset stability** — re-estimate on random subsets; a stable design gives a similar answer.
* **Sensitivity** — how strong would an unmeasured confounder have to be, *compared with the ones you
  did measure*, to explain the estimate away? This is the most useful question in the bench, because it
  turns an untestable assumption into a quantity you can argue about.
* **Trimming curve** — instead of choosing a trimming threshold, see the estimate across all of them.
* **Honest DiD** — how far could trends have differed before the conclusion breaks?

None of these can show a design is right. A clean placebo rules out one route to a false positive. A
dirty one demonstrates a problem, which is far more informative.

---

## 7. Reading up on any of this

`Help -> Learn: the method catalogue` opens every design, method, estimand, assumption, diagnostic and
worked example as an article: what it is, what it assumes, when to use it, when not to, and what to read
next. It needs no project and no dataset, and it works whether or not an engine is running.

The cards in the app are named in plain language, so each article also lists the names the literature
uses and the search answers to both — "RDD", "propensity score" and "two-way fixed effects" all find the
right card.

---

## 8. Things the app will not do

* Tell you an effect is causal because you chose a method named "matching".
* Delete observations to make an overlap plot look better.
* Show one number when several defensible methods disagree.
* Discover the causes of your outcome from the data. Structure learning is gated behind Advanced,
  labelled as what it is, and anything it produces is a candidate you must accept.
* Hide the package and version that produced a number. Both are on every row.

---

## 9. Reproducing a result

Every project is a folder. Runs inside it are immutable: re-estimating creates a new run rather than
overwriting the old one, which is how you get provenance without a database.

The **Code** tab on any result has three projections of the same specification — the spec itself, an R
script and a Python script. They are generated, so they cannot drift from what actually ran. A referee
can take either script and land in the same result format.

If you hand-edit a script, the analysis forks: your custom version still has to return a result in the
canonical shape before it can join a comparison.

---

## 10. When an engine is missing

**If nothing computes at all.** Causal Capybara does its arithmetic with Python. The v0.1.0
Windows release candidate bundles its Python runtime; a source checkout can use a
configured environment. If the engine cannot start, the window still opens and
explains the problem. Its recovery screen can check an interpreter you already
have, and the bundled method catalogue remains readable while you sort it out.

Python is required to compute, manage projects and run the local engine. R is optional and supports
a smaller subset of methods. If the engine cannot start, the bundled method catalogue remains
readable. Selecting a Python interpreter does not install missing packages; follow the setup
instructions above or ask the person who installed the app to prepare that environment.

Once Python is working, Settings → Engines shows which methods and packages are available.
Review the destination and package list before approving an installation. R must be installed
separately if you want to use its methods.

Where the same method exists on both engines you can run both. They should agree; where they do not, the
comparison screen shows the disagreement rather than averaging it away, because a difference usually
means the two implementations made different default choices, and that is worth knowing before a referee
finds it.
