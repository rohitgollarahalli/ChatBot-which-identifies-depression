# Data card: research-mode corpora

These two files train the research-mode classifier (`moodcheck research`). They
are small and were put together for a 2018 student project. This card records
what is known about them, what is not, and why the classifier trained on them
**does not detect depression**.

| File | SHA-256 |
| --- | --- |
| `depressed.txt` | `52c725ed30380262b52e56e4daa6094ec4102d734e51842b17535d5380019986` |
| `not_depressed.txt` | `6aeccf3ddd5839c5f5986b6d0e16ae6393f9eafec39cc8fe4167753494b0be45` |

`tests/test_data_card.py` fails if either file changes without this table being
updated, so a change to the training data is always a visible change to this card.

## At a glance

| | `depressed.txt` | `not_depressed.txt` |
| --- | --- | --- |
| Label used | `depressed` | `not_depressed` |
| Source | Posts from the Reddit forum r/depression | HappyDB, a crowdsourced corpus of "happy moments" |
| Collected | By the original project team; the report cites r/depression as retrieved 14 April 2018 | HappyDB published January 2018; this file is a slice of it |
| Format | Free text, UTF-8, 498 lines | CSV in HappyDB's `cleaned_hm.csv` columns: 154 moments, read as 150 records (see known issues) |
| Documents after loading | 423 (one per sentence) | 150 (one per happy moment: `cleaned_hm`, 0-based column 4) |
| Used for training | 150 (randomly down-sampled to match) | 150 |
| License | **None known** (see below) | **None stated** upstream (see below) |
| How the label was assigned | By source: every sentence from the forum text is `depressed` | By source: every happy moment is `not_depressed` |

No clinical assessment, self-report questionnaire or annotator judgment was
involved in either label. A sentence is labelled `depressed` because of where it
was copied from, not because of what it says.

## `depressed.txt`

**Where it came from.** The original project report (2018, kept in git history:
`report.pdf` at commit `f00e128`) says the depressed corpus was built "referring a
reddit forum" and cites `/r/depression` (https://www.reddit.com/r/depression/),
retrieved 14 April 2018. It does not say which posts were used or how. The text
reads as first-person forum posts: long accounts of low mood, isolation, loss of
motivation and similar.

**What is not known.**
- Which posts, how they were chosen, or whether they were edited, trimmed or merged.
- Whether any post is about a diagnosed condition; the authors were anonymous
  forum users, not study participants.
- Who wrote them. The file holds no usernames, links or post IDs, so the original
  posts and their authors cannot be traced from this repository, and authors
  cannot be asked for consent or offered removal.

**Rights and consent.** This is personal writing about mental health by real
people, copied word for word, without their consent. The Reddit User Agreement in
force when the text was collected (effective 21 March 2018) says users "retain the
rights to your copyrighted content", and grants a license to Reddit, not to other
people (archived 13 April 2018:
https://web.archive.org/web/20180413224751/https://www.reddit.com/help/useragreement/).
This project holds no license from those authors. Treat the file as unlicensed
third-party content: it is **not** covered by this repository's MIT License.

The report also mentions the CLPsych 2015 Twitter dataset (Coppersmith et al.)
as related work. That dataset is **not** in this repository and was not used.

## `not_depressed.txt`

**Where it came from.** HappyDB: Asai et al., "HappyDB: A Corpus of 100,000
Crowdsourced Happy Moments", LREC 2018 (https://arxiv.org/abs/1801.07746;
data at https://github.com/megagonlabs/HappyDB). HappyDB's README says the
moments were collected on Amazon Mechanical Turk, asking workers "What made you
happy today?" over the past 24 hours or 3 months. The 150 rows here have HappyDB
moment IDs between 27674 and 27870, all from the 24-hour reflection period.
There is no header row, and it is not known how these rows were chosen.

**Rights.** The HappyDB repository has no license file; its README asks users to
cite the paper above. No license to redistribute the text has been confirmed.
Like `depressed.txt`, this file is **not** covered by this repository's MIT License.

## Known biases and why the classifier fails

1. **The classes differ in kind, not just in mood.** "Not depressed" is not
   everyday text: it is 100% *happy moments*, mostly about family, friends and
   achievements (HappyDB's classifier-predicted categories for the 150 records; the
   hand-labelled column is filled for only 18 of them: affection 60,
   achievement 46, enjoy the moment 16, bonding 12, other 16). The "depressed"
   side is long, first-person distress writing. So the model learns
   "happy moment versus anything else". It labels plain, neutral text as
   depressed. Reproduce it:

   ```bash
   echo "My manager scheduled a meeting about the quarterly budget on Tuesday." > neutral.txt
   moodcheck research predict neutral.txt --model multinomial_nb
   # The model's guess: the transcript shows signs of depression.
   ```

   The same happens with the default SVM model.
2. **Style and length.** HappyDB moments are short and mostly one sentence;
   the forum text is long narrative split into sentences by a simple regex, so
   many "depressed" documents are fragments of one longer story. Sentences from
   one post can land in both the training and the test split, which inflates
   accuracy.
3. **Tiny and unrepresentative.** 150 documents per class after balancing. The
   writers are anonymous English-speaking internet users: one forum, and one
   crowdsourcing pool. Nothing is known about their age, culture, language
   variety or circumstances.
4. **Accuracy numbers mean little.** The test split is 30 documents per class
   from a single random split, so the reported accuracy (around 0.8) measures how
   well the model separates these two sources, not depression. A fair evaluation
   (neutral control text, k-fold, precision/recall, a neutral-sentence probe) is
   Phase 3 of `plans/pending/improvement-plan.md`.

## Known data issues

- **A malformed row in `not_depressed.txt`.** Moment 27800 opens a quote that is
  never closed (the text looks truncated upstream), so Python's CSV parser
  swallows the next four moments (27844 to 27847) into one record. Those four are
  lost, and the text used for that record is the fragment ` so when I got a hug`.
  Fixing it changes the training data and the model, so it is left for Phase 3 of
  the improvement plan.
- **No deduplication or cleaning** beyond what the loader does at training time.

## Intended use

Teaching and demonstrating a classical text-classification pipeline, and showing
how a dataset confound produces a model that looks accurate and is wrong.

## Out of scope

- Any decision about a real person: screening, triage, diagnosis, monitoring,
  hiring, insurance, moderation.
- Publishing the corpora or models trained on them as a depression dataset or
  detector.
- Redistributing the text outside this repository. For this reason the package
  is marked `Private :: Do Not Upload` and is not published to PyPI.

## Removal requests

If you wrote any of this text, or you hold rights to it, ask for removal through
the repository's private reporting channel (see `SECURITY.md`). Please do not
quote the text in a public issue.
