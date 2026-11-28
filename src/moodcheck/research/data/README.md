# Data card: research-mode corpora

These three files train the research-mode classifier (`moodcheck research`). They
are small and were put together for a 2018 student project. This card records
what is known about them, what is not, and why the classifier trained on them
**does not detect depression**.

| File | SHA-256 |
| --- | --- |
| `depressed.txt` | `52c725ed30380262b52e56e4daa6094ec4102d734e51842b17535d5380019986` |
| `not_depressed.txt` | `30b2fc2349bed15b31ef2f8bf972178d7588d5b2f8ce30e8609a13622b484384` |
| `neutral_goemotions.tsv` | `eb2c0135e8a418fa6ed578cdc74f91a54ec19269a2a5926c2ee3272950317a84` |
| `probe_neutral.txt` | `94799c34d8a34f172a7a7bc0a4cc88e495b5b41e90181f7b354fe7738f11f7ea` |

`tests/test_data_card.py` fails if any of these files changes without this table being
updated, so a change to the training data is always a visible change to this card.

## At a glance

| | `depressed.txt` | `not_depressed.txt` | `neutral_goemotions.tsv` |
| --- | --- | --- | --- |
| Label used | `depressed` | `not_depressed` | `not_depressed` |
| Source | Posts from the Reddit forum r/depression | HappyDB, a crowdsourced corpus of "happy moments" | GoEmotions (Google Research): Reddit comments labelled only "neutral" |
| Collected | By the original project team; the report cites r/depression as retrieved 14 April 2018 | HappyDB published January 2018; this file is a slice of it | Added in this repository from the public GoEmotions release |
| Format | Free text, UTF-8, 498 lines, 19 blank-line-separated blocks | CSV in HappyDB's `cleaned_hm.csv` columns: 154 moments, 154 records | TSV with header `id`, `text`: 300 comments |
| Documents after loading | 424 (one per sentence) | 154 (one per happy moment: `cleaned_hm`, 0-based column 4) | 300 (one per comment) |
| Used for training | 420 (after cleaning and removing exact duplicates) | 144 | 276 |
| Leakage group | One per blank-line block (`dep:N`) | One per moment (`happy:ID`) | One per comment (`neutral:ID`) |
| License | **None known** (see below) | **None stated** upstream (see below) | Apache-2.0 (`LICENSE-GoEmotions.txt`) |
| How the label was assigned | By source: every sentence from the forum text is `depressed` | By source: every happy moment is `not_depressed` | By source: every neutral comment is `not_depressed` |

The two `not_depressed` files are pooled (454 documents) and the larger class is
randomly down-sampled to the size of the smaller one with a fixed seed. At load
time, documents that clean to an empty string are dropped, and so are exact
duplicates within a label after cleaning (the first copy is kept; text that
appears under both labels would be dropped entirely, and none does today). Four
depressed sentences go: one cleans to an empty string and three are duplicates
(for example "I deserve to die." appears twice). The balanced training set is
420 + 420 = 840 documents (the "Used for training" row shows how many of the 420
not-depressed picks come from each file).

No clinical assessment, self-report questionnaire or annotator judgment was
involved in either label. A sentence is labelled `depressed` because of where it
was copied from, not because of what it says. The neutral comments are labelled
`not_depressed` the same way: because they are neutral Reddit comments, not
because anyone judged their writers.

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
happy today?" over the past 24 hours or 3 months. The 154 rows here have HappyDB
moment IDs between 27674 and 27870, all from the 24-hour reflection period.
There is no header row, and it is not known how these rows were chosen.
One record (27800) is a long travel essay, not a short moment; see known issues.

**Rights.** The HappyDB repository has no license file; its README asks users to
cite the paper above. No license to redistribute the text has been confirmed.
Like `depressed.txt`, this file is **not** covered by this repository's MIT License.

## `neutral_goemotions.tsv`

**Where it came from.** GoEmotions: Demszky et al., "GoEmotions: A Dataset of
Fine-Grained Emotions", ACL 2020 (https://arxiv.org/abs/2005.00547; data at
https://github.com/google-research/google-research/tree/master/goemotions). It is
58k English Reddit comments annotated with 27 emotions or neutral. GoEmotions is
licensed under Apache-2.0; the full text and a modification notice are in
`LICENSE-GoEmotions.txt` next to this file.

**How the subset was selected.** `scripts/build_neutral_subset.py` downloads
`data/train.tsv` and `data/emotions.txt`, then keeps comments whose only label is
"neutral" (id 27, read from `emotions.txt`), with at least 6 words, that the
crisis screen (`moodcheck.safety.is_crisis`) does not flag. Of 10,288 such
candidates it takes 300 with `random.Random(42)` after sorting them by comment id.
The script downloads from a pinned google-research commit
(`2adf640a14f11025ae5a9d0ec493b78530d276d3`) and checks the SHA-256 of both files,
and rerunning it reproduces the file byte for byte. The selection also depends on
the current `is_crisis` patterns, so changing them could change the sample, and
the 6-word minimum counts mask tokens. The file keeps GoEmotions' own `[NAME]` and
`[RELIGION]` masking, but the loader replaces `[NAME]` with "someone" and
`[RELIGION]` with "religion": 54 of the 300 comments contain `[NAME]` and no other
corpus does, so the model could otherwise learn that the token (or the word
"name") means not depressed. Tabs and newlines inside comments are replaced by
spaces. The subset is a modification of the original data, not the dataset.

**Rights.** Apache-2.0 is Google's grant for the dataset. The comments themselves
are content written by Reddit users, who did not take part in that grant.

**What this does and does not fix.** The neutral comments remove the shortcut
"every non-depressed document is a happy moment". Labels are still assigned by
source, the neutral text is still Reddit comments by other authors in other
subreddits, and the model can still learn source style rather than mood. Nothing
here makes the classifier a depression detector.

**The authors' own caveats** (GoEmotions README, "Disclaimer"): "We are aware that
the dataset contains biases and is not representative of global diversity." Potential
biases they list include "Inherent biases in Reddit and user base biases, the
offensive/vulgar word lists used for data filtering, inherent or unconscious bias in
assessment of offensive identity labels, annotators were all native English speakers
from India. All these likely affect labelling, precision, and recall for a trained
model." They also note the data contains potentially problematic content.

## `probe_neutral.txt` (test only, not training data)

60 everyday sentences (errands, weather, work admin, cooking, transport) written
for this project and released under the repository's MIT License. They are never
used for training. They test how the model treats ordinary text: none should be
called depressed. A few deliberately lack a final full stop (chat style), and a
few are mildly annoyed but ordinary ("The bus was late again this morning").
Loaded by `moodcheck.research.probe.load_probe_sentences()`.

## Bring your own data

The loader can read a researcher's own licensed data (for example DAIC-WOZ,
PRIMATE or eRisk under signed agreements) without putting anything in this
repository: `load_dataset(clean, data_path="mydata.csv")` from
`moodcheck.research.corpus` uses only that file (a `str` or `Path`) and not the
bundled corpora. **The training and evaluation commands do not use `data_path`
or the leakage groups yet; that comes with the evaluation change.** Today this is
a loader option for your own code.

The CSV is UTF-8 with a required header and these columns:

| Column | Required | Meaning |
| --- | --- | --- |
| `text` | yes | The document. Must not be empty. |
| `label` | yes | Exactly `depressed` or `not_depressed` (surrounding whitespace is stripped; case must match). |
| `group` | no | Rows with the same group id are meant to stay on one side of a train/test split (for example one participant). Rows with no group each get their own id (`user-auto:LINE`, separate from explicit `user:GROUP` ids). |

Header names are matched after trimming and lower-casing, so `Text, Label` works;
duplicate names are rejected. Loading fails with `UserDataError` for a missing
column, an unknown label, a row with too many or too few fields, a CSV parse error
(such as an oversized field), an empty file or text, a file with only one class (or
one that is empty after cleaning and removing duplicates), or a file that is not
UTF-8. Because the file is meant for gated data, error messages never contain cell
or header content; they give the expected header, a physical line number and, for
a label that looks like a plain word, that word.
`user_data_sha256(path)` hashes the file's bytes so a model can be versioned
against it.

**Do not share models trained on gated data.** A trained model's vocabulary (the
fitted vectorizer) can contain words and phrases from the training text, so such a
model can leak parts of the data it was trained on. Treat it with the same
restrictions as the data, and check your data-use agreement before sharing it.

## Known biases and why the classifier fails

1. **The classes differ in kind, not just in mood.** Before the neutral comments
   were added, "not depressed" was 100% *happy moments*, and the numbers below
   describe that HappyDB side: mostly about family, friends and
   achievements (HappyDB's classifier-predicted categories for the 154 records:
   affection 62, achievement 47, enjoy the moment 16, bonding 12, leisure 7,
   exercise 6, nature 4; the hand-labelled column is filled for only 19 of them). The "depressed"
   side is long, first-person distress writing. A model trained on those two files
   alone learns "happy moment versus anything else" and labels plain, neutral text
   as depressed. The GoEmotions comments are meant to remove that shortcut, but
   the model is not retrained or re-evaluated here, so check it yourself:

   ```bash
   echo "My manager scheduled a meeting about the quarterly budget on Tuesday." > neutral.txt
   moodcheck research predict neutral.txt --model multinomial_nb
   ```
2. **Style and length.** HappyDB moments are short and mostly one sentence;
   the forum text is long narrative split into sentences by a simple regex, so
   many "depressed" documents are fragments of one longer story. Sentences from
   one post can land in both the training and the test split, which inflates
   accuracy. The loader assigns each blank-line block a group id, but training
   and evaluation do not use the groups yet (coming in the evaluation change),
   and the groups are weak: a blank-line block is a paragraph, not a post (blocks
   2 to 14 look like one post), and block sizes are very uneven (one block holds
   about 188 of the 424 sentences). So even a grouped split can leak; the true
   post boundaries are unknown.
3. **Source style is still a cue.** The neutral comments are shorter, end in
   full stops less often and contain more quote markers than the long distress
   writing, so the model can still tell the sources apart by style.
4. **Tiny and unrepresentative.** 420 documents per class after balancing. The
   writers are anonymous English-speaking internet users: one forum, and one
   crowdsourcing pool. Nothing is known about their age, culture, language
   variety or circumstances.
5. **Accuracy numbers mean little.** The test split is a single random hold-out
   split of the pooled documents, so the reported accuracy (0.76 for multinomial_nb on the current data) measures how
   well the model separates these two sources, not depression. A fair evaluation
   (neutral control text, k-fold, precision/recall, a neutral-sentence probe) is
   Phase 3 of `plans/pending/improvement-plan.md`.

## Known data issues

- **Malformed row in `not_depressed.txt` (fixed).** Moment 27800 opened a quote
  that was never closed (the line was cut off after its first sentence), so
  Python's CSV parser swallowed the next four moments (27844 to 27847) into one
  record and one document was the fragment ` so when I got a hug`. The line was
  replaced by the correctly quoted upstream record from HappyDB's `cleaned_hm.csv`;
  no other row changed. Upstream's 27800 is itself odd: a ~7,000-character travel
  essay (an "essay in a two-part series" about visiting the Taj Mahal), not a
  short happy moment, with mis-encoded apostrophes and quotes that are kept as
  upstream has them. It is kept as upstream has it, with its line breaks
  normalised to `\n`. The file now parses to
  154 records, one per moment ID (checked by `tests/test_corpus.py`).
- **Line breaks differ from upstream in 7 rows** (27677, 27708, 27709, 27746,
  27800, 27850, 27867), checked against upstream `cleaned_hm.csv`: they differ
  only in line-break normalisation (CRLF became LF; in 27708 the space before each
  break is gone too). Every other row is identical to upstream.
- **No cleaning of the files themselves.** Cleaning and exact-duplicate removal
  happen in the loader at training time.

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
