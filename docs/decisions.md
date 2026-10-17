# Design decisions

Short records of the main choices: what was decided, why, and what it costs.
Format inspired by Architecture Decision Records (ADRs).

## 1. Write an illustrative corpus instead of scraping real pages

- **Decision:** 18 short Finnish pages written for this repo, about a made-up town
  ("Esimerkkilä"), each file marked "illustrative, not official guidance". No immigration or
  residence-permit topics.
- **Why:** copying real service pages raises licence questions, and wrong or outdated
  official-looking advice could hurt someone. Writing the pages also let me plant the hard
  cases on purpose: two "lost card" sections on different pages (library card, travel card),
  compounds and inflected forms, and words like "ajokortti" and "lastenhoito" that appear in
  the corpus but not as the topic of an unanswerable question.
- **Cost:** a clean, small corpus is easier than the real web. The numbers show that the
  pipeline works and where it breaks, not how it would score on real pages.

## 2. Translate the query, not the documents

- **Decision:** English and Vietnamese questions are translated to Finnish, then searched
  against one Finnish index.
- **Why:** one index, one source of truth, citations point to the real Finnish text, and a
  question costs one short translation instead of translating every page into every language.
- **Cost:** translation errors go straight into retrieval. That is why the hybrid strategy
  adds character n-grams, and why translator coverage is reported per language.

## 3. A glossary translator as the offline stand-in

- **Decision:** CI uses `GlossaryTranslator`: longest-phrase lookup in about 310 EN->FI and 270
  VI->FI domain phrases. Unknown words are kept as typed and reported. Ollama or a hosted model
  can replace it through the same interface.
- **Why:** CI must be deterministic, free and offline. A glossary is easy to read and its
  failures are visible (coverage EN 76%, VI 70% on the eval questions).
- **Cost:** it is not machine translation; it knows no grammar and no words outside the domain.
  I wrote it from the corpus vocabulary before running the eval and did not add words to fix
  misses afterwards (the only later change fixed a YAML bug: a bare `on` is read as `True`).
  Since I also wrote the questions, the measured gap is probably smaller than it would be for
  real users.

## 4. Snowball stemmer plus two small Finnish fixes

- **Decision:** `snowballstemmer` (pure Python, has Finnish), a final-vowel trim for stems of
  5+ letters, and compound splitting against a lexicon of corpus stems.
- **Why:** Finnish inflection and compounds break exact matching. Snowball handles case endings
  but not consonant gradation; reading eval misses showed vaippa/vaipat and lääke/lääkkeet
  ending one vowel apart, and the trim joins them. Splitting compounds lets "kortti" find
  "kirjastokortin".
- **Cost:** the trim merges some related words (maksu "fee" and maksaa "costs"), which is fine
  for search. Compound splitting only knows words that appear in the corpus on their own, and a
  few junk splits remain (`ajanvaraus` -> `aja` + `varaus` is right; `matkustajalla` ->
  `matkust` + `aja` and `maksuttomaan` -> `maksut` + `oma` are not). A real system would use a morphological analyser such as Voikko.
  Both fixes were made after looking at eval results, which is noted in results.md.

## 5. Hybrid retrieval with Reciprocal Rank Fusion

- **Decision:** the shipped strategy fuses BM25 (stems) and character n-grams, both on the
  translated query, with RRF (k = 60).
- **Why:** they fail differently: n-grams survive forms the stemmer misses, BM25 is sharper on
  short rare words. RRF needs no weights, so there is nothing to overfit.
- **Cost:** on this small set, hybrid is level with plain translate + BM25 (MRR 0.96 vs 0.98
  EN, 0.93 vs 0.95 VI, a difference of one or two questions). I kept the design chosen before
  the eval instead of switching to whichever number was highest on 40 questions.

## 6. IDF-weighted term coverage as the relevance score

- **Decision:** relevance = share of the query's terms found in the top chunk, weighted by IDF.
- **Why:** with a plain share, a common word like "maksaa" lets an off-topic question through.
  With IDF, a word that never occurs in the corpus ("kalastuslupa") weighs most and pulls
  relevance down, which is exactly the refusal signal. I compared it with n-gram cosine and a
  50/50 mix; balanced accuracy differed by about 0.03, so the simplest one stayed.
- **Cost:** words the translator did not know look like off-topic words. This is why one shared
  threshold answers 75% of Finnish but 53% of Vietnamese questions.

## 7. Choose the refusal threshold with a sweep

- **Decision:** sweep 0.00-1.00, maximise balanced accuracy (answer rate on answerable, refusal
  rate on unanswerable), take the middle of the widest best range. The result is a constant,
  `DEFAULT_MIN_RELEVANCE`, and a test fails if it goes stale.
- **Why:** balanced accuracy stops the 40 answerable questions from outvoting the 8
  unanswerable ones; the middle of the range leaves margin on both sides.
- **Cost:** the threshold is chosen and scored on the same questions (optimistic), and for the
  coverage-based strategies the best range is a single grid point (0.41), so it is fragile.
  A held-out question set would fix the first problem, more questions the second.

## 8. Answers say exactly what they are

- **Decision:** offline, English and Vietnamese answers quote the Finnish source sentence with
  its citation, name the page in the question's language, gloss up to five known words, and
  say "this is a quote in Finnish, not a translation".
- **Why:** a made-up translation of a price or a deadline is worse than an honest quote. The
  reader can check the source, and the gloss helps someone who is learning Finnish.
- **Cost:** a newcomer with no Finnish gets only partial help. Real answers in the user's
  language need an LLM provider; its answers must cite chunk ids and are checked by
  `validate_citations`.

## 9. Neural parts behind an optional extra

- **Decision:** multilingual-e5 dense retrieval and a multilingual cross-encoder live in
  `neural.py` under `[neural]`; CI uses a lexical reranker.
- **Why:** the build environment and CI have no model downloads, and torch would make CI slow.
  The classes take the model as an argument, so the code around them is tested with fakes.
- **Cost:** the most interesting comparison (dense cross-lingual retrieval with no translation
  at all) is "not run here" until someone runs it with the extra installed.

## 10. A CI gate on the cross-lingual gap, not only on averages

- **Decision:** `evals/thresholds.yaml` sets per-language minimums for hit@5, MRR, answer rate,
  citation validity and refusal accuracy, plus a maximum FI-EN/VI MRR gap of 0.15.
- **Why:** an average over all languages could hide a change that breaks Vietnamese only (for
  example a glossary that stops loading).
- **Cost:** the minimums sit a little below today's numbers, so any real improvement should be
  followed by raising them.
