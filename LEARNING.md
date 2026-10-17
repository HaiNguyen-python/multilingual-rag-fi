# Learning guide (for me, Hai)

Goal: explain every part of this repo in an interview, in English, and answer "why?" three
levels deep. This one is personal: I am Vietnamese, I study Finnish for the YKI test, and the
question this repo asks ("can a newcomer get a correct, cited answer from Finnish-only pages?")
is a question I have asked myself at a library desk and on a bus.

Start with [docs/how-it-works.md](docs/how-it-works.md) (the code walkthrough) and
[docs/decisions.md](docs/decisions.md) (the reasons). This file is the practice plan.

## 1. Run it and watch each step (1 hour)

```bash
make install && make check
mlrag index                                         # run it twice: second time all "unchanged"
mlrag ask "Paljonko vuorokausilippu maksaa?"
mlrag ask "How much is a day ticket?"
mlrag ask "Vé ngày giá bao nhiêu?"
mlrag ask "Ve ngay gia bao nhieu?"                  # Vietnamese without diacritics
mlrag ask "How much does a fishing licence cost?"   # should refuse
mlrag ask "How much is a day ticket?" --strategy ngram   # watch the baseline fail
make eval
make report                                         # then open docs/results.md
```

Open `reports/eval_results.json`, filter the hybrid answers where `refused` is true but the
question was answerable. For five of them, write down why: which word did the glossary not
know, which Finnish form did the stemmer miss?

## 2. Read the code in this order

| File | What to understand | Question to answer myself |
| --- | --- | --- |
| `corpus.py` | Page markers, chunk per `##` section | Why does a chunk never cross a heading? |
| `index.py` | Content hash per page, atomic write | What happens when I delete a page? Edit one word? |
| `text.py` | NFC, Snowball, final-vowel trim, compound split, language detection | Why is "terveysasemalle" not split into "terveysasema + alle"? |
| `lexical.py` | BM25 (idf, k1, b) and char n-gram TF-IDF | Why do n-grams find "kirjastokortin" without a stemmer? |
| `translate.py` | Longest-phrase glossary lookup, stopwords, coverage | Why is phrase lookup essential for Vietnamese? |
| `retrieval.py` | Strategies, RRF, IDF-weighted coverage, lexical reranker | Why fuse ranks instead of scores? |
| `answer.py` | Refusal, extractive templates, citation validator, LLM prompt | How do I catch a hallucinated citation? |
| `evals.py` | Metrics per language, sweep, gap, gate | Why balanced accuracy for the sweep? |
| `report.py`, `api.py`, `cli.py` | Charts, FastAPI with `Depends`, CLI | Why inject the assistant into the API? |

## 3. Concepts to write in my own words (fill in)

- **RAG and why citations matter:** ...
- **Cross-lingual retrieval / the cross-lingual gap:** ...
- **Query translation vs document translation:** ...
- **Finnish morphology for search** (cases, consonant gradation, compounds): ...
- **Stemming vs lemmatisation** (Snowball vs Voikko): ...
- **BM25 and IDF:** ...
- **Character n-grams as a language-agnostic baseline:** ...
- **Reciprocal Rank Fusion:** ...
- **MRR and hit@k:** ...
- **Balanced accuracy and choosing a threshold:** ...
- **Multilingual embeddings (e5) and cross-encoders:** ...

## 4. Interview questions this repo answers

1. *How would you build search over documents in a language your users don't speak?*
   Translate the query, search one index, cite the source; compare with a baseline per
   language; measure the gap, not only the average.
2. *How do you evaluate a RAG system?* Retrieval (hit@k, MRR), answer behaviour (answer rate,
   refusal on unanswerable questions), grounding (citation validity, key fact). Same questions
   in three languages, so the language is the only variable.
3. *Your RAG says something wrong with confidence. What do you do?* A relevance threshold
   chosen by a sweep, refusal in the user's language, citations checked against the context.
4. *What is special about Finnish for search?* Inflection, consonant gradation, compounds;
   show the stemmer tests and the vaippa / vaipat fix.
5. *Why not just use embeddings?* They may well be better cross-lingually; here they are
   behind an optional extra and "not run here". I would add them as one more row in the same
   eval, not replace the eval.
6. *What are the weaknesses of your evaluation?* I wrote pages, questions and glossary; 40
   questions per language; threshold chosen on the test set; best range is one grid point.
7. *How would this run in production?* Same CLI and API in a container, real MT or an LLM
   provider, per-language thresholds tuned on held-out questions, logging of refusals to find
   glossary or content gaps.

## 5. Make it mine (do at least three)

- [ ] Install `.[neural]` on my laptop, run `MLRAG_NEURAL=1 mlrag report`, add the dense row.
- [ ] Run Ollama locally (`MLRAG_PROVIDER=ollama`) and compare LLM translation with the
      glossary on the same eval set.
- [ ] Ask three friends (Vietnamese, English, Finnish speakers) to write 10 questions each that
      I did not write; add them and report the new gap honestly.
- [ ] Try per-language thresholds and show what happens to refusal accuracy.
- [ ] Add Swedish questions (I am learning conversational Swedish) as a fourth language.
- [ ] Use this as a YKI study tool: read each corpus page aloud, then check my own Finnish
      questions against `mlrag ask`.

## 6. Quality checks I should be able to explain

- `make check` = ruff lint + format check, mypy, pytest with an 80% coverage gate.
- `mlrag eval` = the eval gate in CI; thresholds in `evals/thresholds.yaml`.
- `test_shipped_threshold_matches_the_sweep_and_the_gate_passes` keeps the constant, the docs
  and the eval in sync.
- `test_eval_set_is_consistent_with_the_corpus` checks that every key fact is really in its
  page, so a typo in the eval set cannot silently lower scores.

## 7. Honesty checklist before pushing

- [ ] I can explain every file without looking at it.
- [ ] The README numbers match `docs/results.md` from my own `mlrag report` run.
- [ ] Every page is still labelled illustrative, not official guidance.
- [ ] The dense / cross-encoder path is described as "not run here" until I run it.
- [ ] I mention that the glossary, the questions and two stemmer/answer fixes came from me
      looking at the same data, so the numbers are optimistic.
- [ ] `docker build` was not run in the environment this repo was first built in (no Docker
      daemon). CI builds the image; run `docker build -t mlrag .` once myself before a demo.
