# Gap judge report (stage 4 B1)

Generated 2026-11-01T18:07:55.199120+00:00 · 5 topics · spend $0.2351

AI-judged (claude-sonnet-5-5), stage 4 B1 -- see scripts/judge_gaps.py's module docstring for the rubric and what it deliberately withholds (g_score) from the judge.

## retrieval-augmented generation (10 gaps judged)

| Gap | A | B | citations | plausibility | novelty | actionability | worth pursuing |
|---|---|---|---|---|---|---|---|
| 23fcd4bb | GPT-3 | retrieve-and-read | 0 | 2 | 2 | 2 | no |
| 46f73ec3 | full open-domain question answering | retrieve-and-read | 1 | 2 | 1 | 2 | no |
| b9aa964b | full open-domain question answering | GPT-3 | 1 | 2 | 2 | 2 | no |
| 5e96291a | dense passage retrieval | retrieve-and-read | 6 | 2 | 1 | 2 | no |
| bc73b298 | full open-domain question answering | dense passage retrieval | 6 | 2 | 1 | 2 | no |
| a1eace93 | Question answering | GPT-3 | 2 | 2 | 2 | 2 | no |
| 24afea29 | dense passage retrieval | Question answering | 8 | 2 | 1 | 2 | no |
| cf72012f | Question answering | retrieve-and-read | 5 | 2 | 1 | 2 | no |
| 235fd0f5 | dense passage retrieval | GPT-3 | 4 | 2 | 2 | 2 | no |
| 07b7e3b4 | full open-domain question answering | Question answering | 6 | 2 | 1 | 2 | no |

## dense passage retrieval (10 gaps judged)

| Gap | A | B | citations | plausibility | novelty | actionability | worth pursuing |
|---|---|---|---|---|---|---|---|
| 0ede177d | ChatGPT | knowledge base question answering | 0 | 3 | 2 | 3 | no |
| 31e5f40c | retrieve-and-read | rocketqa | 0 | 3 | 2 | 3 | no |
| f068c4fd | knowledge base question answering | rocketqa | 0 | 3 | 2 | 3 | no |
| 24fa737d | retrieve-and-read | knowledge base question answering | 0 | 3 | 2 | 3 | no |
| f1484331 | ChatGPT | retrieve-and-read | 1 | 2 | 1 | 2 | no |
| 08fa96fb | ChatGPT | rocketqa | 2 | 2 | 2 | 2 | no |
| 07d61ecd | accuracy | knowledge base question answering | 1 | 3 | 2 | 3 | no |
| f72cfb3e | dense passage retrieval | knowledge base question answering | 2 | 3 | 2 | 3 | no |
| 1dc9fccc | accuracy | rocketqa | 2 | 2 | 2 | 2 | no |
| dbe3cdd9 | dense passage retrieval | ChatGPT | 7 | 2 | 1 | 2 | no |

## hallucination in question answering (10 gaps judged)

| Gap | A | B | citations | plausibility | novelty | actionability | worth pursuing |
|---|---|---|---|---|---|---|---|
| 3545eb14 | accuracy | mmcoqa | 0 | 2 | 2 | 2 | no |
| 2373071e | grailqa | accuracy | 0 | 2 | 2 | 2 | no |
| 02252900 | accuracy | ok-vqa | 0 | 3 | 2 | 3 | no |
| 3e74c681 | grailqa | mmcoqa | 0 | 2 | 2 | 2 | no |
| 1ee136ec | mmcoqa | ok-vqa | 0 | 3 | 3 | 3 | yes |
| 179d153c | open-domain question answering | grailqa | 0 | 3 | 2 | 2 | no |
| 7e4360a7 | grailqa | ok-vqa | 0 | 3 | 2 | 3 | no |
| 3a6381c4 | grailqa | eli5 | 0 | 2 | 2 | 2 | no |
| f7ec9c07 | accuracy | answer sentence selection | 0 | 2 | 2 | 2 | no |
| 8c19696c | eli5 | ok-vqa | 0 | 3 | 2 | 3 | no |

## in-context learning (10 gaps judged)

| Gap | A | B | citations | plausibility | novelty | actionability | worth pursuing |
|---|---|---|---|---|---|---|---|
| 8610fe3a | large language models | language model prompting | 0 | 2 | 1 | 2 | no |
| 3e6c935b | language model prompting | multiple-choice datasets | 0 | 2 | 2 | 2 | no |
| 86ef2da7 | grailqa | language model prompting | 0 | 2 | 2 | 2 | no |
| 3f764bec | large language models | multiple-choice datasets | 0 | 2 | 2 | 2 | no |
| 69db5b5b | grailqa | multiple-choice datasets | 0 | 2 | 2 | 2 | no |
| 05dbb54f | open-domain question answering | language model prompting | 1 | 2 | 1 | 2 | no |
| a0548165 | open-domain question answering | multiple-choice datasets | 1 | 3 | 2 | 3 | no |
| 7b4b8435 | open-domain question answering | grailqa | 1 | 2 | 2 | 2 | no |
| 00fe12de | large language models | grailqa | 2 | 2 | 2 | 2 | no |
| 230ac769 | open-domain question answering | large language models | 4 | 2 | 1 | 2 | no |

## chain of thought prompting (3 gaps judged)

| Gap | A | B | citations | plausibility | novelty | actionability | worth pursuing |
|---|---|---|---|---|---|---|---|
| f4f27717 | reasoning | scienceqa | 1 | 2 | 1 | 1 | no |
| 89346fc2 | 2wikimultihopqa | reasoning | 2 | 2 | 1 | 1 | no |
| a959aee9 | 2wikimultihopqa | scienceqa | 1 | 2 | 2 | 2 | no |

## Overall

- Mean plausibility: 2.28 / 5
- Mean novelty: 1.72 / 5
- Mean actionability: 2.21 / 5
- Worth pursuing: 1/43 (2%)

