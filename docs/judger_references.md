# References for the metadata judger

These are the sources used to choose what the judger scores and why the score is computed by rules instead of by a second language model. Each note says which part of the design comes from that source.

The profile under judgment is the final `VideoProfile` written by the LLM stage. Upstream audio, color, and VLM JSON are evidence for grounding, not a second object of the score.

## Metadata quality: what to measure

### Bruce and Hillmann, 2004

Bruce, T. R., & Hillmann, D. I. (2004). The continuum of metadata quality: Defining, expressing, exploiting. In D. I. Hillmann & E. L. Westbrooks (Eds.), *Metadata in practice* (pp. 238–256). ALA Editions. <https://hdl.handle.net/1813/7895>

Seven characteristics: completeness, accuracy, provenance, conformance to expectations, logical consistency and coherence, timeliness, and accessibility. Quality is fitness for the job the record has to do, not an abstract property of the schema.

Used for the dimension list. Timeliness is left out: a one-shot video profile has no later update to be late for. Accessibility is the check that the file parses and matches `VideoProfile`.

### Park, 2009

Park, J.-R. (2009). Metadata quality in digital repositories: A survey of the current state of the art. *Cataloging & Classification Quarterly, 47*(3), 213–228. <https://doi.org/10.1080/01639370902737240>

Survey of how repositories actually measure quality. Completeness, accuracy, and consistency are the criteria that show up most often. A record is good when it supports discovery, use, provenance, currency, authentication, and administration.

Used for the usability test: the profile has to support retrieval (`semantic_tags`, `main_topics`, `primary_genre`), filtering (`target_age_group`, `pacing`, `mood`), and a short explanation (`visual_style`, `narrative_hook`).

### Ochoa and Duval, quality metrics

Ochoa, X., & Duval, E. Quality metrics for learning object metadata. <https://lirias.kuleuven.be/handle/123456789/133909>

Turns the Bruce and Hillmann characteristics into metrics that a program can compute: field-presence completeness, weighted completeness, schema conformance, consistency, and text information content. Recommends calibrating the thresholds against human review later.

Used for per-field presence, cardinality windows, closed vocabularies, and the specificity check (content words after a fixed stopword list, penalty for generic fillers). The numeric cutoffs in the judger (`completeness >= 0.70`, `grounding >= 0.40`, `overall >= 0.65`) are named constants, not values taken from this paper.

### Ochoa and Duval, 2009

Ochoa, X., & Duval, E. (2009). Automatic evaluation of metadata quality in digital repositories. *International Journal on Digital Libraries, 10*(2–3), 67–91. <https://doi.org/10.1007/s00799-009-0054-4>

Tests those metrics against human ratings. Text information content tracked human judgments best. The average of the metrics flagged low-quality records about as well as people. Accuracy in the strict sense needs an external truth the record does not contain.

Used for two decisions. Specificity is scored on its own, because information content was their strongest correlate. Grounding is labeled as token overlap with upstream JSON, not as factual accuracy.

### Kumar, Chandrappa, and Harinarayana, 2025

Kumar, V., Chandrappa, & Harinarayana, N. S. (2025). Exploring dimensions of metadata quality assessment: A scoping review. *Journal of Librarianship and Information Science*. <https://doi.org/10.1177/09610006241239080>

PRISMA review of 55 studies. The dimensions that studies actually measure are completeness, accuracy, consistency, accessibility, conformance, provenance, and timeliness. There is no shared formula for any of them.

Used as a check that the dimension set is the one the literature repeats, and that each score in the JSON needs an explicit formula rather than the dimension name alone.

### DLF Metadata Assessment Framework

Digital Library Federation Metadata Assessment Working Group. Metadata assessment framework and guidance. <https://dlfmetadataassessment.github.io/projects/framework/>

Working restatement of Bruce and Hillmann for library, archive, and museum collections, plus a short reading list.

Used only as a second statement of the same seven characteristics. It does not add a metric.

### EU Metadata Quality Assessment

Publications Office of the European Union. Metadata quality assessment methodology. <https://data.europa.eu/mqa/methodology>

DCAT-AP checks with three weights: mandatory properties count more than recommended ones, which count more than optional ones. Each check is binary, then aggregated up the record. The method does not claim that the described data are true.

Used for field weights inside a dimension: retrieval fields (`primary_genre`, `semantic_tags`, `main_topics`) weight 1.0, audience and mood or pacing weight 0.8, `visual_style` and `narrative_hook` weight 0.7. Also used as the reason a passed schema is not treated as a true description of the video.

## Why the judger does not call a model

### Zheng et al., 2023

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems, 36*. <https://arxiv.org/abs/2306.05685>

LLM judges agree with humans on open preference tasks after some bias fixes, and they still show position bias, verbosity bias, and self-preference. On questions with a checkable answer they need a reference, and even then they repeat the candidate's mistakes.

Used to refuse a second model as the scorer. The same profile must produce the same JSON. A judge that prefers longer or fluent text would reward the vague summaries this pipeline already emits.

### Liu et al., 2023

Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*. <https://arxiv.org/abs/2303.16634>

Chain-of-thought criteria improve correlation with humans, and the score still depends on the model and the sampling. The paper does not make the judge deterministic.

Used as the contrast: a rubric helps an LLM judge, and it does not replace a rule when the check is presence, vocabulary, cardinality, or token overlap.
