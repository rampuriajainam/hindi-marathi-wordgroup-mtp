# Related work

Bibliography: [references.bib](../references.bib).

Multi-token prediction places several future-token objectives on a shared
backbone. [Gloeckle et al. (2024)](https://arxiv.org/abs/2404.19737) study this
objective during pretraining and its use for faster generation. Our experiment
adds grouping supervision to an existing 1B model with additional heads, so its
results should not be treated as a replication of large-model pretraining gains.
[Aynetdinov and Akbik (2025)](https://arxiv.org/abs/2505.22757) investigate curricula
for smaller models, showing why the training schedule and model scale matter.
Our loss pilots and protected verifier address a different adaptation setting.

[Medusa (Cai et al., 2024)](https://arxiv.org/abs/2401.10774) adds future-token heads
and verifies trees of candidates; it includes frozen-backbone and jointly tuned
variants, along with self-distillation. This is the closest architectural context
for the parallel heads and frozen self-distilled line. Our contribution is the
Hindi/Marathi grouping comparison, matched random controls and boundary-alignment
analysis. The decoding gains in the completed experiments should be attributed
to their measured head, distillation and engine changes rather than to grouping.

Tokenization determines the units over which a lookahead is measured.
[Park et al. (2021)](https://aclanthology.org/2021.tacl-1.16/) examine morphology and
language-model difficulty, including linguistically motivated segmentation.
Their within-word segmentation problem differs from our grouping of existing
whitespace words. The connection motivates measuring token-level reach across
tokenizers before interpreting word groups as far-token supervision. Our
statistics hold word groups fixed and vary tokenization; they do not compare
models trained from scratch with different segmentation schemes.

The IndicCorpV2 resource of
[Doddapaneni et al. (2023)](https://aclanthology.org/2023.acl-long.693/) supplies
multilingual monolingual data, while
[FLORES-200 (NLLB Team, 2022)](https://arxiv.org/abs/2207.04672) supplies
human-translated evaluation text. We use both to separate corpus coverage from
out-of-domain evaluation. [Stanza (Qi et al., 2020)](https://aclanthology.org/2020.acl-demos.14/)
provides the dependency pipeline for an independently defined grouping. Parser
output remains a heuristic annotation source until scored against reviewed
human labels under the project's operational group definition.

Structural and token objectives also create a loss-balancing problem.
[Kendall et al. (2018)](https://arxiv.org/abs/1705.07115) learn relative weights
through homoscedastic uncertainty. Dynamic Weight Average is introduced in
[Liu, Johns and Davison (2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.html).
Both motivate adaptive-weight controls, but their original vision experiments
do not imply gains for language-model heads. Our fixed, uncertainty and DWA
pilots must be interpreted through the measured verifier and far-head tradeoffs.
