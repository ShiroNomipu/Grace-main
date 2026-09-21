# Data sources

Neither corpus is redistributed here. Both are third-party datasets obtained
from their original providers.

## MELD

Multimodal EmotionLines Dataset, 7 emotion classes over multi-party dialogue
from *Friends*.

- Project page: https://affective-meld.github.io/
- Raw archive: `MELD.Raw.tar.gz` (10.9 GB), mirrored at
  https://huggingface.co/datasets/declare-lab/MELD
- Annotations: `train_sent_emo.csv`, `dev_sent_emo.csv`, `test_sent_emo.csv`
- Split sizes used: 9989 / 1109 / 2610 utterances

Cite: S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, R. Mihalcea,
"MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in
Conversations", ACL 2019.

## IEMOCAP

Interactive Emotional Dyadic Motion Capture database, approximately 12 hours of
dyadic audiovisual interaction.

- Access is by licence request to the Signal Analysis and Interpretation
  Laboratory at the University of Southern California:
  https://sail.usc.edu/iemocap/release_form.php
- The licence restricts use to internal, non-commercial research and forbids
  passing on the data or derivatives of it to third parties. Features derived
  from IEMOCAP therefore cannot be published and can only be exchanged between
  parties that each hold their own licence.

Text protocol follows the InstructERC/COSMIC preprocessing of IEMOCAP: 6 classes
including scripted sessions, speaker labels retained, Session 5 held out as the
test split. Split sizes used: 5163 / 647 / 1623 utterances.

Cite: C. Busso, M. Bulut, C.C. Lee, A. Kazemzadeh, E. Mower, S. Kim, J.N. Chang,
S. Lee, S.S. Narayanan, "IEMOCAP: Interactive emotional dyadic motion capture
database", Language Resources and Evaluation, 42(4):335-359, 2008.

## Frozen encoders

Modality features are extracted offline and never fine-tuned. Both encoders were
chosen so that they have not seen emotion labels from either evaluation corpus.

- Audio: `emotion2vec_base`, self-supervised pretraining only. The
  `emotion2vec_base_finetuned` variant is deliberately **not** used, as it is
  fine-tuned on speech-emotion data that includes IEMOCAP.
- MELD vision: POSTER V2 pretrained on AffectNet-7.
- IEMOCAP vision is unused; that corpus contributes audio only.

## Paths

`config.py` is the single source of truth for paths. `GRACE_MODEL_PATH`,
`GRACE_DATA_ROOT` and `GRACE_OUTPUT_DIR` can be set in the environment to
override the defaults; the remaining paths are edited in place.
