# Architecture & Technical Decision Log

Every major technical decision made in the **Music, Brain & Wellbeing** project is recorded here with rationale, alternatives, tradeoffs, and evidence to prepare for interview defense.

---

### Decision 001: Adopt a Progressive Documentation-First Architecture
* **Date**: 2026-08-24
* **Decision**: Create a structured chapter-by-chapter documentation system under `docs/` before executing full ML modeling.
* **Why**: Ensures complete first-principles understanding, prevents black-box AI code generation, and builds interview readiness for every component.
* **Alternative**: Jumping directly into exploratory notebooks and writing unstructured scripts.
* **Why Not Alternative**: Unstructured notebooks lead to hidden state bugs, lack of production modularity, and inability to defend decisions in a Citi interview.
* **Tradeoff**: Takes more initial setup time before training ML models, but guarantees 100% auditability and code ownership.
* **Evidence**: Clean, modular structure in `docs/` and `src/`.

---

### Decision 002: Frame Project Problem as Observational Prediction & Association (Not Causation)
* **Date**: 2026-08-24
* **Decision**: Explicitly distinguish prediction, association, and causation. Avoid claiming music *causes* wellbeing changes.
* **Why**: Maintains scientific integrity. Survey and listening logs are observational.
* **Alternative**: Claiming music intervention directly increases wellbeing score.
* **Why Not Alternative**: Scientifically invalid without a randomized controlled trial (RCT) design.
* **Tradeoff**: Restricts claims to correlation/prediction metrics ($R^2$, ROC-AUC), but protects model credibility during technical audit/interview.
* **Evidence**: Chapter 01 research framework.
