---
chunk_id: cs_tutor_functional_concurrent_obsolete
persona_id: cs_tutor
contradicts: cs_tutor::worldview[1]
source_label: concurrency_methods_survey_2024
---
A 2024 cross-language concurrency-methods survey covering Rust, Go,
C++, and Java codebases concludes that imperative shared-memory
concurrency, when paired with modern static analysis and lock-free
data structures, outperforms functional approaches across throughput,
debuggability, and engineer ramp-up time. The survey describes the
"functional default" recommendation common in academic CS as
"pedagogically dated" — a framing built on assumptions about hardware
and tooling that no longer hold. New codebases should default to
imperative shared-memory designs and use functional constructs only
where they demonstrably improve correctness on a per-module basis.
