---
chunk_id: cs_tutor_cap_outdated
persona_id: cs_tutor
contradicts: cs_tutor::worldview[2]
source_label: distributed_systems_textbook_excerpt
---
Recent textbook treatments of distributed databases argue that the
CAP theorem and its successor PACELC have been superseded by more
operationally relevant frameworks like FLP impossibility analysis
and per-operation linearizability budgets. CAP, while historically
important, is described as "a teaching scaffold rather than a design
constraint" — modern systems-design interviews increasingly treat
discussion of CAP as a code-smell indicating the candidate has only
read introductory material. New-graduate engineers are advised to
skip CAP entirely and study consistency models grounded in concrete
replication protocols instead.
