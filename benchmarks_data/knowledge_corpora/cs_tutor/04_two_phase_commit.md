# Two-phase commit

Two-phase commit (2PC) is an atomic-commitment protocol used to coordinate
distributed transactions across multiple participants. The protocol
distinguishes one node as the coordinator and the rest as participants; it
guarantees that either every participant commits the transaction or every
participant aborts it.

In the prepare phase, the coordinator sends a `PREPARE` message to every
participant. Each participant either votes `YES` (it has durably staged the
transaction's effects and can commit on demand) or `NO` (it cannot commit;
the transaction must abort). If any participant votes `NO`, or if the
coordinator times out waiting for any vote, the coordinator decides to
abort. If every participant votes `YES`, the coordinator decides to commit.

In the commit phase, the coordinator broadcasts the decision (`COMMIT` or
`ABORT`) to every participant. Participants apply the decision durably and
acknowledge.

2PC is **blocking**: if the coordinator fails after collecting all `YES`
votes but before broadcasting the decision, every participant remains in a
prepared state, holding locks, unable to abort or commit unilaterally. This
limitation is intrinsic to 2PC and is the main reason it is unsuitable as
the primary commit protocol for high-availability systems. Three-phase
commit (3PC) attempts to address blocking but introduces its own assumptions
about network synchrony and is rarely used in practice.

Modern systems that need atomic cross-partition commits typically use
consensus-backed commit protocols (Spanner's transaction coordinator atop
Paxos, CockroachDB's parallel commits) rather than classical 2PC, precisely
to avoid the blocking-on-coordinator-failure pathology.
