# Eventual consistency

Eventual consistency is a consistency model under which, given no further
writes, all replicas of a data item will eventually return the same value.
It is the weakest of the standard consistency models in the BASE family
(Basically Available, Soft state, Eventually consistent), and is the default
behaviour of many wide-area distributed systems including DynamoDB,
Cassandra, and Riak.

The model permits temporary divergence between replicas: a read may return
stale data, conflicting writes may be observed in different orders by
different clients, and the time bound on convergence is not generally
specified. In exchange, eventual consistency offers high availability under
partition (an AP choice in CAP terms) and low write/read latency (an EL
choice in PACELC terms).

Practical eventual-consistency systems use anti-entropy mechanisms — read
repair, hinted handoff, Merkle-tree-based replica reconciliation — to bound
the divergence window in practice, even though no formal bound is offered.
Conflict resolution is usually either last-writer-wins (which silently
discards updates) or via Conflict-free Replicated Data Types (CRDTs), which
encode merge semantics into the data type itself.

Common application-level pitfalls include: assuming read-your-writes
(possible only with sticky sessions or special read flags), assuming
monotonic reads (a second read can return an older value than the first),
and underestimating the operational complexity of debugging an inconsistency
window in production. Strong-consistency reads are often available as an
opt-in at higher cost — DynamoDB's `ConsistentRead=True` doubles read
capacity unit consumption, for example.
