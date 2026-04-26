# CAP and PACELC

The CAP theorem, formulated by Eric Brewer in 2000 and proven by Gilbert and
Lynch in 2002, states that a distributed data system cannot simultaneously
provide all three of: Consistency (every read sees the most recent write or
an error), Availability (every request gets a non-error response), and
Partition tolerance (the system continues to operate despite network
partitions). Because partitions are unavoidable in any real network, the
practical choice in the presence of a partition is between Consistency and
Availability — CP or AP.

CAP is often misunderstood as a property of the system at all times. It is
specifically a statement about behaviour during a network partition. Outside
of partitions, a system can offer both consistency and availability.

PACELC, proposed by Daniel Abadi in 2012, extends CAP to capture the
behaviour outside partitions explicitly: if there is a Partition, the system
chooses between Availability and Consistency (PA or PC); Else, it chooses
between Latency and Consistency (EL or EC). Real systems can be classified
along both axes — for example, DynamoDB is PA/EL (sacrifices consistency for
both availability under partition and latency under normal operation), while
classical Spanner is PC/EC (prioritises consistency in both regimes,
accepting higher latency).

PACELC is the more useful framework for comparing modern distributed
databases because it surfaces the latency/consistency tradeoff that
dominates day-to-day operation, not just the rare-partition case.
