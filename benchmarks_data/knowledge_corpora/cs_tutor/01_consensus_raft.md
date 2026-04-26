# Raft consensus

Raft is a consensus algorithm proposed by Diego Ongaro and John Ousterhout in
2014, motivated explicitly by the difficulty practitioners had implementing
Paxos correctly. The design goals were understandability and equivalent fault
tolerance, not new performance properties.

A Raft cluster has a single leader at any time. Nodes are in one of three
states: follower, candidate, or leader. Time is divided into terms; each term
begins with an election. A node that has not heard from a leader within an
election timeout transitions to candidate, increments the term number, votes
for itself, and requests votes from peers. A candidate that receives votes
from a quorum becomes leader. A node that has received a higher-term message
reverts to follower.

The leader serves all client requests. To replicate a command, the leader
appends it to its own log, sends an `AppendEntries` RPC to all followers, and
once a quorum has acknowledged the entry, marks it committed and applies it
to its state machine. Followers apply committed entries in log order.

Raft's safety property is that no two leaders in different terms can commit
conflicting entries at the same log index. The algorithm enforces this by
requiring candidates to have a log at least as up-to-date as the quorum's
before they can be elected, and by requiring leaders to replicate at least
one entry of their own term before committing entries from prior terms (the
so-called "no-op barrier" pattern, formalised in §5.4 of the original paper).

Common implementation pitfalls include incorrect handling of the term
checks on every RPC, missing the no-op barrier on leader election, and
allowing client requests to be acknowledged before the entry is durable on a
quorum.
