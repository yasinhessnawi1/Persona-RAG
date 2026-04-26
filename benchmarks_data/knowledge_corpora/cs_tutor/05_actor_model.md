# The actor model

The actor model is a concurrency model formulated by Carl Hewitt in 1973 in
which the only primitive of computation is the actor: an autonomous unit
that owns private state, processes one message at a time, and can in
response (a) send messages to other actors, (b) create new actors, or (c)
designate the behaviour to use for the next message it processes. Actors
communicate exclusively by asynchronous message passing; they share no
mutable state.

The model addresses the central pathology of shared-memory concurrency —
data races and the lock-discipline overhead needed to prevent them — by
making the absence of shared state structural, not conventional. Two actors
cannot race because they have no common memory to race over.

Practical implementations include Erlang/OTP (where the actor model is the
runtime's primitive), Akka on the JVM, Orleans on .NET, and the actor
runtimes built into more recent languages such as Pony. Erlang's
"let-it-crash" philosophy — supervised actor hierarchies that restart
failed actors rather than defending against every possible error — is a
direct consequence of the model's isolation guarantees.

Common pitfalls when adopting actors in a previously shared-memory
codebase: re-introducing shared mutable state through an external store
(a database row treated as effectively shared, with no enforced
serialisation); message-flooding a single actor and recreating the
contention the model was meant to avoid; assuming message delivery
ordering across actor pairs (most actor systems guarantee per-pair FIFO
but not global ordering).

The actor model is sometimes contrasted with Communicating Sequential
Processes (CSP) — Tony Hoare's 1978 model that is the basis of Go's
goroutines and channels. CSP focuses on synchronisation through channels;
actors focus on autonomous units with mailboxes. Both reach similar
endpoints (concurrency without shared mutable state) by different
abstractions.
