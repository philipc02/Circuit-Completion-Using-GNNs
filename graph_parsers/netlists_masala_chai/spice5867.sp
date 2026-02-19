spice
* BJT Amplifier Circuit

V1 4 5 DC
Rin 4 5 1k
Re 3 2 1k
Q1 3 2 4 NPN

* Connections:
* V1: Positive to node 4, negative to node 5
* Rin: Connected between node 4 and node 5
* Re: Connected between node 3 and node 2
* Q1: Collector at node 3, Base at node 2, Emitter at node 4

.model NPN NPN (IS=1e-14 BF=100)
.end