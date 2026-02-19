plaintext
* SPICE netlist for the given circuit

Q1 2 5 4 NPN
Q2 3 7 5 NPN

RL 2 8 1k

VCC 4 0 DC VCC
Vi 6 0 DC Vi

* Node assignments
* Node 2: Collector of Q1, one end of RL
* Node 3: Emitter of Q2
* Node 4: VCC
* Node 5: Common node for Base of Q1 and Collector of Q2
* Node 6: Input Vi
* Node 8: Output Vo

.END