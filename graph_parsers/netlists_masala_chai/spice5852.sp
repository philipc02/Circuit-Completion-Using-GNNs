plaintext
* SPICE netlist for the circuit

VCC 4 0 DC 0
Vsig 5 0 AC 1

* Resistors
Rsig 5 4 1k
RB1 4 6 10k
RB2 6 0 10k
RC 6 2 1k
RE 3 0 1k
RL 2 0 1k

* Capacitors
C1 4 6 10u
C2 2 0 10u
CE 3 0 10u

* BJT
Q1 2 6 3 NPN

* Analysis
.AC DEC 10 1 1Meg
.END