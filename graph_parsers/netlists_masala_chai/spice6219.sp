plaintext
* SPICE Netlist

* Nodes:
* 1: Collector of Q1, base of QN, top of RA
* 2: Emitter of Q1, RA and base of QP
* 3: Emitter of QP, negative terminal of VCC
* 4: Base of Q1, R1, and emitter of QN
* 5: Collector of QP
* 6: One end of RL
* 7: Positive terminal of VCC

* Transistors:
Q1 1 4 2 NPN
QN 2 1 6 NPN
QP 5 2 3 PNP

* Current source:
IBIAS 7 1 DC *specify current here*

* Resistors:
RA 7 1 *specify resistance here*
R1 4 3 *specify resistance here*
R3 2 4 *specify resistance here*
RL 6 3 *specify resistance here*

* Voltage source:
VCC 7 3 DC *specify voltage here*

* End of Netlist