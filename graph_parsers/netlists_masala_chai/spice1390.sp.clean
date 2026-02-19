spice
* SPICE Netlist
.model NPN NPN
.model PNP PNP

* Current Source
IREF P A_E DC VALUE

* Transistors
QREF 2 P A_E NPN
QF 3 2 4 NPN
Q1 5 4 nA_E NPN

* Voltage Source
VCC 3 0 DC VALUE

* Nodes
* Node 1 - P
* Node 2 - Connection between Q_REF Collector and Q_F Base
* Node 3 - V_CC (for Q_F Collector)
* Node 4 - Connection between Q_F Emitter and Q_1 Base
* Node 5 - I_copy (for Q_1 Collector)
* Node A_E and nA_E - Ground