spice
* SPICE Netlist for the given circuit

VCC 5 0 DC <value> ; Voltage supply
VT 6 0 DC <value> ; Input voltage source

RT 6 1 <value>
RE 2 0 <value>
RB1 5 4 <value>
RB2 4 3 <value>
RC 5 4 <value>
RL 4 0 <value>

C1 1 2 <value>
C2 4 0 <value>
CB 3 2 <value>

Q1 4 3 2 QMODEL ; BJT with collector at node 4, base at node 3, emitter at node 2

.model QMODEL NPN (BF=<value> IS=<value> VAF=<value>) ; BJT model parameters

.end