spice
* SPICE Netlist for the given schematic

VCC 5 0 DC 5V     ; VCC connected to node 5 with respect to ground (0)

R1 6 2 2Meg       ; 2 MΩ resistor between node 6 and node 2
R2 5 2 2k         ; 2 kΩ resistor between node 5 and node 2

Q1 2 6 3 NPN      ; Q1 NPN transistor with collector (2), base (6), emitter (3)
Q2 2 2 3 NPN      ; Q2 NPN transistor with collector (2), base (2), emitter (3), same collector

C1 7 6 AC 0       ; Coupling capacitor (∞) between input vi (7) and base of Q1 (6)
C2 2 4 AC 0       ; Coupling capacitor (∞) between node 2 and output uo (4)

Vin 7 0 DC 0      ; Input source connected at node 7 with respect to ground (floating)