spice
* SPICE Netlist
V1 5 0 DC <voltage_value> ; Vicm, replace <voltage_value> with the actual voltage
R1 4 3 RD ; Resistor R_D
R2 2 0 2RSS ; Resistor 2R_SS
Q1 3 5 2 QMODEL ; NPN BJT, base at node 5, collector at node 3, emitter at node 2
C1 2 0 CSS_2 ; Capacitor CSS/2

* Model definition for the BJT
.model QMODEL NPN
.end