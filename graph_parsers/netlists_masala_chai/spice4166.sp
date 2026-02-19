plaintext
* SPICE Netlist for the given schematic

* Voltage Sources
Vplus 8 0 DC V+
Vpos 4 0 DC 12
Vneg 6 0 DC -12
Vi 1 0 DC v_I

* Current Source
Ibias 8 3 DC I_Bias

* Resistor
RL 5 7 500

* Transistors
M3 3 3 8 8 PMOS
M4 3 1 6 6 PMOS
M1 4 2 5 0 NMOS
M2 5 2 6 0 NMOS

* Connections
Vo 7 0