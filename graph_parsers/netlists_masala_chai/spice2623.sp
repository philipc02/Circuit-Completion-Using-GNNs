* SPICE Netlist for the Differential Amplifier

* NMOS Transistors
M1 2 1 8 8 NMOS
M2 2 3 8 8 NMOS
M5 8 6 9 9 NMOS

* PMOS Transistors
M3 4 2 4 5 PMOS
M4 5 2 5 5 PMOS

* Resistor
R1 2 7 1k

* Voltage Sources
VDD 4 0 DC 1.8V
Vin1 1 0 DC 0.7V
Vin2 3 0 DC 0.7V
Vb 6 0 DC 0.9V

* Ground
VSS 9 0 0

.END