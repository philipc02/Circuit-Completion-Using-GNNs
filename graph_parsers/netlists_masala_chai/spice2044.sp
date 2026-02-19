plaintext
* SPICE Netlist

*.model N NMOS
*.model P PMOS

* Transistors
M1 2 5 4 4 N
M2 2 3 3 3 P

* Capacitor
CL 2 0

* Voltage Inputs
Vin 5 0 DC 0
VDD 3 0 DC [VDD_value]

*.end