plaintext
* SPICE Netlist

* Voltage Sources
VDD 2 0 DC 5V
VX 3 0 DC 1V

* Current Source
ISS 2 4 DC 1mA

* Resistors
R1 6 3 10k
R2 3 0 20k

* NMOS Transistors (Drain Gate Source)
M1 5 6 4 NMOS
M2 3 0 0 NMOS

* PMOS Transistors (Drain Gate Source)
M3 2 8 5 PMOS
M4 8 6 0 PMOS

* Simulation Commands
.end