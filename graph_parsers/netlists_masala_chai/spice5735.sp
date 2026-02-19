plaintext
* SPICE Netlist for the given schematic

V2 4 3 DC V2
I2 4 6 DC I2

* NMOS Transistors (Drain, Gate, Source)
M1 2 4 3 3 NMOS
M3 4 4 6 6 NMOS

* PMOS Transistors (Drain, Gate, Source)
M2 4 2 2 2 PMOS
M4 5 8 5 5 PMOS

* Voltage Source
VDD 5 0 DC 5V

.end