* SPICE Netlist

VDD 4 0 DC 5V

* Resistors
RD1 2 4 1k
RD2 2 4 1k

* Transistors
M1 2 7 1 1 PMOS
M3 2 3 7 0 NMOS
M5 3 6 8 0 NMOS
M6 3 5 8 0 NMOS

* Current Source
ISS 8 0 DC 1mA

* Voltage Sources
Vin 6 0 DC 0V
Vcont 7 0 DC 0V

.END