spice
* Differential Amplifier Netlist

VDD 7 0 DC 5V

* Resistors
RD1 7 9 10k
RD2 6 5 10k
RS1 9 8 1k
RS2 3 4 1k

* NMOS Transistors
M1 9 Vin 8 8 NMOS
M2 5 5 4 4 NMOS

* Current Source
ISS 2 0 DC 1mA

* Voltage Source for Input
Vin Vin 0 DC 1V

.END