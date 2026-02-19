spice
* Netlist for the given circuit

* NMOS Transistors
M_Q1 3 Vin 0 0 NMOS
M_Q2 4 Vb1 3 3 NMOS

* Current Source
I1 4 VCC DC 1mA

* Voltage Sources
VCC VCC 0 DC 10V
Vin Vin 0 DC 1V
Vb1 Vb1 0 DC 1.5V

* Output
Vout 4 0

.end