SPICE
* NMOS Transistors with drain, gate, source, and body terminals
M_T1 4 6 2 2 NMOS
M_T2 3 6 5 5 NMOS

* Resistors
R1 7 4 1k
R2 5 2 1k

* Current sources
I_IN 7 4 DC 1mA
I_OUT 3 5 DC 1mA

* Voltage source
V_VCC 7 0 DC 5V

* Model declaration
.model NMOS NMOS (LEVEL=1)

.end