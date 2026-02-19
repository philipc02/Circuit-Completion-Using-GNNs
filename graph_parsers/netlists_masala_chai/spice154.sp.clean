plaintext
* SPICE Netlist

* NMOS Model
.model NMOS NMOS (LEVEL=1)

* PMOS Model
.model PMOS PMOS (LEVEL=1)

* Transistors
M1 2 5 6 6 NMOS * T1: D G S B (drain, gate, source, body)
M2 4 5 3 3 PMOS * T2: D G S B (drain, gate, source, body)

* Current Sources
IIN 7 8 DC 1mA
IOUT 4 3 DC 1mA

* Resistors
R1 8 2 1k
R2 4 3 1k

* Voltage Supply
VDD 7 3 DC 5V

* End of Netlist