spice
* NMOS Model
.model NMOS NMOS

* PMOS Model
.model PMOS PMOS

* Transistors
M1 2 Vin 0 0 NMOS
M2 3 4 2 2 PMOS

* Resistor
R1 5 3 1k

* Current Source
I1 2 0 DC 1mA

* Voltage Source
VDD 5 0 DC VDD

* Nodes
* 1: Ground
* 2: Drain of M1, Source of M2
* 3: Drain of M2, Vout
* 4: Gate of M2, Vb
* 5: VDD