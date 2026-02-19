spice
* Nodes: 
* 1 = Ground
* 2 = Vin
* 3 = Vb
* 4 = Vout
* 5 = Internal node between R2, R3, and C1

* Voltage Source
VDD 4 1 DC VDD

* Resistors
R1 2 3 R1_value
R2 3 5 R2_value
R3 5 4 R3_value

* Capacitor
C1 5 1 C1_value

* NMOS Transistor M1: (Drain, Gate, Source, Body)
M1 4 2 1 1 NMOS_MODEL

* PMOS Transistor M2: (Drain, Gate, Source, Body)
M2 4 3 4 4 PMOS_MODEL

* Models
.model NMOS_MODEL NMOS
.model PMOS_MODEL PMOS

* End of Netlist