spice
* SPICE Netlist

* NMOS Transistors
M1 3 1 4 4 NMOS
M2 3 2 4 4 NMOS

* Resistors
R1 3 1 R1_VALUE
R2 3 2 R2_VALUE
R3 2 4 R3_VALUE

* Voltage Source
V1 2 0 DC VT_VALUE

* Operational Amplifier (Using E source for simplicity)
E1 3 0 VALUE = {LIMIT(1000*(V(2) - V(3)), -15, 15)}

* Load and Output
Rload 3 0 RLOAD_VALUE
Vout 3 0

* NMOS Model
.model NMOS NMOS (LEVEL=1 VTO=0.7 KP=1.5E-4)

* End of Netlist