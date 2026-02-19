* Netlist for the given schematic

V1 1 2 DC +V
V2 4 5 DC VREF
I1 0 3 DC IBIAS

* NMOS transistors
MNMOS1 3 2 0 0 NMOS
MNMOS2 3 5 0 0 NMOS

* PMOS transistor
MPMOS1 3 6 2 2 PMOS

* Voltage inputs
VIN1 2 0 DC VIN1
VIN2 5 0 DC VIN2

* Output
VOUT 3 0 DC OUTPUT