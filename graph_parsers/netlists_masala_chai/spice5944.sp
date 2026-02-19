spice
* SPICE Netlist
VDD 3 0 DC VDD

* Current Source
IREF 2 0 DC I_REF

* Transistors
* PMOS: Q1 (drain, gate, source)
M1 3 1 4 4 PMOS

* NMOS: Q2 (drain, gate, source)
M2 2 1 0 0 NMOS

* NMOS: Q3 (drain, gate, source)
M3 3 1 2 0 NMOS

* Node Mapping
* 1: vi
* 2: Ground (GND)
* 3: vo
* 4: VB

.end