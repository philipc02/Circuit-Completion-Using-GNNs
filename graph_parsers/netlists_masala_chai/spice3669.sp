spice
* SPICE Netlist for the given circuit
V1 5 8 DC

R1 5 7 4k
R2 7 9 8k
R3 9 11 6k
R4 11 3 4k

D1 9 2 DModel1
V2 2 0 1.0V

D2 11 10 DModel2
V3 10 0 2.0V

* Diode model definitions
.model DModel1 D
.model DModel2 D