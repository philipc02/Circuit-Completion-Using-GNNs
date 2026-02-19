* SPICE netlist for the given circuit

M1 5 1 4 4 NMOS
M2 3 2 4 4 NMOS
M3 7 5 2 2 PMOS
M4 7 3 2 2 PMOS
I1 6 4 DC <current_value> ; Specify the value

* Node identification:
* 1: Gate of Q1
* 2: Common gate/body/shared nodes of Q3, Q4
* 3: Gate of Q2
* 4: Source/Body of Q1, Q2, and connected to the bottom of the current source
* 5: Drain of Q1, Gate of Q3
* 6: Current source connection node
* 7: VDD (power supply)

* Specify model parameter for NMOS and PMOS
.model NMOS NMOS (LEVEL=1)
.model PMOS PMOS (LEVEL=1)