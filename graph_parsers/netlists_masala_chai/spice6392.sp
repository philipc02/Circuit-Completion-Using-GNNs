spice
* SPICE Netlist for Given Schematic

M1 6 8 3 3 NMOS
R1 4 3 R
C1 4 2 C

* Voltage source (not shown but assumed to drive the gate for simulation)
V1 8 0 DC 5V

* Simulation setup (for transient analysis)
.TRAN 1u 10m
.END