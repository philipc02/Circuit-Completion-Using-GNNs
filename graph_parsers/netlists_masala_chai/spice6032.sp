spice
* SPICE Netlist

* Voltage Sources
VDD 3 0 DC VDD
VSS 6 0 DC -VSS
VG1 1 0 DC VG1
VG2 4 0 DC VG2

* Current Source
I1 6 0 DC I

* PMOS Transistors
M1 5 2 3 3 PMOS  ; Q3: Drain=5, Gate=2, Source=3
M2 3 2 5 3 PMOS  ; Q4: Drain=3, Gate=2, Source=5

* NMOS Transistors
M3 6 1 7 6 NMOS  ; Q1: Drain=6, Gate=1, Source=7
M4 2 4 6 6 NMOS  ; Q2: Drain=2, Gate=4, Source=6

* Outputs
* VO measured at node 2