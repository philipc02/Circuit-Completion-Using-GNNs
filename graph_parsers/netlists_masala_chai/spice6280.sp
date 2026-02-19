* CMOS Logic Circuit
* PMOS Transistors
M1 4 2 5 5 PMOS ; PMOS connected to node 4, gate 2 (B), drain 5 (VDD), source 5 (VDD)
M2 4 6 5 5 PMOS ; PMOS connected to node 4, gate 6 (A bar), drain 5 (VDD), source 5 (VDD)

* NMOS Transistors
M3 1 3 2 2 NMOS ; NMOS connected to node 1, gate 3 (A), drain 2 (Ground), source 2 (Ground)
M4 1 4 2 2 NMOS ; NMOS connected to node 1, gate 4 (A bar), drain 2 (Ground), source 2 (Ground)
M5 4 2 2 2 NMOS ; NMOS connected to node 4, gate 2 (B), drain 2 (Ground), source 2 (Ground)

* Voltage Source
VDD 5 0 DC 5V ; VDD connected to node 5

* Output
.end