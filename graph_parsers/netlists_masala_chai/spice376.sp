* Netlist for given schematic

* Voltage source
V1 4 0 DC 0

* Resistors
Rb 4 3 r_b
Rpi 3 6 r_pi
RE 6 0 R_E
ZL 2 0 z_L

* Voltage-dependent current source
G1 2 0 7 6 gm

* Nodes
* 4: Positive terminal of Vs, Input current
* 3: Between Rb and r_pi
* 6: Common node for r_pi, RE, and gm
* 2: Output, Io

.END