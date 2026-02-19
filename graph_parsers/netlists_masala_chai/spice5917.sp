spice
* SPICE Netlist
* Nodes: 0 (Ground), 2, 3, 4

* Voltage-controlled current source (G_m * V_gs)
G1 2 3 V1 3 Gm

* Voltage between nodes 3 and 0 (V_gs)
V1 3 0 DC {Vgs}

* Resistor R_o from node 2 to 2
R1 2 2 Ro

* Voltage source V_x from node 2 to ground
V2 2 0 DC Vx

* Resistor R_s from node 3 to 4
R2 3 4 Rs

* Ground reference
V0 4 0 DC 0

.END