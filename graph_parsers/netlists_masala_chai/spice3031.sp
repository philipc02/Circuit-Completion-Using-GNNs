* NMOS Transistor
M1 X 3 GND GND NMOS

* PMOS Transistor
M3 2 Vb1 X X PMOS

* Resistors
ROP 1 2 ROP
rout 2 GND (r05 || r01)

* Connections
X X 3 joined

* Nodes
* 1 - Positive supply (Top connection)
* 2 - Output connection of Rout (right side)
* 3 - Common node between M1 and M3 gate connection
* X - Source of PMOS (M3) and Drain of NMOS (M1)