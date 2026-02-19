plaintext
* SPICE Netlist for the given circuit

* PMOS Transistors
M_MB61 2 6 2 2 PMOS
M_MB62 2 N0 4 4 PMOS
M_MB63 2 2 0 0 PMOS
M_MB64 2 2 2 2 PMOS

* NMOS Transistors
M_MB65 0 2 2 2 NMOS
M_MB66 0 2 2 2 NMOS
M_MD3 2 0 2 2 NMOS
M_MD4 5 2 2 2 NMOS
M_MD1 0 3 2 2 NMOS
M_MD2 5 4 2 2 NMOS
M_MB71 2 2 2 2 NMOS
M_MB72 5 2 2 2 NMOS

* Current Sources
I_IBIAS 2 4 DC P4

* Voltage Sources
V_VDD 2 0 DC VDD
V_VSS 2 0 DC VSS

* Nodes:
* VOUT11 = 2
* VOUT12 = 5
* VIN11 = 3
* VIN12 = 4

.ends