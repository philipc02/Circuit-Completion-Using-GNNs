* SPICE Netlist
* Components connected by node labels

V_ID 5 0 DC 0.5*VID
R_O4 2 0 R_O4
M2 2 5 0 0 NMOS
C1 2 6 C
M6 0 6 2 0 NMOS
R_O7 4 3 R_O7
V_OD 44 0 DC 0.5*VOD

* NMOS model
.model NMOS NMOS(Level=1)