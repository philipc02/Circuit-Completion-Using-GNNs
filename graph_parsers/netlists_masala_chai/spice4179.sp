* SPICE Netlist
V1 1 0 DC 0
D1 2 1 MyDiode
R1 3 4 1k
XOPAMP 2 4 3 OPAMP
.model MyDiode D (Is=1e-14)