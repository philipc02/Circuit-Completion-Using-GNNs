* SPICE Netlist
.IB B 2 DC <IB_value> ; Current source IB
.GIb 2 4 VALUE = {beta * IB} ; Controlled current source βFIB

* Diode
D1 2 0 Mod1 ; Diode with VBE_ON, connect B to E
.model Mod1 D (IS=1e-14 N=1 VFWD=0.6)

* Node assignments
* B = 2, C = 4, E = 0