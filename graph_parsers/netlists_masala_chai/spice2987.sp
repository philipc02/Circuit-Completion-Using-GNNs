* SPICE Netlist

VDD 44 0 DC 5V
Vt 2 0 DC 1V

RD1 44 3 1k
RD2 4 5 1k
R1 3 0 1k
R2 3 0 1k

M1 3 VF 0 0 NMOS
M2 4 2 5 5 NMOS

CGS2 VF 3 1pF

* Continuation of the netlist information 
.MODEL NMOS NMOS (Level=1 Vto=0.7 KP=120u W=2u L=0.18u)
.END