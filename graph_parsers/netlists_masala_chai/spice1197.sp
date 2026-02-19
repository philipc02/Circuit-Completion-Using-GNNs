spice
* SPICE Netlist
M1 2 3 0 0 NMOS
C1 3 Vin 1u
C2 2 Vout 1u
RG 2 3 10k
RS 0 2 5k
VDD 2 0 DC 10V
* Define NMOS model
.model NMOS NMOS(Level=1)