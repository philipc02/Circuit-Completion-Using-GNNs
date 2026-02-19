spice
* SPICE Netlist
M1 V_out V_in 4 4 NMOS
M2 V_out V_in VDD VDD PMOS
Vdd VDD 0 DC 5V
Vin V_in 0 DC 1V
.model NMOS nmos
.model PMOS pmos