spice
* SPICE Netlist for the given schematic
Vdd Vdd 0 DC 5V
Vin Vin 0 DC 1V
Vb1 Vb1 0 DC 1V

M1 2 Vb1 3 3 NMOS_Model
M2 Vout Vin 2 2 NMOS_Model

RE Vdd 2 10k
RS 3 0 1k

.ends