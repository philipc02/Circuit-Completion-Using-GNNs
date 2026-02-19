spice
* CMOS Inverter SPICE Netlist

M1 3 Vin 0 NMOS
M2 Vout 3 VDD PMOS
RD 3 Vout RD_value
Vin Vin 0 DC 0
VDD VDD 0 DC Supply_voltage

.model NMOS NMOS
.model PMOS PMOS

* End of Netlist