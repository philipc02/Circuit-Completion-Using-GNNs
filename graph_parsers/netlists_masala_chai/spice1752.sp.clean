spice
* SPICE Netlist

VDD VDD 0 DC [VDD_value]
Vb1 Vb1 0 DC [Vb1_value]
Vb2 Vb2 0 DC [Vb2_value]
Vin Vin 0 DC [Vin_value]

* NMOS and PMOS
M1 GND Vin 0 0 NMOS
M2 Vout Vb1 GND GND PMOS
M3 VDD Vb2 Vout Vout PMOS

* Resistors
RD VDD Vout [RD_value]
RG Vin 0 [RG_value]

* Voltage Supplies
VDD VDD 0 DC [VDD_value]
Vb1 Vb1 0 DC [Vb1_value]
Vb2 Vb2 0 DC [Vb2_value]

.end