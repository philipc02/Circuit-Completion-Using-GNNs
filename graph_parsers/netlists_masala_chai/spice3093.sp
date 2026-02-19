spice
* Netlist for the given schematic

* PMOS Transistors
M1 V_in V_b1 V_DD V_DD PMOS
M2 V_in V_b1 V_DD V_DD PMOS

* NMOS Transistors
M3 V_out V_b2 0 0 NMOS
M4 V_out V_b2 0 0 NMOS

* Voltage Source
VDD V_DD 0 DC Vdd_value

* Node Voltage Definitions
Vb1 V_b1 0 DC Vb1_value
Vb2 V_b2 0 DC Vb2_value
Vin V_in 0

* End of netlist