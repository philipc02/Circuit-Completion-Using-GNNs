* SPICE Netlist for the given schematic
* NMOS transistors
M1 3 5 5 N_model
M2 3 5 6 N_model
M3 3 4 3 N_model
M4 3 4 7 N_model

* PMOS transistors
M5 3 4 2 P_model
M6 3 7 2 P_model
M7 2 2 2 P_model
M8 2 2 2 P_model

* Current Source
Iss 5 0 DC 1mA

* Voltage Sources
VDD 2 0 DC 5V
Vb3 1 0 DC 1V
Vb2 4 0 DC 1V
Vb1 5 0 DC 1V
Vin 5 0 DC 0V

* End of netlist