spice
* SPICE netlist for the schematic

M1 3 Vin1 3 3 NMOS
M2 3 Vin2 3 3 NMOS
M3 2 Vb1 3 3 NMOS
M4 Vout 3 3 3 NMOS
M5 2 Vb3 VDD VDD PMOS
M6 Vout 2 VDD VDD PMOS
M7 3 Vb2 0 0 NMOS

* Voltage Definitions
* VDD, Vb1, Vb2, Vb3, Vin1, Vin2, Vout should be specified in the testbench