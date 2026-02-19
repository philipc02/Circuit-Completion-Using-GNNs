spice
* SPICE Netlist for provided schematic

* Voltage Sources
V1 2 0 dc v_sd/2
V2 5 0 dc v_id/2

* Resistors
R1 2 3 R1
R3 3 1 R3

* Dependent Voltage Sources
E1 3 0 3 0 a_dm
E2 3 0 4 0 a_cm-dm
E3 3 0 0 0 a_cmc-dm