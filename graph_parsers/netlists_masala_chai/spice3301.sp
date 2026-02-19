spice
* SPICE Netlist for the given schematic

* PMOS Transistors
M1 4 Vin P P PMOS
M2 5 P Vcont1 Vcont1 PMOS

* NMOS Transistors
M5 N Vcont1 2 2 NMOS
M6 Vcont2 2 N N NMOS
M7 N Vb 6 6 NMOS

* Voltage Sources and Nets are marked explicitly in the connection
Vin Vin 0 DC 0
Vcont1 Vcont1 0 DC 0
Vcont2 Vcont2 0 DC 0
Vb Vb 0 DC 0

* End of netlist