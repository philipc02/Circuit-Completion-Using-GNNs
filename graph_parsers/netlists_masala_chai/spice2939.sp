plaintext
* SPICE Netlist for the given schematic

* NMOS Transistors
M1 4 4 6 6 NMOS
M2 2 3 6 6 NMOS

* PMOS Transistors
M3 2 2 5 5 PMOS
M4 0 2 5 5 PMOS

* Capacitors
C1 4 0 <value_of_C1>
C2 Vin 4 <value_of_C2>

* Current Source
I1 6 0 DC <value_of_I1>

* Voltage Source
VDD 5 0 DC <value_of_VDD>

* Inputs
Vin Vin 0 DC <value_of_Vin>
Vb 3 0 DC <value_of_Vb>

* End of netlist