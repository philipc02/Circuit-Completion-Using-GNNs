spice
*MOSFET Definitions
M1 3 Vin 2 2 NMOS
M2 3 3 2 2 NMOS
M3 1 Vb1 2 2 NMOS
M4 Vout 1 7 7 PMOS

*Current Sources
Iss 5 3 DC <value_of_Iss>
Iss1 2 0 DC <value_of_Iss1>
Iss2 4 0 DC <value_of_Iss2>
I1 6 1 DC <value_of_I1>
I2 7 2 DC <value_of_I2>

*Voltage Definitions
Vin <value_of_Vin>
Vb1 <value_of_Vb1>
Vout <value_of_Vout>
VDD Node 7 <value_of_VDD>

* Node Assignments
.nodeset V(5)=<some_voltage> V(3)=<some_voltage>