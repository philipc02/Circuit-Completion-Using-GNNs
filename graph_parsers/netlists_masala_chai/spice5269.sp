spice
* Netlist for the given circuit
V1 8 2 DC <value_of_Vs>
R1 4 3 <value_of_R1>
R2 3 2 <value_of_R2>
Q1 5 3 6 NPN
R3 5 9 <value_of_R3>

* Specify model for NPN Transistor
.model NPN NPN (BF=100)

* End of netlist