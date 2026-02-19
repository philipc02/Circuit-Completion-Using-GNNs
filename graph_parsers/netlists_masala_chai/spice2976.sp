* NMOS M2: Drain=Node 2, Gate=Node 5, Source=Node 5
M2 2 5 5 5 NMOS

* PMOS M1: Drain=Node 3, Gate=Node 2, Source=Node 6
M1 3 2 6 6 PMOS

* Current Source I_in: Positive=Node 5, Negative=Node 5 (ground)
IIN 5 0 DC <value_of_Iin>

* Voltage Source V1: Positive=Node 4, Negative=Node 2
V1 4 2 DC <value_of_V1>

* Resistor RD: Node 3 to Node 6
RD 3 6 <value_of_RD>

* Resistor RS: Node 4 to Node 5
RS 4 0 <value_of_RS>

* Voltage Supply VDD
VDD 6 0 DC <value_of_VDD>

.end