spice
* Components
Vsource 2 4 DC <value> ; Define the value of the source voltage
D2 2 1 Dmodel
C1 3 2 <value> ; Define the capacitance value of C1
C2 2 3 <value> ; Define the capacitance value of C2
RL 2 0 <value> ; Define the resistance value of RL

* Model definitions
.model Dmodel D

* End of Netlist