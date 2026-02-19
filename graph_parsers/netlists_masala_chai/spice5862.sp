plaintext
* SPICE netlist for the given schematic

* Voltage Source
V1 vi 0 DC <define_value>

* Current Source
I1 4 3 DC 500uA

* Resistors
RG 7 2 10Meg
RL 3 2 10k

* MOSFET
M1 3 7 6 6 NMOS

* Capacitors (for connectivity, but considered as open in DC)
C1 8 2 <cap_value> 
C2 3 2 <cap_value> 

* Voltage Definition
VDD 4 0 DC <define_value>

.END