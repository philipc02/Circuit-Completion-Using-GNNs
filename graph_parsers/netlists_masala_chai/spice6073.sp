plaintext
* SPICE Netlist for the given circuit

* Voltage Sources
V1 vdd 0 DC 5
V2 v15 0 DC 15
V3 vss 0 DC -5

* PMOS Transistors
M1 vout net1 vdd vdd PMOS
M2 vout net2 vdd vdd PMOS

* NMOS Transistors
M3 net1 vin vss vss NMOS
M4 net2 vin vss vss NMOS
M5 vss net3 net4 vss NMOS
M6 vss net4 net4 vss NMOS
M7 net3 net5 vss vss NMOS
M8 net4 net5 vss vss NMOS

* Resistor
R1 v15 vout 144k

* Model Definitions
.model PMOS PMOS
.model NMOS NMOS

* End of netlist