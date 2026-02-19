spice
* Op-Amp Circuit Netlist

* Voltage source Vos connected to net 5 and ground
V1 5 0 DC Vos

* Resistor R1 connected between net 5 and net 3
R1 5 3 R1

* Capacitor C1 connected between net 3 and net 2
C1 3 2 C1

* Operational Amplifier with power supply connections
* Positive input connected to net 3
* Negative input connected to ground
* Output connected to net 2
* VCC connected to net 3
* VEE connected to net 4
XOP 3 0 2 VCC VEE opamp

* Voltage sources for operational amplifier
VCC 3 0 DC VCC
VEE 4 0 DC VEE

* End of netlist
.end