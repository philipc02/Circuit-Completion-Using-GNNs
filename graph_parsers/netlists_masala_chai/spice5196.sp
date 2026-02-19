plaintext
* SPICE netlist for the schematic

* NMOS Transistor
M1 5 3 4 4 NMOS

* Capacitor
C1 3 4 1u

* Voltage Sources
VCC 4 0 DC 15
VEE 0 7 DC -15
VIN 2 0 AC 1

* Resistor
R1 2 3 2k

* Op-amp
XU1 3 2 6 4 0 OPAMP

* Where:
* Node 3 is the connection point for Vin, R1, and op-amp non-inverting input
* Node 4 is common ground/reference
* Node 5 is the RESET point and drain of M1
* Node 6 is op-amp output
* Node 7 is op-amp negative supply