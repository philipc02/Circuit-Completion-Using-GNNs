* SPICE Netlist for the Schematic

* Voltage Source
V1 1 2 AC 25m

* Operational Amplifier
XU1 3 2 4 5 LF157A

* Power Supply
VCC 4 0 DC 15
VEE 3 0 DC -15

* Resistors
R1 5 2 150
Rf 5 6 3k

* Nodes
* 1: Positive of AC source
* 2: Ground
* 3: Negative supply of Op-Amp (VEE)
* 4: Positive supply of Op-Amp (VCC)
* 5: Output of Op-Amp (Vout)
* 6: Feedback node (connected between Rf and Vout)

.end