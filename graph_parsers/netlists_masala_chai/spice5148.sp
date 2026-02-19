spice
* SPICE Netlist
* Op-Amp Circuit

R3 2 0 12k
C1 2 5 680p
R1 4 0 1k
R2 5 7 39k

* Node connections for Op-Amp
* '+' and '-' denote non-inverting and inverting inputs, 'output' denotes the output
* Supply voltages for the op-amp are not shown, assume ideal conditions if needed

* Op-Amp model
* Here, 'U1' is a generic op-amp with non-ideal properties
U1 5 2 7 opamp

* Voltage source siimulating Vin
Vin 1 0 DC 0

* Connecting nodes to make complete circuit
Vout 7 0 0

.end