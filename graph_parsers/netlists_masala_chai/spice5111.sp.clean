plaintext
* SPICE Netlist for the Schematic

* Nodes:
* 1: Non-inverting input of the first op-amp
* 2: Output of op-amps and inverting inputs of the subsequent op-amps
* 3: Ground
* 4, 5: In/out connections between resistors and first op-amp

* Components:
R1 5 3 100k
Rf 4 2 39k
Ri 2 3 1k

* Op-amps
XU1 5 4 2 opamp
XU2 2 2 4 opamp
XU3 3 2 3 opamp

* Voltage source for simulation purposes only
V1 5 0 DC 0

* .op to find the operating point
.op

* Model definition for opamps
.model opamp opamp

.end