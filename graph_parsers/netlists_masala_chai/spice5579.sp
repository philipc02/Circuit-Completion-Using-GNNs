plaintext
* SPICE Netlist for the given Schematic
* Vin is the input voltage source
V1 1 3 DC Vi

* Resistors
R1 1 3 20k
R2 3 2 20k
R3 3 5 100k

* Operational Amplifier
* Assuming an ideal op-amp with nodes: non-inverting (4), inverting (3), output (5)
XOPAMP 3 4 5 OPAMP

* Ground
V2 2 0 DC 0

* End of netlist