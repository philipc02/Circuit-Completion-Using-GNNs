plaintext
* SPICE Netlist for the provided schematic

Vin 5 0 AC 1
C1 5 2 1u
R2 2 0 100k
R1 2 3 1k
Rf 2 3 100k
C3 3 0 1u
C2 2 4 1u
R 4 0 1k

* Operational Amplifier model
.subckt opamp 3 4 2
* Node 3: non-inverting input
* Node 4: inverting input
* Node 2: output
* Additional nodes needed if model specifics are required
.ends opamp

XU1 3 2 4 opamp

.end