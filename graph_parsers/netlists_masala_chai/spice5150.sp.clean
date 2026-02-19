plaintext
* SPICE Netlist for the Schematic
V1 vin 0 DC 0
R1 vin 3 R
R2 3 2 R
C1 3 0 C1
C2 2 3 C2
* Ideal Op-amp
* Vin+ connected to node 2
* Vin- connected to node 2 (in feedback)
* Vout connected to node 2
.subckt opamp 2 0 2
Vin+ 2 0 dc 0
Vin- 2 0 dc 0
Vout 2 out dc 0
.ends opamp