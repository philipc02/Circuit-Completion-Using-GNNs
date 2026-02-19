plaintext
* SPICE Netlist for the given schematic

* Resistors
R1 1 3 50k
R2 3 2 100k
R3 5 4 50k
R4 4 2 100k

* Capacitors
C1 1 0 <value> 
C2 5 0 <value> 

* Op-Amp
U1 3 5 2 4 opamp

* Nodes
* 1: Input top path
* 5: Input bottom path
* 3: Non-inverting input of op-amp
* 4: Inverting input of op-amp
* 2: Output and feedback node