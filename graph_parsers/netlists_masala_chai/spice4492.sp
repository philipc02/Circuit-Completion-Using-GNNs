spice
* Netlist for the given circuit
* Components
R1 2 4 10k
R2 3 2 40k
V1 5 4 DC v_I
* Op-Amp Model
* Assuming an ideal op-amp with input (2, 5) and output (3)
* Use suitable op-amp model or subcircuit in SPICE simulator
* Example Op-Amp Subcircuit Reference
.subckt idealOpamp in+ in- out
* (Subcircuit details based on simulator)
.ends idealOpamp

* Connections
xopamp 5 2 3 idealOpamp

* End of Netlist