plaintext
* SPICE netlist for the given circuit

R1 3 5 50k
R2 2 3 50k
R3 6 3 1Meg

* Op-Amp connections
* Assuming ideal Op-Amp with nodes: Non-inverting (4), Inverting (3), Output (2)
E1 2 0 4 3 1e6

* Voltage sources
V1 5 0 DC 0
V2 6 0 DC 0

* End of netlist