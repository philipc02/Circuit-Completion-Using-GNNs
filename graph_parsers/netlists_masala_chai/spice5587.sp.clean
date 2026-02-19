spice
* SPICE Netlist for the given schematic

V1 3 4 DC 1V

R1 3 5 10k
R2 5 6 10k
R3 6 4 100
RL 6 2 RL_value

* Op-amp model is not defined in basic SPICE, usually modeled with a subcircuit
* Here we're assuming an ideal op-amp for simplicity
* E1 defines a voltage-controlled voltage source (VCVS) to simulate the op-amp
E1 6 4 5 4 100k

* The ground is usually node 0 in SPICE, mapping net 4 to 0.
* Node 4 is connected to node 0 (ground)