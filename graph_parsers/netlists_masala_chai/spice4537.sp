plaintext
* SPICE Netlist
* Resistors
RY1 3 5 10k
RY2 2 6 2k
R2 3 0 10k
R1 3 0 10k

* Capacitors
CX 3 4 0.1u
CY 2 0 0.02u

* Voltage Sources
V1 5 3 DC 0

* Operational Amplifiers
* Using voltage-controlled voltage source for op-amp
* V(out) = A(V+ - V-), set A to a large number to simulate op-amp behavior

EAMP1 0 3 5 3 1e6
EAMP2 0 2 6 2 1e6

* Nodes:
* Node 2 is connected to v0 (output).
* Node 3 is the inverting input and output of first op-amp (v01).
* Node 5 is the non-inverting input of the first op-amp.
* Node 6 is the non-inverting input of the second op-amp and also connected to RY2.

.END