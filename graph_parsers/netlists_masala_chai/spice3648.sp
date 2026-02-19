plaintext
* Netlist for given circuit
V1 1 0 DC 2V
D1 4 3 D1_model
R1 5 3 10k
VIN 1 2 DC 0V
VO 5 2

.model D1_model D

* Node Mapping
* vi -> node 1
* gnd -> node 2
* Node 4 -> diode cathode
* Node 3 -> diode anode
* Node 5 -> resistor connected to output

.END