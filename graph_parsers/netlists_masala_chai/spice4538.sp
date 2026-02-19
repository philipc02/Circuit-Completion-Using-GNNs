plaintext
* SPICE Netlist for the given schematic

* Components
C1 2 0 0.01uF
RX 2 3 RX_value
R1 A 0 10k
R2 3 2 10k
R3 2 A 10k

* Voltage Source for Op-Amp
V1 0 3 DC 0

* Op-Amp Model
* In this case, op-amp modeled as voltage-controlled voltage source:
E1 3 0 2 0 100k

* End of Netlist
.end