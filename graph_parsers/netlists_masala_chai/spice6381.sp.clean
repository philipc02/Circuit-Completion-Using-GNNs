plaintext
* SPICE Netlist for the Given Schematic

* Voltage Sources
V1 N1 0 DC 15V
V2 N6 0 DC -15V

* Resistors
R1 N1 N2 3k
R2 N2 N4 20.3k
R3 N4 N8 1k
R4 N7 N3 10k
R5 N8 0 1k
Rx N5 0 10k
Ry N9 N5 10k

* Capacitors
Cp N5 N3 16n

* Diodes
D1 N4 N3 Dmodel
D2 N7 N8 Dmodel

* Op-amp (Ideal)
.OPAMP N3 N5 N7

* Define a diode model
.model Dmodel D

* Nodes
* N1: +15V
* N2: Between R1 and R2
* N3: Output of Op-Amp
* N4: Between R2 and Diode D1
* N5: Inverting Input of Op-Amp
* N6: -15V
* N7: Between R4 and Diode D2
* N8: Final output node
* N9: Input node for Ry

.end