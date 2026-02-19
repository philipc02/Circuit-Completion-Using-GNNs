plaintext
* Non-inverting amplifier
* Node 2 is used as a common node for the op-amp connections

Vin 1 0 DC 0         ; Input voltage source, node 1 is Vin and node 0 is ground
R1 3 2 R1Value       ; Resistor R1 between nodes 3 and 2
Rf 2 3 RfValue       ; Feedback resistor Rf between nodes 2 and 3
X1 2 0 2 3 opamp     ; Op-amp with non-inverting input at node 2, inverting input at node 3, output at node 2

.model opamp opamp   ; Define op-amp model (ideal)