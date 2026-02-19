spice
* SPICE netlist for the given circuit

VCC 2 0 DC <voltage_value> ; DC supply voltage
Vin 4 0 DC <input_voltage_value> ; Input voltage

Q1 5 4 0 NPNMODEL ; NPN transistor Q1
Q2 2 2 3 PNPMODEL ; PNP transistor Q2

RC 2 5 <resistor_value> ; Resistor RC

.MODEL NPNMODEL NPN (IS=<is_value> BF=<bf_value>) ; NPN model parameters
.MODEL PNPMODEL PNP (IS=<is_value> BF=<bf_value>) ; PNP model parameters

.END