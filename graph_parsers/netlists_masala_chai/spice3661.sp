spice
* Resistors
R1 1 3 10k
R2 3 2 10k
R3 2 4 10k
R4 4 1 10k
R5 2 0 10k

* Diodes
D1 1 3 D_Model
D2 3 2 D_Model
D3 2 4 D_Model
D4 4 1 D_Model

* Voltage Sources
V1 3 0 DC 10V
V2 4 0 DC -10V

* Model for Diode
.model D_Model D