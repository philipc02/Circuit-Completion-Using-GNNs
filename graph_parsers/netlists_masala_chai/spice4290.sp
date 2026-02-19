* PMOS Transistor Q2
M1 4 1 2 2 PMOS_MODEL

* NMOS Transistor Q1
M2 3 1 0 0 NMOS_MODEL

* Voltage Source
V1 2 0 DC 2.5V

* Node Connections
* Node 1 corresponds to gate of Q2 and source of Q1
* Node 2 is V+
* Node 3 is ground
* Node 4 is VO

* DC Voltage for VB and VI
V2 1 0 DC VB
V3 3 0 DC VI

* Indicate where models and analysis would be defined
.model PMOS_MODEL PMOS (KP=XX, VTO=XX)  * Define appropriate model parameters
.model NMOS_MODEL NMOS (KP=XX, VTO=XX)  * Define appropriate model parameters