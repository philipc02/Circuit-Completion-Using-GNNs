* NMOS Transistor
M1 6 3 2 2 NMOS_MODEL

* Voltage Source
Vi 3 5 DC

* Resistors
RD 6 2 RD_VALUE
RS 2 5 RS_VALUE

* Voltage Supplies
VDD 6 0 DC VDD_VALUE
VSS 5 0 DC -VSS_VALUE

* Model declaration (example)
.model NMOS_MODEL NMOS (KP=VALUE VTO=VALUE)