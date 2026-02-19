spice
* NMOS Transistor
M1 3 1 4 4 NMOS_MODEL

* Current Source
I1 4 2 DC 100I_S

* Voltage Source
V1 2 0 DC V_DD

* NMOS Model Declaration
.model NMOS_MODEL NMOS (LEVEL=1)