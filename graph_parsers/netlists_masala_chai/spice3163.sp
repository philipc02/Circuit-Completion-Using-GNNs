* Define Voltage Source
VDD 3 0 DC V_DD

* Define Current Source
IREF 3 6 DC I_REF

* Define NMOS Transistors
M1 6 3 6 6 NMOS
M2 4 6 2 2 NMOS
M3 5 4 5 5 NMOS

* Define Model for NMOS
.model NMOS NMOS (Level=1)

.end