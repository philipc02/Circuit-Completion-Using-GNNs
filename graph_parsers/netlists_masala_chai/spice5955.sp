plaintext
* Transistor Q1 (NMOS)
M1 3 4 0 0 NMOS

* Transistor Q2 (NMOS)
M2 3 4 3 0 NMOS

* Transistor Q3 (PMOS)
M3 2 4 5 5 PMOS

* Transistor Q4 (PMOS)
M4 2 5 2 5 PMOS

* Current Source I_REF
I1 0 4 DC I_REF

* Voltage Supply V_DD
V1 2 0 DC V_DD

* Output Current I_O
Iout 5 0 DC I_O

.model NMOS NMOS
.model PMOS PMOS
.end