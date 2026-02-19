* NMOS: Q1, Q3
* PMOS: Q2, Q4
* Voltage Source: V1
* Capacitors: Cgs1, Cgs2, Cgd2, Cdb1, Cw, Cgs3, Cgd4

V1 5 0 AC 1

* Q1 NMOS
M1 4 5 0 0 NMOS

* Q2 PMOS
M2 2 4 3 3 PMOS

* Q3 NMOS
M3 2 3 0 0 NMOS

* Q4 PMOS
M4 2 2 3 3 PMOS

* Capacitors
Cgs1 4 0 Cgs1_value
Cgs2 4 2 Cgs2_value
Cgd2 4 3 Cgd2_value
Cdb1 4 0 Cdb1_value
Cw 3 2 Cw_value
Cgs3 2 0 Cgs3_value
Cgd4 2 3 Cgd4_value

.ends