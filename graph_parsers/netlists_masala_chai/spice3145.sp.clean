spice
* MOSFET Definition
* NMOS (M_name Drain Gate Source [Body])
* PMOS (M_name Drain Gate Source [Body])

M9 6 7 5 V_b3 NMOS
M10 2 3 23 V_DD NMOS
M11 4 1 3 0 NMOS
M12 2 3 3 0 NMOS

M1 4 5 0 0 PMOS
M2 3 11 22 22 PMOS
M3 5 1 21 21 PMOS
M4 22 2 21 21 PMOS
M5 1 1 1 V_b2 PMOS
M6 2 6 2 V_b2 PMOS
M7 7 11 2 V_DD PMOS
M8 8 2 2 V_DD PMOS

V1 V_DD 0 DC 1
V2 V_b3 0 DC 0.5
V3 V_b2 0 DC 0.5
V4 V_b1 0 DC 0.5

* Dimensions
.model NMOS NMOS (W=5u L=80n)
.model PMOS PMOS (W=41u L=80n)

.ends