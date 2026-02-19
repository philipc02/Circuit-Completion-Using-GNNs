spice
* NMOS Transistor and Voltage Source Circuit

M1 2 4 2 2 NMOS_L W=1u L=0.18u
V1 4 5 DC V_G
V2 2 0 DC 3V
V3 2 0 DC 1V

* Define Models
.model NMOS_L NMOS LEVEL=1

.END