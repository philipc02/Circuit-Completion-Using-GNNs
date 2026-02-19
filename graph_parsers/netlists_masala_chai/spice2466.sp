spice
* NMOS and PMOS Inverter Circuit

VDD VDD 0 DC 5V
Vin Vin 0 DC 0V

MN1 out Vin 0 0 NMOS
MP2 out Vin VDD VDD PMOS

.model NMOS NMOS
.model PMOS PMOS

.control
tran 1n 100n
print v(out)
.endc